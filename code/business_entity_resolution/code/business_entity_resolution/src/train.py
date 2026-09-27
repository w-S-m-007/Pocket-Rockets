"""
Business Entity Resolution — Model Training
=============================================
Trains a GradientBoosting classifier on training data ground truth.
Uses rapidfuzz string-similarity features + scikit-learn.

Usage:
    python src/train.py

Output:
    model.pkl  — serialised classifier + optimal F0.5 threshold
"""

import csv
import os
import re
import sys
import random
import pickle
import time
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score

# ─────────────────────────── Configuration ───────────────────────────

TRAIN_DIR = r"C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\train"
MODEL_PATH = "model.pkl"
SAMPLE_S1 = 60000          # S1 entities to sample for training
NEG_PER_POS_ENTITY = 5     # negative pairs per S1 entity (with matches)
NEG_SAMPLE_RATE = 0.005    # probability of keeping a random S2/S3 record for neg pool
RANDOM_SEED = 42

# ─────────────────────────── Feature names ───────────────────────────

FEATURE_NAMES = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_jw", "name_jaccard",
    "addr_ratio", "addr_partial_ratio", "addr_token_sort_ratio",
    "addr_jw", "addr_jaccard", "name_len_diff",
]

# ─────────────────────────── Normalisation ───────────────────────────

_CORPORATE_SUFFIXES = frozenset({
    "inc", "incorporated", "llc", "llp", "corp", "corporation",
    "ltd", "limited", "co", "company", "pvt", "private",
    "plc", "gmbh", "sa", "sarl", "sas", "ag", "nv", "bv",
    "lp", "pllc", "dba",
})

def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    tokens = name.split()
    tokens = [t for t in tokens if t not in _CORPORATE_SUFFIXES]
    return " ".join(tokens).strip()


# ─────────────────────────── Feature extraction ──────────────────────

def _token_jaccard(s1: str, s2: str) -> float:
    t1, t2 = set(s1.split()), set(s2.split())
    union = t1 | t2
    if not union:
        return 0.0
    return len(t1 & t2) / len(union)


def extract_features(name1: str, addr1: str, name2: str, addr2: str) -> list:
    """Return a list of 12 float features for a candidate pair."""
    n1 = name1.lower().strip() if name1 else ""
    n2 = name2.lower().strip() if name2 else ""
    a1 = addr1.lower().strip() if addr1 else ""
    a2 = addr2.lower().strip() if addr2 else ""

    # ── name features ──
    if n1 and n2:
        name_ratio      = fuzz.ratio(n1, n2) / 100.0
        name_partial     = fuzz.partial_ratio(n1, n2) / 100.0
        name_tsort       = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_tset        = fuzz.token_set_ratio(n1, n2) / 100.0
        name_jw          = JaroWinkler.normalized_similarity(n1, n2)
        name_jaccard     = _token_jaccard(n1, n2)
    else:
        name_ratio = name_partial = name_tsort = name_tset = name_jw = name_jaccard = 0.0

    # ── address features ──
    if a1 and a2:
        addr_ratio       = fuzz.ratio(a1, a2) / 100.0
        addr_partial     = fuzz.partial_ratio(a1, a2) / 100.0
        addr_tsort       = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_jw          = JaroWinkler.normalized_similarity(a1, a2)
        addr_jaccard     = _token_jaccard(a1, a2)
    else:
        addr_ratio = addr_partial = addr_tsort = addr_jw = addr_jaccard = 0.0

    # ── length feature ──
    name_len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)

    return [
        name_ratio, name_partial, name_tsort, name_tset, name_jw, name_jaccard,
        addr_ratio, addr_partial, addr_tsort, addr_jw, addr_jaccard,
        name_len_diff,
    ]


# ─────────────────────────── Main training routine ───────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    t0 = time.time()

    # ── 1. Load ground truth ──
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    print(f"[1/7] Loading ground truth from {gt_path} ...")
    gt = {}                           # s1_id → list[matched_id]
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            matched_str = row["matched_entity_ids"].strip()
            gt[s1_id] = matched_str.split(",") if matched_str else []
    print(f"    {len(gt)} S1 entities in ground truth")
    n_with    = sum(1 for v in gt.values() if v)
    n_without = sum(1 for v in gt.values() if not v)
    print(f"    {n_with} with matches, {n_without} singletons")

    # ── 2. Sample S1 entities ──
    print(f"[2/7] Sampling {SAMPLE_S1} S1 entities ...")
    with_match_ids    = [k for k, v in gt.items() if v]
    without_match_ids = [k for k, v in gt.items() if not v]
    n_pos_sample = min(int(SAMPLE_S1 * 0.6), len(with_match_ids))
    n_neg_sample = min(SAMPLE_S1 - n_pos_sample, len(without_match_ids))
    sampled_pos = set(random.sample(with_match_ids, n_pos_sample))
    sampled_neg = set(random.sample(without_match_ids, n_neg_sample))
    sampled_s1 = sampled_pos | sampled_neg
    print(f"    {len(sampled_pos)} with matches, {len(sampled_neg)} singletons")

    # ── 3. Collect needed entity IDs ──
    needed_s1 = set(sampled_s1)
    needed_s23 = set()
    for s1_id in sampled_pos:
        needed_s23.update(gt[s1_id])
    print(f"    Need {len(needed_s1)} S1 records + {len(needed_s23)} matched S2/S3 records")

    # ── 4. Stream source files ──
    records = {}   # entity_id → (name, addr, country)
    neg_pool = defaultdict(list)   # country → list[(eid, name, addr)]

    # Source 1
    s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv")
    print(f"[3/7] Streaming {s1_path} ...")
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            if eid in needed_s1:
                records[eid] = (
                    row.get("business_name", ""),
                    row.get("business_address", ""),
                    row.get("country", ""),
                )
    print(f"    Collected {len(records)} S1 records")

    # Source 2
    s2_path = os.path.join(TRAIN_DIR, "train_source2.tsv")
    print(f"[4/7] Streaming {s2_path} ...")
    count = 0
    with open(s2_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            name = row.get("business_name", "")
            addr = row.get("business_address", "")
            country = row.get("country", "")
            if eid in needed_s23:
                records[eid] = (name, addr, country)
            if random.random() < NEG_SAMPLE_RATE:
                neg_pool[country].append((eid, name, addr))
            count += 1
    print(f"    Scanned {count} S2 records, neg pool sizes: {', '.join(f'{k}={len(v)}' for k, v in neg_pool.items())}")

    # Source 3
    s3_path = os.path.join(TRAIN_DIR, "train_source3.tsv")
    print(f"[5/7] Streaming {s3_path} ...")
    count = 0
    with open(s3_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            name = row.get("business_name", "")
            addr = row.get("business_address", "")
            country = row.get("country", "")
            if eid in needed_s23:
                records[eid] = (name, addr, country)
            if random.random() < NEG_SAMPLE_RATE:
                neg_pool[country].append((eid, name, addr))
            count += 1
    print(f"    Scanned {count} S3 records, neg pool sizes: {', '.join(f'{k}={len(v)}' for k, v in neg_pool.items())}")

    # ── 5. Build training pairs ──
    print(f"[6/7] Building training pairs + extracting features ...")
    X, y = [], []
    pos_count = neg_count = skip_count = 0

    for s1_id in sampled_pos:
        if s1_id not in records:
            skip_count += 1
            continue
        s1_name, s1_addr, s1_country = records[s1_id]

        # Positive pairs
        for m_id in gt[s1_id]:
            if m_id not in records:
                skip_count += 1
                continue
            m_name, m_addr, _ = records[m_id]
            feat = extract_features(s1_name, s1_addr, m_name, m_addr)
            X.append(feat)
            y.append(1)
            pos_count += 1

        # Negative pairs: random S2/S3 from same country
        true_set = set(gt[s1_id])
        pool = neg_pool.get(s1_country, [])
        if pool:
            n_neg = min(NEG_PER_POS_ENTITY, len(pool))
            neg_samples = random.sample(pool, n_neg)
            for neg_eid, neg_name, neg_addr in neg_samples:
                if neg_eid in true_set:
                    continue
                feat = extract_features(s1_name, s1_addr, neg_name, neg_addr)
                X.append(feat)
                y.append(0)
                neg_count += 1

    print(f"    Positives: {pos_count}, Negatives: {neg_count}, Skipped: {skip_count}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"    Feature matrix shape: {X.shape}")

    # ── 6. Train / validate split ──
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"    Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # ── 7. Train classifier ──
    print("[7/7] Training GradientBoosting classifier ...")
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    # ── 8. Optimise threshold for F0.5 ──
    y_proba = clf.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f05 = 0.0
    results = []
    for thr in np.arange(0.30, 0.96, 0.01):
        y_pred = (y_proba >= thr).astype(int)
        f05 = fbeta_score(y_val, y_pred, beta=0.5, zero_division=1.0)
        prec = precision_score(y_val, y_pred, zero_division=1.0)
        rec  = recall_score(y_val, y_pred, zero_division=1.0)
        results.append((thr, f05, prec, rec))
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thr

    print("\n    Threshold optimisation (top 10):")
    results.sort(key=lambda x: -x[1])
    for thr, f05, prec, rec in results[:10]:
        marker = " <<<" if abs(thr - best_threshold) < 0.005 else ""
        print(f"      thr={thr:.2f}  F0.5={f05:.4f}  P={prec:.4f}  R={rec:.4f}{marker}")

    # ── 9. Feature importance ──
    print("\n    Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"      {name:30s}  {imp:.4f}")

    # ── 10. Save model ──
    model_data = {
        "classifier": clf,
        "threshold": float(best_threshold),
        "feature_names": FEATURE_NAMES,
        "f05_val": float(best_f05),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    elapsed = time.time() - t0
    print(f"\n[OK] Model saved to {MODEL_PATH}")
    print(f"  Best threshold: {best_threshold:.2f},  Val F0.5: {best_f05:.4f}")
    print(f"  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
