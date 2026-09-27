"""
Business Entity Resolution — Memory-Efficient Evaluation
=========================================================
Two-pass approach:
  Pass 1: Compute blocking keys for eval S1 entities
  Pass 2: Stream S2/S3, keeping ONLY records whose blocking keys overlap

This keeps memory under 2GB instead of 8GB+.
"""

import csv
import os
import re
import sys
import time
import pickle
import random
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

TRAIN_DIR = r"C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\train"
MODEL_PATH = "model.pkl"
EVAL_S1_COUNT = 30000
RANDOM_SEED = 99
MAX_BLOCK_SIZE = 500

_CORPORATE_SUFFIXES = frozenset({
    "inc", "incorporated", "llc", "llp", "corp", "corporation",
    "ltd", "limited", "co", "company", "pvt", "private",
    "plc", "gmbh", "sa", "sarl", "sas", "ag", "nv", "bv",
    "lp", "pllc", "dba",
})

def normalize_name(name: str) -> str:
    if not name: return ""
    name = re.sub(r"[^a-z0-9\s]", " ", name.lower())
    return " ".join(t for t in name.split() if t not in _CORPORATE_SUFFIXES).strip()

def generate_blocking_keys(norm_name: str, country: str) -> list:
    keys = []
    c = country.strip().lower() if country else ""
    if not norm_name: return keys
    if len(norm_name) >= 3:
        keys.append(f"{c}|p5|{norm_name[:5]}")
    tokens = norm_name.split()
    if tokens:
        sorted_tok = "_".join(sorted(tokens)[:3])
        keys.append(f"{c}|st|{sorted_tok[:25]}")
    if tokens:
        longest = max(tokens, key=len)
        if len(longest) >= 5:
            keys.append(f"{c}|lw|{longest}")
    return keys

def _token_jaccard(s1: str, s2: str) -> float:
    t1, t2 = set(s1.split()), set(s2.split())
    union = t1 | t2
    return len(t1 & t2) / len(union) if union else 0.0

def extract_features(name1, addr1, name2, addr2):
    n1 = name1.lower().strip() if name1 else ""
    n2 = name2.lower().strip() if name2 else ""
    a1 = addr1.lower().strip() if addr1 else ""
    a2 = addr2.lower().strip() if addr2 else ""
    if n1 and n2:
        nr = fuzz.ratio(n1, n2) / 100.0
        np_ = fuzz.partial_ratio(n1, n2) / 100.0
        nts = fuzz.token_sort_ratio(n1, n2) / 100.0
        nte = fuzz.token_set_ratio(n1, n2) / 100.0
        njw = JaroWinkler.normalized_similarity(n1, n2)
        nj = _token_jaccard(n1, n2)
    else:
        nr = np_ = nts = nte = njw = nj = 0.0
    if a1 and a2:
        ar = fuzz.ratio(a1, a2) / 100.0
        ap = fuzz.partial_ratio(a1, a2) / 100.0
        ats = fuzz.token_sort_ratio(a1, a2) / 100.0
        ajw = JaroWinkler.normalized_similarity(a1, a2)
        aj = _token_jaccard(a1, a2)
    else:
        ar = ap = ats = ajw = aj = 0.0
    nld = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
    return [nr, np_, nts, nte, njw, nj, ar, ap, ats, ajw, aj, nld]


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    t0 = time.time()

    # 1. Load ground truth
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    print("[1/7] Loading ground truth ...")
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            matched_str = row["matched_entity_ids"].strip()
            gt[s1_id] = matched_str.split(",") if matched_str else []
    print(f"    {len(gt)} total S1 entities")

    # 2. Sample evaluation set
    print(f"[2/7] Sampling {EVAL_S1_COUNT} S1 entities ...")
    eval_s1_ids = set(random.sample(list(gt.keys()), min(EVAL_S1_COUNT, len(gt))))
    eval_gt = {s1: gt[s1] for s1 in eval_s1_ids}
    n_with = sum(1 for v in eval_gt.values() if v)
    n_without = len(eval_s1_ids) - n_with
    print(f"    {n_with} with matches, {n_without} singletons")

    # Collect GT match IDs (to check blocking recall)
    gt_match_ids = set()
    for matches in eval_gt.values():
        gt_match_ids.update(matches)
    print(f"    {len(gt_match_ids)} unique GT match IDs")

    # 3. Load S1 records
    s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv")
    print("[3/7] Loading S1 records ...")
    s1_records = {}
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            if eid in eval_s1_ids:
                s1_records[eid] = (
                    row.get("business_name", ""),
                    row.get("business_address", ""),
                    row.get("country", ""),
                )
    print(f"    {len(s1_records)} S1 records loaded")

    # 4. Compute blocking keys for all eval S1 entities
    print("[4/7] Computing S1 blocking keys ...")
    s1_block_keys = set()
    s1_key_map = {}  # s1_id -> list of keys
    for s1_id in eval_s1_ids:
        if s1_id not in s1_records:
            continue
        name, addr, country = s1_records[s1_id]
        nn = normalize_name(name)
        keys = generate_blocking_keys(nn, country)
        s1_key_map[s1_id] = keys
        s1_block_keys.update(keys)
    print(f"    {len(s1_block_keys)} unique S1 blocking keys")

    # 5. Stream S2/S3: only keep records with matching blocking keys OR in GT
    print("[5/7] Streaming S2+S3 (memory-efficient) ...")
    entity_records = {}   # eid -> (name, addr, country)
    block_index = defaultdict(set)  # key -> set of eids

    for src_name in ["train_source2.tsv", "train_source3.tsv"]:
        src_path = os.path.join(TRAIN_DIR, src_name)
        print(f"    Streaming {src_name} ...")
        count = 0
        kept = 0
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if count == 0:
                    # Parse header
                    headers = line.strip().split("\t")
                    count += 1
                    continue
                count += 1
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                eid = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                nn = normalize_name(name)
                keys = generate_blocking_keys(nn, country)

                # Only keep if ANY key overlaps with S1 keys OR it's a GT match
                dominated = any(k in s1_block_keys for k in keys) or eid in gt_match_ids
                if dominated:
                    entity_records[eid] = (name, addr, country)
                    for k in keys:
                        if k in s1_block_keys:
                            block_index[k].add(eid)
                    kept += 1

                if count % 2_000_000 == 0:
                    print(f"      ... {count:,} scanned, {kept:,} kept")

        print(f"    Scanned {count:,}, kept {kept:,} from {src_name}")

    print(f"    Total kept: {len(entity_records):,} entity records")
    print(f"    Block index keys: {len(block_index):,}")

    # 6. Load ML model
    print(f"[6/7] Loading model ...")
    clf = None
    threshold = 0.5
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        clf = model_data["classifier"]
        threshold = model_data["threshold"]
        print(f"    Threshold: {threshold:.2f}")
    else:
        print("    WARNING: No model found!")
        return

    # 7. Run prediction + evaluate
    print(f"[7/7] Running prediction on {len(eval_s1_ids)} entities ...")
    predicted = {}
    total_candidates = 0
    total_matches = 0
    blocked_out = 0
    total_gt_pairs = sum(len(v) for v in eval_gt.values())

    for i, s1_id in enumerate(eval_s1_ids):
        if s1_id not in s1_records:
            predicted[s1_id] = []
            continue

        s1_name, s1_addr, s1_country = s1_records[s1_id]
        s1_norm = normalize_name(s1_name)

        # Blocking
        candidates = set()
        for key in s1_key_map.get(s1_id, []):
            bucket = block_index.get(key)
            if bucket and len(bucket) <= MAX_BLOCK_SIZE:
                candidates.update(bucket)

        total_candidates += len(candidates)

        # Check blocking recall
        gt_set = set(eval_gt[s1_id])
        if gt_set:
            missed = gt_set - candidates
            # Also check if missed IDs were even loaded
            missed_loaded = missed & set(entity_records.keys())
            missed_not_loaded = missed - set(entity_records.keys())
            blocked_out += len(missed)

        # Classification with pre-filter
        matched_ids = []
        if candidates:
            cand_list = list(candidates)
            # Quick pre-filter
            filtered = []
            for cid in cand_list:
                if cid not in entity_records:
                    continue
                c_name, c_addr, c_country = entity_records[cid]
                c_norm = normalize_name(c_name)
                if fuzz.ratio(s1_norm, c_norm) >= 40:
                    filtered.append(cid)

            if filtered:
                feat_matrix = np.zeros((len(filtered), 12), dtype=np.float32)
                for j, cid in enumerate(filtered):
                    c_name, c_addr, _ = entity_records[cid]
                    feat_matrix[j] = extract_features(s1_name, s1_addr, c_name, c_addr)

                probs = clf.predict_proba(feat_matrix)[:, 1]
                matched_ids = [cid for cid, p in zip(filtered, probs) if p >= threshold]

        predicted[s1_id] = matched_ids
        total_matches += len(matched_ids)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i+1:>6}/{len(eval_s1_ids)} "
                  f"cands={total_candidates:,} matches={total_matches:,} "
                  f"blocked_out={blocked_out} rate={rate:.0f}/s")

    # ── RESULTS ──
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Compute per-entity F0.5
    scores = []
    tp_total = fp_total = fn_total = 0
    fp_singleton_count = 0

    for s1_id in eval_s1_ids:
        pred_set = set(predicted.get(s1_id, []))
        gt_s = set(eval_gt[s1_id])

        if not gt_s and not pred_set:
            scores.append(1.0)
            continue
        if not gt_s and pred_set:
            scores.append(0.0)
            fp_singleton_count += 1
            fp_total += len(pred_set)
            continue

        tp = len(pred_set & gt_s)
        fp = len(pred_set - gt_s)
        fn = len(gt_s - pred_set)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = 1.25 * p * r / (0.25 * p + r) if (p + r) > 0 else 0.0
        scores.append(f05)

    macro_f05 = np.mean(scores)

    print(f"  Entities evaluated:     {len(eval_s1_ids):,}")
    print(f"  Singletons (GT):        {n_without:,}")
    print(f"  With matches (GT):      {n_with:,}")
    print(f"  Total GT pairs:         {total_gt_pairs:,}")
    print(f"  Total predicted pairs:  {total_matches:,}")
    print(f"  Total candidates:       {total_candidates:,} (avg {total_candidates/len(eval_s1_ids):.1f}/entity)")
    print(f"  Blocking recall miss:   {blocked_out:,} / {total_gt_pairs:,} ({blocked_out/max(total_gt_pairs,1)*100:.2f}%)")
    print()
    print(f"  TP: {tp_total:,}  FP: {fp_total:,}  FN: {fn_total:,}")
    if tp_total + fp_total > 0:
        print(f"  Micro Precision: {tp_total/(tp_total+fp_total)*100:.2f}%")
    if tp_total + fn_total > 0:
        print(f"  Micro Recall:    {tp_total/(tp_total+fn_total)*100:.2f}%")
    print(f"  FP on singletons:      {fp_singleton_count:,}")
    print()
    print(f"  >>> MACRO F0.5 (leaderboard score): {macro_f05:.4f} <<<")
    print()

    # Score distribution
    score_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
    hist, _ = np.histogram(scores, bins=score_bins)
    print("  Score distribution:")
    for lo, hi, cnt in zip(score_bins[:-1], score_bins[1:], hist):
        bar = "#" * (cnt * 50 // len(scores))
        print(f"    [{lo:.1f}-{hi:.1f}): {cnt:>6} {bar}")

    # Error analysis: worst entities
    entity_scores = []
    for s1_id in eval_s1_ids:
        pred_set = set(predicted.get(s1_id, []))
        gt_s = set(eval_gt[s1_id])
        if not gt_s and not pred_set:
            sc = 1.0
        elif not gt_s and pred_set:
            sc = 0.0
        else:
            tp = len(pred_set & gt_s)
            fp = len(pred_set - gt_s)
            fn = len(gt_s - pred_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sc = 1.25 * p * r / (0.25 * p + r) if (p + r) > 0 else 0.0
        entity_scores.append((s1_id, sc, len(gt_s), len(pred_set)))

    entity_scores.sort(key=lambda x: x[1])
    zero_count = sum(1 for _, s, _, _ in entity_scores if s == 0.0)
    perfect_count = sum(1 for _, s, _, _ in entity_scores if s == 1.0)
    print(f"\n  Score=0.0 entities: {zero_count}")
    print(f"  Score=1.0 entities: {perfect_count}")

    # Show worst non-singleton examples
    print(f"\n  10 worst non-singleton entities:")
    shown = 0
    for s1_id, sc, n_gt, n_pred in entity_scores:
        if n_gt == 0:
            continue
        if shown >= 10:
            break
        s1_name = s1_records.get(s1_id, ("?", "?", "?"))[0]
        print(f"    score={sc:.3f} GT={n_gt} pred={n_pred} name='{s1_name}'")
        pred_set = set(predicted.get(s1_id, []))
        gt_s = set(eval_gt[s1_id])
        for fn_id in list(gt_s - pred_set)[:2]:
            fn_name = entity_records.get(fn_id, ("?",))[0]
            print(f"      MISSED: '{fn_name}'")
        for fp_id in list(pred_set - gt_s)[:2]:
            fp_name = entity_records.get(fp_id, ("?",))[0]
            print(f"      FALSE+: '{fp_name}'")
        shown += 1

    # FP singleton examples
    print(f"\n  5 singleton false-positive examples:")
    shown = 0
    for s1_id, sc, n_gt, n_pred in entity_scores:
        if n_gt == 0 and n_pred > 0 and shown < 5:
            s1_name = s1_records.get(s1_id, ("?", "?", "?"))[0]
            print(f"    name='{s1_name}' pred={n_pred}")
            for pid in predicted.get(s1_id, [])[:3]:
                pn = entity_records.get(pid, ("?",))[0]
                print(f"      FP: '{pn}'")
            shown += 1

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
