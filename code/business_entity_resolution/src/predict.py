"""
Business Entity Resolution — Prediction Pipeline
==================================================
Loads test data, builds an in-memory blocking index, scores candidate pairs
with a trained classifier, and outputs matching_results.tsv + candidate_pairs.tsv.

Usage:
    python src/predict.py                  # uses trained model
    python src/predict.py --heuristic      # fallback: fixed-threshold heuristic

Requires model.pkl from train.py (unless --heuristic is used).
"""

import csv
import os
import re
import sys
import time
import pickle
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

# ─────────────────────────── Configuration ───────────────────────────

TEST_DIR  = r"C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\test"
MODEL_PATH = "model.pkl"
OUTPUT_DIR = "output"
MAX_BLOCK_SIZE = 500        # ignore blocking keys that match > N candidates

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


# ─────────────────────────── Blocking ────────────────────────────────

def generate_blocking_keys(norm_name: str, country: str) -> list:
    """Generate multiple blocking keys for high-recall candidate retrieval."""
    keys = []
    country = country.strip().lower() if country else ""
    if not norm_name:
        return keys

    # Key 1: first 5 characters of normalised name
    if len(norm_name) >= 3:
        keys.append(f"{country}|p5|{norm_name[:5]}")

    # Key 2: sorted first 3 tokens (catches word reordering)
    tokens = norm_name.split()
    if tokens:
        sorted_tok = "_".join(sorted(tokens)[:3])
        keys.append(f"{country}|st|{sorted_tok[:25]}")

    # Key 3: longest word (catches distinctive business word)
    if tokens:
        longest = max(tokens, key=len)
        if len(longest) >= 4:
            keys.append(f"{country}|lw|{longest}")

    # Key 4: first 3 chars (broader recall fallback)
    if len(norm_name) >= 3:
        keys.append(f"{country}|p3|{norm_name[:3]}")

    return keys


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

    if n1 and n2:
        name_ratio   = fuzz.ratio(n1, n2) / 100.0
        name_partial = fuzz.partial_ratio(n1, n2) / 100.0
        name_tsort   = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_tset    = fuzz.token_set_ratio(n1, n2) / 100.0
        name_jw      = JaroWinkler.normalized_similarity(n1, n2)
        name_jaccard = _token_jaccard(n1, n2)
    else:
        name_ratio = name_partial = name_tsort = name_tset = name_jw = name_jaccard = 0.0

    if a1 and a2:
        addr_ratio   = fuzz.ratio(a1, a2) / 100.0
        addr_partial = fuzz.partial_ratio(a1, a2) / 100.0
        addr_tsort   = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_jw      = JaroWinkler.normalized_similarity(a1, a2)
        addr_jaccard = _token_jaccard(a1, a2)
    else:
        addr_ratio = addr_partial = addr_tsort = addr_jw = addr_jaccard = 0.0

    name_len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)

    return [
        name_ratio, name_partial, name_tsort, name_tset, name_jw, name_jaccard,
        addr_ratio, addr_partial, addr_tsort, addr_jw, addr_jaccard,
        name_len_diff,
    ]


# ─────────────────────────── Heuristic fallback ──────────────────────

def heuristic_match(features: list) -> bool:
    """Simple threshold-based matching when no trained model is available."""
    name_ratio, name_partial, name_tsort, name_tset, name_jw, name_jaccard = features[:6]
    addr_ratio = features[6]
    addr_jw    = features[9]

    if name_jw > 0.92:
        return True
    if name_tsort > 0.90 and name_tset > 0.90:
        return True
    if name_jw > 0.82 and addr_jw > 0.70:
        return True
    return False


# ─────────────────────────── Main prediction ─────────────────────────

def main():
    use_heuristic = "--heuristic" in sys.argv
    t0 = time.time()

    # ── 1. Load model (if available) ──
    clf = None
    threshold = 0.5
    if not use_heuristic:
        if os.path.exists(MODEL_PATH):
            print(f"[1/5] Loading model from {MODEL_PATH} ...")
            with open(MODEL_PATH, "rb") as f:
                model_data = pickle.load(f)
            clf       = model_data["classifier"]
            threshold = model_data["threshold"]
            print(f"    Threshold: {threshold:.2f}, Val F0.5: {model_data.get('f05_val', '?')}")
        else:
            print(f"[1/5] WARNING: {MODEL_PATH} not found — falling back to heuristic mode")
            use_heuristic = True
    else:
        print("[1/5] Running in heuristic mode (no trained model)")

    # ── 2. Load Source 1 ──
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    print(f"[2/5] Loading Source 1 from {s1_path} ...")
    s1_records = []   # list of (eid, name, addr, country, norm_name)
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            name = row.get("business_name", "")
            addr = row.get("business_address", "")
            country = row.get("country", "")
            nn = normalize_name(name)
            s1_records.append((eid, name, addr, country, nn))
    print(f"    Loaded {len(s1_records)} S1 entities")

    # ── 3. Load Source 2 + 3  &  build blocking index ──
    entity_records = {}    # eid → (name, addr, country)
    block_index = defaultdict(set)   # key → set of eids

    for src_num, src_name in [(2, "test_source2.tsv"), (3, "test_source3.tsv")]:
        src_path = os.path.join(TEST_DIR, src_name)
        print(f"[3/5] Loading {src_name} + building blocking index ...")
        count = 0
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row["entity_id"]
                name = row.get("business_name", "")
                addr = row.get("business_address", "")
                country = row.get("country", "")
                nn = normalize_name(name)

                entity_records[eid] = (name, addr, country)

                for key in generate_blocking_keys(nn, country):
                    block_index[key].add(eid)

                count += 1
                if count % 1_000_000 == 0:
                    print(f"      ... {count:,} records loaded")
        print(f"    Loaded {count:,} records from {src_name}")

    print(f"    Total S2+S3 entities: {len(entity_records):,}")
    print(f"    Blocking index keys:  {len(block_index):,}")

    # Compute block size stats
    block_sizes = [len(v) for v in block_index.values()]
    print(f"    Block sizes — median: {int(np.median(block_sizes))}, "
          f"mean: {np.mean(block_sizes):.0f}, max: {max(block_sizes)}, "
          f"keys > {MAX_BLOCK_SIZE}: {sum(1 for s in block_sizes if s > MAX_BLOCK_SIZE)}")

    # ── 4. Process S1 entities ──
    print(f"[4/5] Processing {len(s1_records):,} S1 entities ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = len(s1_records)
    total_candidates = 0
    total_matches = 0

    with open(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), "w",
              encoding="utf-8", newline="") as fcand, \
         open(os.path.join(OUTPUT_DIR, "matching_results.tsv"), "w",
              encoding="utf-8", newline="") as fmatch:

        fcand.write("source1_entity_id\tcandidate_entity_ids\n")
        fmatch.write("source1_entity_id\tmatched_entity_ids\n")

        for i, (s1_id, s1_name, s1_addr, s1_country, s1_norm) in enumerate(s1_records):

            # ── blocking ──
            candidates = set()
            for key in generate_blocking_keys(s1_norm, s1_country):
                bucket = block_index.get(key)
                if bucket and len(bucket) <= MAX_BLOCK_SIZE:
                    candidates.update(bucket)

            cand_list = list(candidates)
            total_candidates += len(cand_list)

            # ── scoring ──
            matched_ids = []
            if cand_list:
                if clf is not None and not use_heuristic:
                    # Batch feature extraction
                    feat_matrix = np.zeros((len(cand_list), len(FEATURE_NAMES)),
                                          dtype=np.float32)
                    for j, cid in enumerate(cand_list):
                        c_name, c_addr, _ = entity_records[cid]
                        feat_matrix[j] = extract_features(
                            s1_name, s1_addr, c_name, c_addr
                        )
                    probs = clf.predict_proba(feat_matrix)[:, 1]
                    matched_ids = [
                        cid for cid, p in zip(cand_list, probs) if p >= threshold
                    ]
                else:
                    # Heuristic mode
                    for cid in cand_list:
                        c_name, c_addr, _ = entity_records[cid]
                        feat = extract_features(s1_name, s1_addr, c_name, c_addr)
                        if heuristic_match(feat):
                            matched_ids.append(cid)

            total_matches += len(matched_ids)

            # ── write output ──
            fcand.write(f"{s1_id}\t{','.join(cand_list)}\n")
            fmatch.write(f"{s1_id}\t{','.join(matched_ids)}\n")

            if (i + 1) % 50_000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(f"    {i+1:>10,}/{total:,}  "
                      f"cands={total_candidates:,}  matches={total_matches:,}  "
                      f"rate={rate:.0f}/s  ETA={eta/60:.1f}min")

    # ── 5. Summary ──
    elapsed = time.time() - t0
    print(f"\n[5/5] Done!")
    print(f"    S1 entities processed: {total:,}")
    print(f"    Total candidates:      {total_candidates:,}  "
          f"(avg {total_candidates/total:.1f}/entity)")
    print(f"    Total matches:         {total_matches:,}  "
          f"(avg {total_matches/total:.2f}/entity)")
    print(f"    Output files:")
    print(f"      {os.path.join(OUTPUT_DIR, 'matching_results.tsv')}")
    print(f"      {os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')}")
    print(f"    Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
