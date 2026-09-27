"""
Business Entity Resolution — V7 Prediction (Ultra Fast & Lean + Multiprocessing)
================================================================================
- NO TF-IDF matrix at test time (saves ~6GB RAM)
- Rule-based blocking only with pruning (MAX_BLOCK_SIZE = 25 for high recall)
- Pre-normalized strings (saves regex overhead)
- Multiprocessing Pool (12 cores) for parallel feature extraction
- Ultra-safe dual short-circuiting to bypass ML inference for complete garbage candidates
"""

import os
import re
import sys
import time
import pickle
import gc
import warnings
from collections import defaultdict
import multiprocessing

warnings.filterwarnings("ignore")

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

TEST_DIR = r"C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\test"
OUTPUT_DIR = "output"
MODEL_PATH = "model_v5.pkl"
MAX_BLOCK_SIZE = 25
BUF_SIZE = 8 * 1024 * 1024

_CORPORATE_SUFFIXES = frozenset({
    "inc", "incorporated", "llc", "llp", "corp", "corporation",
    "ltd", "limited", "co", "company", "pvt", "private",
    "plc", "gmbh", "sa", "sarl", "sas", "ag", "nv", "bv",
    "lp", "pllc", "dba", "and",
})

def normalize_name(name: str) -> str:
    if not name: return ""
    return " ".join(t for t in re.sub(r"[^a-z0-9\s]", " ", name.lower()).split()
                    if t not in _CORPORATE_SUFFIXES).strip()

def normalize_addr(addr: str) -> str:
    if not addr: return ""
    return re.sub(r"[^a-z0-9\s]", " ", addr.lower()).strip()

def extract_street_number(addr: str) -> str:
    if not addr: return ""
    tokens = addr.split()
    for t in tokens:
        if t.isdigit() and len(t) <= 6:
            return t
    return ""

def generate_rule_keys(norm_name: str, norm_addr: str, country: str) -> list:
    keys = []
    c = country.strip().lower() if country else ""
    if norm_name:
        if len(norm_name) >= 3:
            keys.append(f"{c}|p5|{norm_name[:5]}")
            keys.append(f"{c}|p3|{norm_name[:3]}")
        tokens = norm_name.split()
        if tokens:
            keys.append(f"{c}|st|{'_'.join(sorted(tokens)[:3])[:25]}")
        for tok in tokens:
            if len(tok) >= 4:
                keys.append(f"{c}|tk|{tok[:10]}")
    if norm_addr:
        addr_tokens = norm_addr.split()
        numbers = [t for t in addr_tokens if t.isdigit()]
        alpha = [t for t in addr_tokens if t.isalpha() and len(t) >= 3]
        if numbers and alpha:
            keys.append(f"{c}|an|{numbers[0]}_{alpha[0][:8]}")
        sig = [t for t in addr_tokens if len(t) >= 3][:3]
        if len(sig) >= 2:
            keys.append(f"{c}|a3|{'_'.join(sig)}")
        for t in addr_tokens:
            if t.isdigit() and len(t) >= 3:
                keys.append(f"{c}|anum|{t}")
                break
    return keys

def _token_jaccard(s1, s2):
    t1, t2 = set(s1.split()), set(s2.split())
    u = t1 | t2
    return len(t1 & t2) / len(u) if u else 0.0

def _char_ngram_jaccard(s1, s2, n=3):
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    g1 = {s1[i:i+n] for i in range(len(s1)-n+1)}
    g2 = {s2[i:i+n] for i in range(len(s2)-n+1)}
    u = g1 | g2
    return len(g1 & g2) / len(u) if u else 0.0

def extract_features(n1, a1, s1_country, c_name, c_addr, c_country, cand_count=1):
    n2 = c_name
    a2 = c_addr
    
    # ULTRA-SAFE DUAL SHORT CIRCUIT
    nr = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    ar = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    if nr < 0.25 and ar < 0.25:
        return None # Guaranteed garbage match

    if n1 and n2:
        np_ = fuzz.partial_ratio(n1, n2) / 100.0
        nts = fuzz.token_sort_ratio(n1, n2) / 100.0
        nte = fuzz.token_set_ratio(n1, n2) / 100.0
        njw = JaroWinkler.normalized_similarity(n1, n2)
        nj = _token_jaccard(n1, n2)
        nc3 = _char_ngram_jaccard(n1, n2, n=3)
        nld = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
        n_exact = 1.0 if n1 == n2 else 0.0
        w1 = n1.split()[0] if n1 else ""
        w2 = n2.split()[0] if n2 else ""
        n_first = 1.0 if w1 and w1 == w2 else 0.0
    else:
        np_ = nts = nte = njw = nj = nc3 = nld = n_exact = n_first = 0.0

    if a1 and a2:
        ap = fuzz.partial_ratio(a1, a2) / 100.0
        ats = fuzz.token_sort_ratio(a1, a2) / 100.0
        ate = fuzz.token_set_ratio(a1, a2) / 100.0
        ajw = JaroWinkler.normalized_similarity(a1, a2)
        aj = _token_jaccard(a1, a2)
        a_empty = 0.0
    else:
        ap = ats = ate = ajw = aj = 0.0
        a_empty = 1.0

    num1 = extract_street_number(a1)
    num2 = extract_street_number(a2)
    if num1 and num2:
        num_match = 1.0 if num1 == num2 else 0.0
        has_num = 1.0
    elif num1 or num2:
        num_match = 0.0
        has_num = 1.0
    else:
        num_match = 0.5
        has_num = 0.0

    c1 = s1_country.strip().lower() if s1_country else ""
    c2 = c_country.strip().lower() if c_country else ""
    if c1 and c2:
        c_match = 1.0 if c1 == c2 else 0.0
    else:
        c_match = 0.5

    cand_log = float(np.log1p(cand_count))

    tfidf_sim = 0.0
    tfidf_rank = 50.0
    tfidf_margin = 0.0

    return [
        nr, np_, nts, nte, njw, nj, nc3, nld, n_exact, n_first,
        ar, ap, ats, ate, ajw, aj, num_match, has_num, a_empty,
        tfidf_sim, c_match, cand_log, tfidf_rank, tfidf_margin
    ]

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize pool now so Windows multiprocessing doesn't infinite-loop
    pool = multiprocessing.Pool(processes=12)

    print(f"[1/4] Loading model from {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found!")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    xgb_clf = model_data["xgb_model"]
    xgb_clf.set_params(device="cpu")
    lgb_clf = model_data["lgb_model"]
    threshold = max(model_data["threshold"] - 0.10, 0.50)
    print(f"    Model loaded. Adjusted threshold: {threshold:.2f}")

    print("[2/4] Loading S1 test records ...")
    s1_records = {}
    s1_key_map = {}
    s1_all_keys = set()

    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    count = 0
    with open(s1_path, "r", encoding="utf-8", errors="replace", buffering=BUF_SIZE) as f:
        while True:
            line = f.readline()
            if not line: break
            count += 1
            if count == 1: continue
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            eid = parts[0]
            name = parts[1]
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""
            
            nn = normalize_name(name)
            na = normalize_addr(addr)
            s1_records[eid] = (nn, na, country)

            keys = generate_rule_keys(nn, na, country)
            s1_key_map[eid] = keys
            s1_all_keys.update(keys)

    print(f"    {len(s1_records):,} S1 records, {len(s1_all_keys):,} unique rule keys")

    print("[3/4] Streaming S2/S3 and building pruned block index ...")
    s23_records = {}
    block_index = defaultdict(list)
    block_sizes = defaultdict(int)

    for src_name in ["test_source2.tsv", "test_source3.tsv"]:
        print(f"    {src_name} ...")
        path = os.path.join(TEST_DIR, src_name)
        count = kept = 0
        with open(path, "r", encoding="utf-8", errors="replace", buffering=BUF_SIZE) as f:
            while True:
                line = f.readline()
                if not line: break
                count += 1
                if count == 1: continue
                parts = line.strip().split("\t")
                if len(parts) < 2: continue
                eid = parts[0]
                name = parts[1]
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""

                nn = normalize_name(name)
                na = normalize_addr(addr)
                keys = generate_rule_keys(nn, na, country)

                added = False
                for k in keys:
                    if k in s1_all_keys:
                        if block_sizes[k] < MAX_BLOCK_SIZE:
                            block_index[k].append(eid)
                            block_sizes[k] += 1
                            added = True

                if added:
                    if eid not in s23_records:
                        s23_records[eid] = (nn, na, country)
                        kept += 1

                if count % 2_000_000 == 0:
                    print(f"      {count:,} scanned, {kept:,} stored")

        print(f"      Done: {count:,} scanned, {kept:,} stored")

    del block_sizes
    gc.collect()
    print(f"    Total S2/S3 stored: {len(s23_records):,}")
    print(f"    Block index keys: {len(block_index):,}")

    print("[4/4] Processing S1 entities ...")
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    total_s1 = len(s1_records)
    matches_count = 0
    processed = 0

    with open(match_path, "w", encoding="utf-8", newline="") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        s1_ids = list(s1_records.keys())
        BATCH_SIZE = 5000

        for batch_start in range(0, len(s1_ids), BATCH_SIZE):
            batch_ids = s1_ids[batch_start:batch_start + BATCH_SIZE]
            
            jobs = []
            job_meta = [] # (s1_id, cid)

            for s1_id in batch_ids:
                nn, na, country = s1_records[s1_id]
                cand_set = set()
                for k in s1_key_map.get(s1_id, []):
                    bl = block_index.get(k)
                    if bl:
                        cand_set.update(bl)

                filt = [c for c in cand_set if c in s23_records]
                if filt:
                    cand_count = len(filt)
                    for cid in filt:
                        cn, ca, cc = s23_records[cid]
                        jobs.append((nn, na, country, cn, ca, cc, cand_count))
                        job_meta.append((s1_id, cid))

            all_features = []
            all_meta = []
            
            if jobs:
                # Chunksize 200 reduces IPC overhead
                results = pool.starmap(extract_features, jobs, chunksize=200)
                for res, meta in zip(results, job_meta):
                    if res is not None:
                        all_features.append(res)
                        all_meta.append(meta)

            entity_matches = defaultdict(list)
            if all_features:
                X_batch = np.array(all_features, dtype=np.float32)
                p_xgb = xgb_clf.predict_proba(X_batch)[:, 1]
                p_lgb = lgb_clf.predict_proba(X_batch)[:, 1]
                p_blend = 0.5 * p_xgb + 0.5 * p_lgb

                for idx, (s1_id, cid) in enumerate(all_meta):
                    if p_blend[idx] >= threshold:
                        entity_matches[s1_id].append(cid)

            for s1_id in batch_ids:
                matched = entity_matches.get(s1_id, [])
                matches_count += len(matched)
                fm.write(f"{s1_id}\t{','.join(matched)}\n")

            processed += len(batch_ids)
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta_min = (total_s1 - processed) / rate / 60 if rate > 0 else 0
            print(f"    {processed:,} / {total_s1:,} ... matches={matches_count:,}, rate={rate:.0f}/s, ETA={eta_min:.0f}min")

    pool.close()
    pool.join()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total matches: {matches_count:,}")
    print(f"Output: {match_path}")


if __name__ == "__main__":
    main()
