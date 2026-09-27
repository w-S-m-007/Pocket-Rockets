"""
Test V5 Hybrid Blocking Recall: Rule Keys + TF-IDF Top-N + Dense Embedding
"""

import csv
import os
import re
import random
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

TRAIN_DIR = r"C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\train"
MAX_BLOCK_SIZE = 500

_CORPORATE_SUFFIXES = frozenset({
    "inc", "incorporated", "llc", "llp", "corp", "corporation",
    "ltd", "limited", "co", "company", "pvt", "private",
    "plc", "gmbh", "sa", "sarl", "sas", "ag", "nv", "bv",
    "lp", "pllc", "dba", "and",
})

def normalize_name(name):
    if not name: return ""
    return " ".join(t for t in re.sub(r"[^a-z0-9\s]", " ", name.lower()).split()
                    if t not in _CORPORATE_SUFFIXES).strip()

def normalize_addr(addr):
    if not addr: return ""
    return re.sub(r"[^a-z0-9\s]", " ", addr.lower()).strip()

def generate_rule_keys(norm_name, norm_addr, country):
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

def main():
    print("[1] Loading Ground Truth ...")
    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s = row["matched_entity_ids"].strip()
            gt[row["source1_entity_id"]] = s.split(",") if s else []

    with_gt = [k for k, v in gt.items() if v]
    random.seed(42)
    random.shuffle(with_gt)
    eval_ids = set(with_gt[:5000])

    print(f"Evaluating blocking recall on {len(eval_ids)} S1 entities with ground truth ...")

    # Load S1
    s1_data = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            eid = row["entity_id"]
            if eid in eval_ids:
                s1_data[eid] = (row.get("business_name",""), row.get("business_address",""), row.get("country",""))

    needed_gt = set()
    for eid in eval_ids:
        needed_gt.update(gt[eid])

    # Build S1 Rule Keys
    s1_rule_keys = set()
    s1_key_map = {}
    for eid, (n, a, c) in s1_data.items():
        keys = generate_rule_keys(normalize_name(n), normalize_addr(a), c)
        s1_key_map[eid] = keys
        s1_rule_keys.update(keys)

    print("[2] Streaming S2 & S3 ...")
    s23_ids = []
    s23_texts = []
    s23_records = {}
    block_index = defaultdict(list)

    for src in ["train_source2.tsv", "train_source3.tsv"]:
        count = kept = 0
        with open(os.path.join(TRAIN_DIR, src), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if count == 0: count += 1; continue
                count += 1
                parts = line.strip().split("\t")
                if len(parts) < 2: continue
                eid = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                nn = normalize_name(name)
                na = normalize_addr(addr)

                keys = generate_rule_keys(nn, na, country)
                relevant = any(k in s1_rule_keys for k in keys) or eid in needed_gt

                if relevant or len(s23_ids) < 300000:  # sample 300k total
                    s23_ids.append(eid)
                    combined_text = f"{nn} {na}".strip()
                    s23_texts.append(combined_text)
                    s23_records[eid] = (name, addr, country)
                    for k in keys:
                        if k in s1_rule_keys:
                            block_index[k].append(eid)
                    kept += 1
        print(f"  {src}: {count:,} scanned, {kept:,} kept")

    print(f"Total S2/S3 indexed: {len(s23_ids):,}")

    # Check Rule-Based Blocking Recall
    rule_miss = 0
    total_gt_pairs = sum(len(gt[eid]) for eid in eval_ids)

    for eid in eval_ids:
        gt_set = set(gt[eid])
        cands = set()
        for k in s1_key_map.get(eid, []):
            b = block_index.get(k)
            if b and len(b) <= MAX_BLOCK_SIZE:
                cands.update(b)
        rule_miss += len(gt_set - cands)

    print(f"\nRule-based Blocking Miss: {rule_miss}/{total_gt_pairs} ({rule_miss/total_gt_pairs*100:.2f}%)")
    print(f"Rule-based Blocking Recall: {(1 - rule_miss/total_gt_pairs)*100:.2f}%")

    # Now add TF-IDF Character N-Gram Top-K Matching!
    print("\n[3] Fitting TF-IDF Vectorizer (char_wb (2,4)) ...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    X_s23 = vectorizer.fit_transform(s23_texts)
    print(f"TF-IDF fit on {len(s23_texts):,} records in {time.time()-t0:.2f}s, matrix shape: {X_s23.shape}")

    s1_eval_list = list(eval_ids)
    s1_eval_texts = [f"{normalize_name(s1_data[eid][0])} {normalize_addr(s1_data[eid][1])}".strip() for eid in s1_eval_list]
    X_s1 = vectorizer.transform(s1_eval_texts)

    print("[4] Querying TF-IDF Top-20 per S1 entity via sparse_dot_topn ...")
    t0 = time.time()
    topn_matches = sp_matmul_topn(X_s1, X_s23.T, top_n=25, threshold=0.15)
    print(f"TF-IDF matrix matmul finished in {time.time()-t0:.2f}s")

    # Combine Rule + TF-IDF Top-20 candidates
    combined_miss = 0
    s23_id_arr = np.array(s23_ids)

    # Convert sparse matmul to row-by-row matches
    topn_coo = topn_matches.tocsr()

    for idx, eid in enumerate(s1_eval_list):
        gt_set = set(gt[eid])

        # Candidates from Rules
        cands = set()
        for k in s1_key_map.get(eid, []):
            b = block_index.get(k)
            if b and len(b) <= MAX_BLOCK_SIZE:
                cands.update(b)

        # Candidates from TF-IDF
        row = topn_coo[idx]
        for col_idx in row.indices:
            cands.add(s23_ids[col_idx])

        combined_miss += len(gt_set - cands)

    print(f"\n==================================================")
    print(f"HYBRID (Rule + TF-IDF Top-25) Blocking Miss: {combined_miss}/{total_gt_pairs} ({combined_miss/total_gt_pairs*100:.2f}%)")
    print(f"HYBRID Blocking Recall: {(1 - combined_miss/total_gt_pairs)*100:.2f}%")
    print(f"==================================================")

if __name__ == "__main__":
    main()
