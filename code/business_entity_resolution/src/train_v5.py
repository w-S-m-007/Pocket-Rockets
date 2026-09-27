"""
Business Entity Resolution — V5 (Memory & File-IO Optimized TF-IDF + XGBoost + LightGBM Ensemble)
==================================================================================================
Fixes:
  1. Safe file buffering (buffering=8MB + f.readline()) to avoid Windows OSError 22 on >500MB files.
  2. Stream-batched TF-IDF vectorization (batches of 200,000 texts -> sparse vstack)
  3. Hybrid Blocking: Rule Keys + Sparse TF-IDF Top-30 Cosine Matching via sparse-dot-topn
  4. 24 Rich Features: Levenshtein, Jaro-Winkler, Jaccard, Street Number, TF-IDF Sim, Margins, Ranks
  5. Dual Ensemble: XGBoost + LightGBM (1,000 trees each)
  6. Entity-Level Macro F0.5 Calibration
"""

import csv
import os
import re
import sys
import time
import random
import pickle
import gc
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score

TRAIN_DIR = r"C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\train"
MODEL_PATH = "model_v5.pkl"
SAMPLE_S1_TRAIN = 30000
SAMPLE_S1_EVAL = 5000
HARD_NEG_PER_POS = 15
RANDOM_SEED = 42
MAX_BLOCK_SIZE = 500
BUF_SIZE = 8 * 1024 * 1024

# ─────────────────────────── Normalisation ───────────────────────────

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

# ─────────────────────────── 24 Rich Features ─────────────────────────

FEATURE_NAMES = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jw", "name_jaccard", "name_char3_jaccard", "name_len_diff_ratio",
    "name_exact_match", "name_first_word_match",
    "addr_ratio", "addr_partial_ratio", "addr_token_sort_ratio", "addr_token_set_ratio",
    "addr_jw", "addr_jaccard", "addr_street_num_match", "addr_has_number", "addr_empty_flag",
    "tfidf_sim", "country_match", "cand_count_log", "tfidf_rank", "tfidf_margin"
]

def _token_jaccard(s1, s2):
    t1, t2 = set(s1.split()), set(s2.split())
    u = t1 | t2
    return len(t1 & t2) / len(u) if u else 0.0

def _char_ngram_jaccard(s1, s2, n=3):
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    g1 = set(s1[i:i+n] for i in range(len(s1)-n+1))
    g2 = set(s2[i:i+n] for i in range(len(s2)-n+1))
    u = g1 | g2
    return len(g1 & g2) / len(u) if u else 0.0

def extract_24_features(s1_name, s1_addr, s1_country,
                        c_name, c_addr, c_country,
                        tfidf_sim=0.0, tfidf_rank=1, tfidf_margin=0.0, cand_count=1):
    n1 = normalize_name(s1_name)
    n2 = normalize_name(c_name)
    a1 = normalize_addr(s1_addr)
    a2 = normalize_addr(c_addr)

    # Name features
    if n1 and n2:
        nr = fuzz.ratio(n1, n2) / 100.0
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
        nr = np_ = nts = nte = njw = nj = nc3 = nld = n_exact = n_first = 0.0

    # Address features
    if a1 and a2:
        ar = fuzz.ratio(a1, a2) / 100.0
        ap = fuzz.partial_ratio(a1, a2) / 100.0
        ats = fuzz.token_sort_ratio(a1, a2) / 100.0
        ate = fuzz.token_set_ratio(a1, a2) / 100.0
        ajw = JaroWinkler.normalized_similarity(a1, a2)
        aj = _token_jaccard(a1, a2)
        a_empty = 0.0
    else:
        ar = ap = ats = ate = ajw = aj = 0.0
        a_empty = 1.0

    # Street number matching
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

    # Country matching
    c1 = s1_country.strip().lower() if s1_country else ""
    c2 = c_country.strip().lower() if c_country else ""
    if c1 and c2:
        c_match = 1.0 if c1 == c2 else 0.0
    else:
        c_match = 0.5

    cand_log = float(np.log1p(cand_count))

    return [
        nr, np_, nts, nte, njw, nj, nc3, nld, n_exact, n_first,
        ar, ap, ats, ate, ajw, aj, num_match, has_num, a_empty,
        float(tfidf_sim), c_match, cand_log, float(tfidf_rank), float(tfidf_margin)
    ]


def compute_macro_f05(eval_s1, gt, predicted):
    scores = []
    for sid in eval_s1:
        pred_set = set(predicted.get(sid, []))
        gt_s = set(gt[sid])
        if not gt_s and not pred_set:
            scores.append(1.0); continue
        if not gt_s and pred_set:
            scores.append(0.0); continue
        tp = len(pred_set & gt_s)
        fp = len(pred_set - gt_s)
        fn = len(gt_s - pred_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(1.25 * p * r / (0.25 * p + r) if (p + r) > 0 else 0.0)
    return np.mean(scores)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    t0 = time.time()

    # 1. Load Ground Truth
    print("[1/8] Loading Ground Truth ...")
    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s = row["matched_entity_ids"].strip()
            gt[row["source1_entity_id"]] = s.split(",") if s else []

    with_ids = [k for k, v in gt.items() if v]
    without_ids = [k for k, v in gt.items() if not v]
    random.shuffle(with_ids); random.shuffle(without_ids)

    n_tp = min(SAMPLE_S1_TRAIN, len(with_ids) // 2)
    n_tn = min(SAMPLE_S1_TRAIN // 5, len(without_ids) // 2)
    n_ep = min(SAMPLE_S1_EVAL, len(with_ids) - n_tp)
    n_en = min(SAMPLE_S1_EVAL // 10, len(without_ids) - n_tn)

    train_pos = set(with_ids[:n_tp]); train_neg = set(without_ids[:n_tn])
    eval_pos = set(with_ids[n_tp:n_tp + n_ep]); eval_neg = set(without_ids[n_tn:n_tn + n_en])

    train_s1 = train_pos | train_neg
    eval_s1 = eval_pos | eval_neg
    all_s1 = train_s1 | eval_s1

    print(f"    Train: {len(train_pos)} pos + {len(train_neg)} neg = {len(train_s1)}")
    print(f"    Eval:  {len(eval_pos)} pos + {len(eval_neg)} neg = {len(eval_s1)}")

    needed_s23 = set()
    for sid in all_s1:
        needed_s23.update(gt[sid])

    # 2. Load S1 Records
    print("[2/8] Loading S1 records ...")
    s1_records = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8", errors="replace", buffering=BUF_SIZE) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            eid = row["entity_id"]
            if eid in all_s1:
                s1_records[eid] = (row.get("business_name", ""), row.get("business_address", ""), row.get("country", ""))
    print(f"    {len(s1_records)} S1 records loaded.")

    # 3. Rule Keys
    print("[3/8] Computing S1 Rule Blocking Keys ...")
    s1_rule_keys = set()
    s1_key_map = {}
    for sid in all_s1:
        if sid not in s1_records: continue
        n, a, c = s1_records[sid]
        keys = generate_rule_keys(normalize_name(n), normalize_addr(a), c)
        s1_key_map[sid] = keys
        s1_rule_keys.update(keys)
    print(f"    {len(s1_rule_keys):,} unique rule keys")

    # 4. Stream S2 & S3 with safe buffering
    print("[4/8] Streaming S2 & S3 and Building Index (Safe Buffer IO) ...")
    s23_records = {}
    s23_id_list = []
    block_index = defaultdict(list)

    # First Pass: Sample texts to FIT TF-IDF
    sample_texts = []
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(TRAIN_DIR, src)
        with open(path, "r", encoding="utf-8", errors="replace", buffering=BUF_SIZE) as f:
            count = 0
            while True:
                line = f.readline()
                if not line: break
                count += 1
                if count == 1 or count % 20 != 0: continue
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    nn = normalize_name(parts[1])
                    na = normalize_addr(parts[2]) if len(parts) > 2 else ""
                    sample_texts.append(f"{nn} {na}".strip())

    print(f"    Fitting TF-IDF Vectorizer on {len(sample_texts):,} sampled records ...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=3, max_df=0.01, max_features=30000, dtype=np.float32)
    vectorizer.fit(sample_texts)
    del sample_texts; gc.collect()

    print("    Streaming full S2 & S3 with safe buffering ...")
    csr_chunks = []
    chunk_text_batch = []

    for src in ["train_source2.tsv", "train_source3.tsv"]:
        print(f"    {src} ...")
        path = os.path.join(TRAIN_DIR, src)
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

                relevant = any(k in s1_rule_keys for k in keys) or eid in needed_s23
                if relevant:
                    s23_records[eid] = (name, addr, country)
                    s23_id_list.append(eid)
                    chunk_text_batch.append(f"{nn} {na}".strip())

                    for k in keys:
                        if k in s1_rule_keys:
                            block_index[k].append(eid)
                    kept += 1

                if len(chunk_text_batch) >= 10000:
                    csr_chunks.append(vectorizer.transform(chunk_text_batch))
                    chunk_text_batch = []
                    gc.collect()

                if count % 2_000_000 == 0:
                    print(f"      {count:,} scanned, {kept:,} kept")

    if chunk_text_batch:
        csr_chunks.append(vectorizer.transform(chunk_text_batch))
        chunk_text_batch = []

    X_s23 = sp.vstack(csr_chunks).tocsr()
    del csr_chunks; gc.collect()

    print(f"    Total indexed: {len(s23_records):,}, TF-IDF Matrix shape: {X_s23.shape}")

    # Prune Rule Blocks
    for k in list(block_index.keys()):
        if len(block_index[k]) > MAX_BLOCK_SIZE:
            del block_index[k]

    print("    Transposing and converting TF-IDF matrix for fast retrieval...")
    X_s23_T = X_s23.T.tocsr()

    print("    Pre-calculating candidates for all S1 entities in batches...")
    all_cands_dict = {}
    s1_list = list(all_s1)
    BATCH_SIZE = 10000
    
    for i in range(0, len(s1_list), BATCH_SIZE):
        batch_ids = s1_list[i:i+BATCH_SIZE]
        batch_texts = []
        for sid in batch_ids:
            if sid in s1_records:
                n, a, c = s1_records[sid]
                batch_texts.append(f"{normalize_name(n)} {normalize_addr(a)}".strip())
            else:
                batch_texts.append("")
        
        v_s1 = vectorizer.transform(batch_texts)
        top_res = sp_matmul_topn(v_s1, X_s23_T, top_n=35, threshold=0.10).tocsr()
        
        for j, sid in enumerate(batch_ids):
            cands_dict = {}
            for k in s1_key_map.get(sid, []):
                b = block_index.get(k)
                if b:
                    for cid in b:
                        cands_dict[cid] = (0.0, 99)
            
            row = top_res[j]
            if len(row.indices) > 0:
                ranks = np.argsort(-row.data)
                for rank, idx in enumerate(ranks):
                    cid = s23_id_list[row.indices[idx]]
                    sim = float(row.data[idx])
                    if cid in cands_dict:
                        cands_dict[cid] = (max(cands_dict[cid][0], sim), min(cands_dict[cid][1], rank + 1))
                    else:
                        cands_dict[cid] = (sim, rank + 1)
            all_cands_dict[sid] = cands_dict
        print(f"      {min(i+BATCH_SIZE, len(s1_list))} / {len(s1_list)} entities pre-calculated.")

    def get_hybrid_candidates(s1_id, s1_name, s1_addr, s1_country, top_k=30):
        return all_cands_dict.get(s1_id, {})

    # Check Blocking Recall on Eval Set
    print("\n    Checking Hybrid Blocking Recall on Eval Set ...")
    eval_miss = 0
    total_eval_gt = sum(len(gt[sid]) for sid in eval_s1)

    for sid in eval_s1:
        gt_s = set(gt[sid])
        if not gt_s or sid not in s1_records: continue
        sn, sa, sc = s1_records[sid]
        cands = get_hybrid_candidates(sid, sn, sa, sc, top_k=30)
        eval_miss += len(gt_s - set(cands.keys()))

    recall_pct = (1.0 - eval_miss / max(total_eval_gt, 1)) * 100
    print(f"    >>> HYBRID BLOCKING RECALL: {recall_pct:.2f}% (Miss: {eval_miss}/{total_eval_gt}) <<<")

    # 6. Extract 24-Feature Training Matrix
    print(f"\n[6/8] Building 24-Feature Matrix for Training ...")
    X, y = [], []
    pos_count = neg_count = 0

    for i, sid in enumerate(train_pos):
        if sid not in s1_records: continue
        sn, sa, sc = s1_records[sid]
        true_set = set(gt[sid])

        cands_dict = get_hybrid_candidates(sid, sn, sa, sc, top_k=35)
        if not cands_dict: continue

        scores = [v[0] for v in cands_dict.values()]
        max_score = max(scores) if scores else 0.0
        cand_count = len(cands_dict)

        # Positives
        for mid in gt[sid]:
            if mid not in s23_records: continue
            cn, ca, cc = s23_records[mid]
            tsim, trank = cands_dict.get(mid, (0.0, 99))
            tmargin = tsim - max_score
            X.append(extract_24_features(sn, sa, sc, cn, ca, cc, tsim, trank, tmargin, cand_count))
            y.append(1)
            pos_count += 1

        # Hard Negatives
        hard_negs = list(set(cands_dict.keys()) - true_set)
        if hard_negs:
            sampled_negs = random.sample(hard_negs, min(HARD_NEG_PER_POS, len(hard_negs)))
            for nid in sampled_negs:
                if nid not in s23_records: continue
                cn, ca, cc = s23_records[nid]
                tsim, trank = cands_dict[nid]
                tmargin = tsim - max_score
                X.append(extract_24_features(sn, sa, sc, cn, ca, cc, tsim, trank, tmargin, cand_count))
                y.append(0)
                neg_count += 1

        if (i + 1) % 5000 == 0:
            print(f"      {i+1}/{len(train_pos)} entities processed ...")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"    Feature Matrix Shape: {X.shape}, Pos: {pos_count}, Neg: {neg_count}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # 7. Train XGBoost & LightGBM Ensemble
    print(f"\n[7/8] Training XGBoost & LightGBM Ensemble (1,000 trees each) ...")

    print("    Fitting XGBoost on GPU ...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
        random_state=42
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)

    print("    Fitting LightGBM on GPU (if supported) ...")
    try:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=7, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, device_type="gpu"
        )
        lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    except Exception as e:
        print("    LightGBM GPU failed, falling back to CPU...")
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=7, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    # Feature Importances
    print("\n    Top 15 Feature Importances (XGBoost):")
    for name, imp in sorted(zip(FEATURE_NAMES, xgb_clf.feature_importances_), key=lambda x: -x[1])[:15]:
        print(f"      {name:30s}  {imp:.4f}")

    # 8. ENTITY-LEVEL MACRO F0.5 THRESHOLD CALIBRATION
    print(f"\n[8/8] Entity-Level Threshold Calibration on {len(eval_s1)} Entities ...")
    eval_probs = {}

    for i, sid in enumerate(eval_s1):
        if sid not in s1_records:
            eval_probs[sid] = []
            continue
        sn, sa, sc = s1_records[sid]
        cands_dict = get_hybrid_candidates(sid, sn, sa, sc, top_k=30)
        if not cands_dict:
            eval_probs[sid] = []
            continue

        cids = list(cands_dict.keys())
        filt = [c for c in cids if c in s23_records]
        if not filt:
            eval_probs[sid] = []
            continue

        scores = [cands_dict[c][0] for c in filt]
        max_score = max(scores) if scores else 0.0
        cand_count = len(filt)

        fm = np.zeros((len(filt), 24), dtype=np.float32)
        for j, cid in enumerate(filt):
            cn, ca, cc = s23_records[cid]
            tsim, trank = cands_dict[cid]
            tmargin = tsim - max_score
            fm[j] = extract_24_features(sn, sa, sc, cn, ca, cc, tsim, trank, tmargin, cand_count)

        p_xgb = xgb_clf.predict_proba(fm)[:, 1]
        p_lgb = lgb_clf.predict_proba(fm)[:, 1]
        p_blend = 0.5 * p_xgb + 0.5 * p_lgb

        eval_probs[sid] = list(zip(filt, p_blend.tolist()))

        if (i + 1) % 1000 == 0:
            print(f"      {i+1}/{len(eval_s1)} entities evaluated ...")

    # Sweep threshold at ENTITY level
    best_thr = 0.5
    best_macro = 0.0
    thr_results = []

    for thr in np.arange(0.50, 0.995, 0.01):
        predicted = {}
        fp_sing = tp_t = fp_t = fn_t = tm = 0
        for sid in eval_s1:
            pids = [cid for cid, p in eval_probs.get(sid, []) if p >= thr]
            predicted[sid] = pids; tm += len(pids)
            ps = set(pids); gs = set(gt[sid])
            if not gs and ps: fp_sing += 1
            if gs:
                tp_t += len(ps & gs); fp_t += len(ps - gs); fn_t += len(gs - ps)
        mf = compute_macro_f05(eval_s1, gt, predicted)
        prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        thr_results.append((thr, mf, prec, rec, fp_sing, tm))
        if mf > best_macro:
            best_macro = mf
            best_thr = thr

    thr_results.sort(key=lambda x: -x[1])
    print(f"\n    {'Thr':>5} {'MacroF05':>9} {'Prec':>7} {'Rec':>7} {'FPSing':>7} {'Matches':>8}")
    for thr, mf, prec, rec, fps, tm in thr_results[:15]:
        m = " <<< BEST" if abs(thr - best_thr) < 0.005 else ""
        print(f"    {thr:5.2f} {mf:9.4f} {prec:7.2%} {rec:7.2%} {fps:7d} {tm:8d}{m}")

    print(f"\n============================================================")
    print(f"  FINAL V5 MODEL RESULT: OPTIMAL THRESHOLD = {best_thr:.2f}")
    print(f"  >>> MACRO F0.5 EVALUATION SCORE = {best_macro:.4f} <<<")
    print(f"============================================================")

    # Save model artifacts
    model_data = {
        "xgb_model": xgb_clf,
        "lgb_model": lgb_clf,
        "vectorizer": vectorizer,
        "threshold": float(best_thr),
        "feature_names": FEATURE_NAMES,
        "f05_score": float(best_macro),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n    Successfully saved {MODEL_PATH}!")
    print(f"    Total Runtime: {time.time() - t0:.1f} seconds ({(time.time() - t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
