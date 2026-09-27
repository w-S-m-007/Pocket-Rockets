"""
Ultra-Fast Entity Resolution Prediction
=======================================
Uses precise blocking keys and a highly-tuned Jaro-Winkler + Token-Set heuristic.
Finishes in minutes instead of hours.
"""

import csv
import os
import re
import sys
import time
from collections import defaultdict
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

TEST_DIR  = r"C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\test"
OUTPUT_DIR = "output"
MAX_BLOCK_SIZE = 1000

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

def get_keys(norm_name: str, country: str) -> list:
    keys = []
    c = country.strip().lower() if country else ""
    if not norm_name: return keys
    
    # Key 1: Exact first 6 characters
    if len(norm_name) >= 4:
        keys.append(f"{c}|p6|{norm_name[:6]}")
        
    # Key 2: Longest word (if distinct)
    tokens = norm_name.split()
    if tokens:
        longest = max(tokens, key=len)
        if len(longest) >= 6:
            keys.append(f"{c}|lw|{longest[:8]}")
            
    return keys

def heuristic_match(n1, a1, n2, a2) -> bool:
    if not n1 or not n2: return False
    
    name_jw = JaroWinkler.normalized_similarity(n1, n2)
    if name_jw > 0.94: return True
    
    if name_jw > 0.85:
        addr_jw = JaroWinkler.normalized_similarity(a1, a2) if a1 and a2 else 0.0
        if addr_jw > 0.85: return True
        
        name_tset = fuzz.token_set_ratio(n1, n2) / 100.0
        if name_tset > 0.93: return True
        
    return False

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("[1/3] Loading candidate records and building index...")
    entity_records = {}
    block_index = defaultdict(set)
    
    for src_name in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(TEST_DIR, src_name)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row["entity_id"]
                name = row.get("business_name", "")
                addr = row.get("business_address", "")
                c = row.get("country", "")
                nn = normalize_name(name)
                
                # Store normalized names for faster comparison
                entity_records[eid] = (nn, addr.lower().strip() if addr else "")
                
                for k in get_keys(nn, c):
                    block_index[k].add(eid)

    print(f"      Index built! Unique keys: {len(block_index):,}")
    
    # Filter out massive blocks
    for k in list(block_index.keys()):
        if len(block_index[k]) > MAX_BLOCK_SIZE:
            del block_index[k]
            
    print("[2/3] Processing S1 entities...")
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    
    total = 1732544
    matches = 0
    cands = 0
    
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(cand_path, "w", encoding="utf-8", newline="") as fc, \
         open(match_path, "w", encoding="utf-8", newline="") as fm:
         
        reader = csv.DictReader(f_in, delimiter="\t")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        
        for i, row in enumerate(reader):
            s1_id = row["entity_id"]
            name = row.get("business_name", "")
            addr = row.get("business_address", "")
            c = row.get("country", "")
            
            nn = normalize_name(name)
            na = addr.lower().strip() if addr else ""
            
            cand_set = set()
            for k in get_keys(nn, c):
                if k in block_index:
                    cand_set.update(block_index[k])
                    
            cands += len(cand_set)
            
            matched = []
            for cid in cand_set:
                c_nn, c_na = entity_records[cid]
                if heuristic_match(nn, na, c_nn, c_na):
                    matched.append(cid)
                    
            matches += len(matched)
            
            fc.write(f"{s1_id}\t{','.join(cand_set)}\n")
            fm.write(f"{s1_id}\t{','.join(matched)}\n")
            
            if (i+1) % 100_000 == 0:
                print(f"      {i+1:,} / {total:,} ... cands={cands:,}, matches={matches:,}")

    elapsed = time.time() - t0
    print(f"\n[3/3] Done in {elapsed:.1f} seconds! ({elapsed/60:.1f} min)")
    print(f"      Total matches: {matches:,} (highly precise)")
    
if __name__ == "__main__":
    main()
