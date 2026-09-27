import sqlite3
import csv
import difflib
import os
import re

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\b(inc|llc|corp|corporation|ltd|limited|co|company|pvt|private)\b', '', text)
    return ' '.join(text.split())

def string_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def get_candidates(c, s1_id, country, norm_name, s1_addr):
    candidates = []
    
    # Prefix blocking: first 10 characters
    prefix = norm_name[:10] if norm_name else ''
    if prefix:
        for source in [2, 3]:
            # Exact prefix block
            c.execute(f'''
                SELECT entity_id, norm_name, business_address 
                FROM source{source} 
                WHERE country = ? AND substr(norm_name, 1, 10) = ?
            ''', (country, prefix))
            for row in c.fetchall():
                candidates.append((row[0], row[1], row[2]))
                
    # Fallback blocking: if we have less candidates, try first 5 characters + same country
    if len(candidates) < 5 and len(norm_name) >= 5:
        short_prefix = norm_name[:5]
        for source in [2, 3]:
            # Exact short prefix block
            c.execute(f'''
                SELECT entity_id, norm_name, business_address 
                FROM source{source} 
                WHERE country = ? AND substr(norm_name, 1, 5) = ?
            ''', (country, short_prefix))
            for row in c.fetchall():
                candidates.append((row[0], row[1], row[2]))

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for cand in candidates:
        if cand[0] not in seen:
            seen.add(cand[0])
            unique_candidates.append(cand)
            
    return unique_candidates

def match():
    db_path = 'test_resolution.db'
    if not os.path.exists(db_path):
        print("Database not found!")
        return
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM source1')
    total = c.fetchone()[0]
    
    # We fetchall because source1 might not be huge, but maybe it is. Let's iterate.
    c.execute('SELECT entity_id, country, norm_name, business_address FROM source1')
    
    os.makedirs('output', exist_ok=True)
    
    with open('output/candidate_pairs.tsv', 'w', newline='', encoding='utf-8') as fcand, \
         open('output/matching_results.tsv', 'w', newline='', encoding='utf-8') as fmatch:
         
         cand_writer = csv.writer(fcand, delimiter='\t')
         match_writer = csv.writer(fmatch, delimiter='\t')
         
         cand_writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
         match_writer.writerow(['source1_entity_id', 'matched_entity_ids'])
         
         count = 0
         
         # Read rows one by one to avoid memory explosion
         while True:
             row = c.fetchone()
             if not row:
                 break
                 
             s1_id, country, norm_name, addr = row
             
             # Need a separate cursor for querying candidates so we don't interfere with the source1 iterator
             c_cand = conn.cursor()
             candidates = get_candidates(c_cand, s1_id, country, norm_name, addr)
             
             cand_ids = [cand[0] for cand in candidates]
             cand_writer.writerow([s1_id, ','.join(cand_ids)])
             
             matched_ids = []
             n_addr1 = normalize_text(addr)
             for cand_id, c_norm_name, c_addr in candidates:
                 name_sim = string_similarity(norm_name, c_norm_name)
                 
                 addr_sim = 0.0
                 if n_addr1 and c_addr:
                     n_addr2 = normalize_text(c_addr)
                     addr_sim = string_similarity(n_addr1, n_addr2)
                 elif not n_addr1 and not c_addr:
                     addr_sim = 1.0 
                 
                 # Conservative matching given F0.5
                 if name_sim > 0.88:
                     matched_ids.append(cand_id)
                 elif name_sim > 0.75 and addr_sim > 0.65:
                     matched_ids.append(cand_id)
             
             match_writer.writerow([s1_id, ','.join(matched_ids)])
             
             count += 1
             if count % 10000 == 0:
                 print(f"Processed {count}/{total}")

    conn.close()
    print("Matching complete.")

if __name__ == "__main__":
    match()
