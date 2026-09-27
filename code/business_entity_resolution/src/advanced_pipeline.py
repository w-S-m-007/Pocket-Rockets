import sqlite3
import csv
import os
import torch
from sentence_transformers import SentenceTransformer, util
from rapidfuzz.distance import JaroWinkler

def hybrid_advanced_pipeline():
    print("Initializing Hybrid GPU Entity Resolution Pipeline...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device.upper()}")
    
    # Load SentenceTransformer Model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    db_path = 'test_resolution.db'
    if not os.path.exists(db_path):
        print("Error: test_resolution.db not found. Please run load_test_db.py first!")
        return
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Preload S2 and S3 for fast lookup
    print("Pre-loading candidate dictionaries...")
    c.execute('SELECT entity_id, business_name, business_address, country FROM source2')
    s2_data = {row[0]: row for row in c.fetchall()}
    
    c.execute('SELECT entity_id, business_name, business_address, country FROM source3')
    s3_data = {row[0]: row for row in c.fetchall()}
    
    cand_dict = {**s2_data, **s3_data}
    
    def prep_text(row):
        # Format: "name address country"
        name = row[1] if row[1] else ""
        addr = row[2] if row[2] else ""
        country = row[3] if row[3] else ""
        return f"{name.lower()} {addr.lower()} {country.lower()}"

    c.execute('SELECT COUNT(*) FROM source1')
    total = c.fetchone()[0]
    
    c.execute('SELECT entity_id, business_name, business_address, country, norm_name FROM source1')
    
    os.makedirs('output', exist_ok=True)
    
    batch_size = 5000
    
    with open('output/candidate_pairs.tsv', 'w', newline='', encoding='utf-8') as fcand, \
         open('output/matching_results.tsv', 'w', newline='', encoding='utf-8') as fmatch:
         
         cand_writer = csv.writer(fcand, delimiter='\t')
         match_writer = csv.writer(fmatch, delimiter='\t')
         
         cand_writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
         match_writer.writerow(['source1_entity_id', 'matched_entity_ids'])
         
         count = 0
         
         while True:
             batch_rows = c.fetchmany(batch_size)
             if not batch_rows:
                 break
                 
             s1_texts = []
             all_cands = []
             c_cand = conn.cursor()
             
             # 1. SQLITE BLOCKING
             for row in batch_rows:
                 s1_id, s1_name, s1_addr, country, norm_name = row
                 s1_texts.append(prep_text(row))
                 
                 candidates = []
                 prefix = norm_name[:10] if norm_name else ''
                 if prefix:
                     for source in [2, 3]:
                         c_cand.execute(f'SELECT entity_id FROM source{source} WHERE country = ? AND substr(norm_name, 1, 10) = ?', (country, prefix))
                         candidates.extend([r[0] for r in c_cand.fetchall()])
                         
                 if len(candidates) < 5 and len(norm_name) >= 5:
                     short_prefix = norm_name[:5]
                     for source in [2, 3]:
                         c_cand.execute(f'SELECT entity_id FROM source{source} WHERE country = ? AND substr(norm_name, 1, 5) = ?', (country, short_prefix))
                         candidates.extend([r[0] for r in c_cand.fetchall()])
                         
                 unique_candidates = list(set(candidates))
                 all_cands.append(unique_candidates)
                 cand_writer.writerow([s1_id, ','.join(unique_candidates)])
             
             # 2. BATCHED SEMANTIC SCORING
             # Collect all unique candidate strings needed for this entire batch to encode them in bulk
             unique_cand_ids_in_batch = list(set([cid for sublist in all_cands for cid in sublist]))
             
             if unique_cand_ids_in_batch:
                 cand_texts_batch = [prep_text(cand_dict[cid]) for cid in unique_cand_ids_in_batch]
                 
                 # Bulk Encode S1 texts and Candidate texts for massive speedup
                 s1_embs = model.encode(s1_texts, batch_size=256, convert_to_tensor=True, show_progress_bar=False)
                 cand_embs_bulk = model.encode(cand_texts_batch, batch_size=256, convert_to_tensor=True, show_progress_bar=False)
                 
                 # Create a mapping of cand_id -> embedding index
                 cand_id_to_idx = {cid: idx for idx, cid in enumerate(unique_cand_ids_in_batch)}
                 
                 # 3. HYBRID CLASSIFICATION
                 for i, row in enumerate(batch_rows):
                     s1_id, s1_name, s1_addr, country, norm_name = row
                     unique_candidates = all_cands[i]
                     matched_ids = []
                     
                     if unique_candidates:
                         s1_emb = s1_embs[i]
                         
                         # Gather candidate embeddings for this specific S1 entity
                         indices = [cand_id_to_idx[cid] for cid in unique_candidates]
                         local_cand_embs = cand_embs_bulk[indices]
                         
                         # Compute cosine similarities instantly on GPU/CPU
                         cosine_scores = util.cos_sim(s1_emb, local_cand_embs)[0]
                         
                         for j, cand_id in enumerate(unique_candidates):
                             semantic_score = cosine_scores[j].item()
                             cand_name = cand_dict[cand_id][1] if cand_dict[cand_id][1] else ""
                             s1_name_clean = s1_name if s1_name else ""
                             
                             # Use RapidFuzz Jaro-Winkler
                             jw_score = JaroWinkler.normalized_similarity(s1_name_clean.lower(), cand_name.lower())
                             
                             if semantic_score > 0.88 or jw_score > 0.90:
                                 matched_ids.append(cand_id)
                             elif semantic_score > 0.80 and jw_score > 0.80:
                                 matched_ids.append(cand_id)
                                 
                     match_writer.writerow([s1_id, ','.join(matched_ids)])
             else:
                 # No candidates for this whole batch
                 for row in batch_rows:
                     match_writer.writerow([row[0], ''])
             
             count += len(batch_rows)
             print(f"Processed {count}/{total}")

    conn.close()
    print("Advanced matching complete! Outputs saved to 'output/' directory.")

if __name__ == "__main__":
    hybrid_advanced_pipeline()
