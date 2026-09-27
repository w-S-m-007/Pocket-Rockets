import sqlite3
import csv
import sys
import re

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\b(inc|llc|corp|corporation|ltd|limited|co|company|pvt|private)\b', '', text)
    return ' '.join(text.split())

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create tables
    for source in [1, 2, 3]:
        c.execute(f'''
            CREATE TABLE IF NOT EXISTS source{source} (
                entity_id TEXT PRIMARY KEY,
                business_name TEXT,
                business_address TEXT,
                country TEXT,
                norm_name TEXT
            )
        ''')
        # Create FTS tables for source 2 and 3
        if source in [2, 3]:
            c.execute(f'''
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_source{source} USING fts5(
                    entity_id UNINDEXED, 
                    norm_name,
                    tokenize="unicode61 remove_diacritics 1"
                )
            ''')
    conn.commit()
    return conn

def load_data(conn, source_num, filepath):
    c = conn.cursor()
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        batch = []
        fts_batch = []
        for row in reader:
            name = row.get('business_name', '')
            address = row.get('business_address', '')
            n_name = normalize_text(name)
            
            batch.append((
                row['entity_id'],
                name,
                address,
                row.get('country', ''),
                n_name
            ))
            
            if source_num in [2, 3]:
                fts_batch.append((row['entity_id'], n_name))
                
            if len(batch) >= 100000:
                c.executemany(f'INSERT OR IGNORE INTO source{source_num} VALUES (?, ?, ?, ?, ?)', batch)
                if source_num in [2, 3]:
                    c.executemany(f'INSERT INTO fts_source{source_num}(entity_id, norm_name) VALUES (?, ?)', fts_batch)
                    fts_batch = []
                batch = []
        if batch:
            c.executemany(f'INSERT OR IGNORE INTO source{source_num} VALUES (?, ?, ?, ?, ?)', batch)
            if source_num in [2, 3]:
                c.executemany(f'INSERT INTO fts_source{source_num}(entity_id, norm_name) VALUES (?, ?)', fts_batch)
    
    print(f"Creating indices for source{source_num}...")
    c.execute(f'CREATE INDEX IF NOT EXISTS idx_s{source_num}_norm_name ON source{source_num}(norm_name)')
    c.execute(f'CREATE INDEX IF NOT EXISTS idx_s{source_num}_country ON source{source_num}(country)')
    conn.commit()

if __name__ == "__main__":
    db_path = "business_resolution.db"
    conn = init_db(db_path)
    base_dir = r"C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\train"
    load_data(conn, 1, f"{base_dir}\\train_source1.tsv")
    load_data(conn, 2, f"{base_dir}\\train_source2.tsv")
    load_data(conn, 3, f"{base_dir}\\train_source3.tsv")
    conn.close()
    print("Done loading.")
