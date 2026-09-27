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
    c.execute('PRAGMA synchronous = OFF')
    c.execute('PRAGMA journal_mode = MEMORY')
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
    conn.commit()
    return conn

def load_data(conn, source_num, filepath):
    c = conn.cursor()
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f, delimiter='\t')
        batch = []
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
            
            if len(batch) >= 50000:
                c.executemany(f'INSERT OR IGNORE INTO source{source_num} VALUES (?, ?, ?, ?, ?)', batch)
                batch = []
        if batch:
            c.executemany(f'INSERT OR IGNORE INTO source{source_num} VALUES (?, ?, ?, ?, ?)', batch)
    
    print(f"Creating indices for source{source_num}...")
    # Index on a combination of country and the first 10 characters of norm_name
    c.execute(f'CREATE INDEX IF NOT EXISTS idx_s{source_num}_match ON source{source_num}(country, substr(norm_name, 1, 10))')
    conn.commit()

if __name__ == "__main__":
    db_path = "test_resolution.db"
    conn = init_db(db_path)
    base_dir = r"C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\test"
    load_data(conn, 1, f"{base_dir}\\test_source1.tsv")
    load_data(conn, 2, f"{base_dir}\\test_source2.tsv")
    load_data(conn, 3, f"{base_dir}\\test_source3.tsv")
    conn.close()
    print("Done loading test data.")
