import os
import zipfile

print("Building Zip...")

with zipfile.ZipFile('business_entity_resolution_submission.zip', 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    
    # 1. output/
    if os.path.exists('output/matching_results.tsv'):
        zf.write('output/matching_results.tsv', 'output/matching_results.tsv')
        print("Added matching_results.tsv")
    if os.path.exists('output/candidate_pairs.tsv'):
        zf.write('output/candidate_pairs.tsv', 'output/candidate_pairs.tsv')
        print("Added candidate_pairs.tsv")
        
    # 2. code/
    code_base = 'code/business_entity_resolution/'
    
    # Add all files in src/
    for root, _, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                src_path = os.path.join(root, file)
                zf.write(src_path, code_base + src_path.replace('\\', '/'))
                print(f"Added {src_path}")
                
    if os.path.exists('model_v4.pkl'):
        zf.write('model_v4.pkl', code_base + 'model_v4.pkl')
        print("Added model_v4.pkl")
        
    zf.write('README.md', code_base + 'README.md')
    zf.write('requirements.txt', code_base + 'requirements.txt')
    
    # 3. Documentation
    zf.write('Documentation_template.md', 'Documentation_template.md')
    
print("Submission zip built successfully!")
