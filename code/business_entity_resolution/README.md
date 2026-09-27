# Business Entity Resolution Pipeline

A hyper-optimized production-grade ML pipeline for cross-source business entity matching,
designed for the ML Challenge 2026.

## Architecture

The pipeline uses a **3-stage approach**: Blocking → Feature Extraction → Ensemble Classification.

1. **Blocking & Pruning**: Multi-key in-memory inverted index partitioned by country. 
   Uses 7 blocking strategies (prefix-5, sorted-tokens, tokens, address-number-alpha, etc.) 
   to achieve high recall. During inference, block sizes are capped at 100, and candidates are 
   pruned on-the-fly.

2. **Feature Extraction**: 24 rich string-similarity and categorical features per candidate pair
   computed with RapidFuzz (C-optimised) across a 12-core multiprocessing pool.
   Features include character n-gram jaccards, Jaro-Winkler, Levenshtein variants, 
   and street number extractions. Ultra-safe dual short-circuiting bypasses heavy feature 
   extraction for mathematically impossible matches.

3. **Classification**: Blended ensemble of **XGBoost (Hist)** and **LightGBM**. 
   Trained on ground-truth pairs with decision threshold optimized for macro-average F₀.₅.

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

Install with:
```bash
pip install -r requirements.txt
```

## How to Reproduce

### Step 1: Train the model
```bash
python src/train_v5.py
```
This loads the training source files, uses TF-IDF cosine similarity along with rule-based 
blocking to construct a robust dataset of positive and hard negative pairs. It extracts 
the 24 features and trains both an XGBoost and a LightGBM classifier. Finally, it blends 
their outputs and optimizes the decision threshold for F₀.₅, saving `model_v5.pkl`.

**Runtime:** ~15–20 minutes.

### Step 2: Generate predictions (Fast CPU Inference)
```bash
python src/predict_v7_final.py
```
This script leverages a completely lean memory footprint (no TF-IDF matrix loaded), 
pre-normalizes all strings to save regex overhead, and uses `multiprocessing` across all 
available CPU cores. Candidates with <25% raw similarity are instantly short-circuited to bypass 
the ML inference, leading to massive speedups. Output is written to:
- `output/matching_results.tsv` — final matches (for leaderboard upload)

**Runtime:** ~2-3 hours for all 1.7M entities depending on CPU cores.

### Step 3: Validate
```bash
python "C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\utils\validate_submission.py" --matching output/matching_results.tsv --test-dir "C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\test"
```

### Step 4: Package submission
```bash
python build_submission.py
```
