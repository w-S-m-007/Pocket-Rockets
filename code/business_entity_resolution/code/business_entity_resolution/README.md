# Business Entity Resolution Pipeline

A production-grade ML pipeline for cross-source business entity matching,
designed for the ML Challenge 2026.

## Architecture

The pipeline uses a **3-stage approach**: Blocking → Feature Extraction → Classification.

1. **Blocking** (candidate generation): Multi-key in-memory inverted index
   partitioned by country. Uses 4 blocking strategies (prefix-5, sorted-tokens,
   longest-word, prefix-3) to achieve high recall while reducing the search
   space from ~17 trillion pairs to a manageable candidate set.

2. **Feature Extraction**: 12 string-similarity features per candidate pair
   computed with RapidFuzz (C-optimised):
   - Name: ratio, partial_ratio, token_sort_ratio, token_set_ratio,
     Jaro-Winkler, token Jaccard
   - Address: ratio, partial_ratio, token_sort_ratio, Jaro-Winkler, token Jaccard
   - Length difference

3. **Classification**: Scikit-learn GradientBoostingClassifier trained on
   ground-truth pairs with threshold optimised for macro-average F₀.₅.

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
python src/train.py
```
This streams through the training source files, samples ~60K S1 entities,
extracts features for positive/negative pairs, trains a GradientBoosting
classifier, optimises the decision threshold for F₀.₅, and saves `model.pkl`.

**Runtime:** ~10–15 minutes.

### Step 2: Generate predictions
```bash
python src/predict.py
```
This loads the test source files into memory, builds an in-memory blocking
index, scores all candidate pairs with the trained model, and writes:
- `output/matching_results.tsv` — final matches (for leaderboard upload)
- `output/candidate_pairs.tsv` — blocking candidate set

**Runtime:** ~20–40 minutes (depending on hardware).

If `model.pkl` is not available, use heuristic mode:
```bash
python src/predict.py --heuristic
```

### Step 3: Validate
```bash
python "C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\utils\validate_submission.py" --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir "C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\test"
```

### Step 4: Package submission
```bash
python build_submission.py
```
