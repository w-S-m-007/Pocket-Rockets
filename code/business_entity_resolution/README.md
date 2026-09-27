# 🚀 Pocket Rockets: Business Entity Resolution

A hyper-optimized, production-grade Machine Learning pipeline for cross-source business entity matching. Built for the **ML Challenge 2026**, this project tackles the problem of resolving entity records (companies, restaurants, stores) across multiple messy data sources using advanced string similarity metrics, robust rule-based blocking, and a blended tree-based ensemble classifier.

## 🎯 Project Overview

Entity resolution (or record linkage) across large databases is computationally expensive. Comparing 1.7 million test entities against 10 million candidate entities results in **17 trillion possible pairs**. 

To solve this within a strict 15-hour time budget while maximizing the $F_{0.5}$ score (which heavily penalizes false positives), we developed a **3-stage pipeline**:

1. **Intelligent Blocking & Pruning** (Reduces 17 trillion pairs to just a few million)
2. **Rich Feature Extraction** (Computes 24 string and categorical similarities per pair)
3. **Ensemble Classification** (Blends XGBoost and LightGBM for maximum precision)

---

## 🏗️ Architecture & Methodology

### 1. Blocking & Pruning (Candidate Generation)
We partition all comparisons strictly by **country** (e.g., US entities only match with US entities). We then use **7 complementary blocking keys** to retrieve candidates from an in-memory inverted index:

- **Prefix-5**: First 5 characters of the normalized business name.
- **Prefix-3**: First 3 characters of the normalized business name.
- **Sorted-Tokens**: First 3 tokens of the name sorted alphabetically (catches word-reordering like "Bakery New York" vs "New York Bakery").
- **Token-Exact**: First 10 characters of any token >= 4 characters.
- **Address Number + Alpha**: Extracts the primary street number and combines it with the first significant word in the address (e.g., "123_Main").
- **Address Alpha-3**: First 3 significant words in the address combined.
- **Address Num**: First pure number sequence in the address.

**Dynamic Pruning:** If a blocking key maps to more than `25` candidates during inference, it is ignored. This prunes highly generic keys (like "Inc" or "Street") and prevents a combinatorial explosion, guaranteeing fast execution times.

### 2. Feature Engineering
For every candidate pair that survives blocking, we extract **24 features** using the C++ optimized `RapidFuzz` library:

- **Name Metrics**: Levenshtein Ratio, Partial Ratio, Token Sort Ratio, Token Set Ratio, Jaro-Winkler, and Token Jaccard.
- **Address Metrics**: Same as above, but applied to the address field.
- **Deep Metrics**: Character 3-gram Jaccard (catches deep sub-token similarities), normalized length differences.
- **Categorical Extraction**: Exact street number matches, exact first-word matches, and country code validation.

**Ultra-Safe Dual Short-Circuiting:** If a candidate pair has a raw name match `< 25%` AND a raw address match `< 25%`, it is **instantly discarded** before any complex features are computed. This saves millions of redundant calculations.

### 3. Classification Ensemble
Instead of relying on a single model, we blend two gradient-boosted architectures:
- **XGBoost (Hist Tree Method)**
- **LightGBM**

Both models are trained on a carefully sampled dataset of 41,500 entities (containing hard negatives mined via TF-IDF). The final probability is calculated as:
`Prediction = (0.5 * XGB_Prob) + (0.5 * LGB_Prob)`

The decision threshold is optimized on a held-out validation set specifically to maximize the $F_{0.5}$ metric.

---

## ⚡ Performance Optimizations (V7)

To ensure the prediction script finishes in under 3 hours, the pipeline features several extreme optimizations:
- **No TF-IDF at Inference**: TF-IDF matrices consume ~6GB of RAM. We use TF-IDF for training (to find hard negatives) but completely drop it at inference, relying purely on the rule-based index.
- **Pre-Normalization**: Strings are lowercased and stripped of punctuation *once* when loaded into memory, saving millions of redundant Regex compilations during the prediction loop.
- **Multiprocessing**: The candidate feature extraction is spread across **12 CPU cores** using a chunked `multiprocessing.Pool` to maximize hardware utilization and bypass Python's GIL.
- **ML Skipping**: Short-circuited candidates are never passed to the ML ensemble, dropping the prediction matrix size from ~500,000 rows to just ~5,000 rows per batch.

---

## 🛠️ Setup & Installation

Requires **Python 3.10+**.

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/w-S-m-007/Pocket-Rockets.git
cd Pocket-Rockets
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### Step 1: Train the Models
Train the XGBoost and LightGBM models. This script loads the training data, builds the TF-IDF matrix for hard-negative mining, extracts features, trains the models, optimizes the threshold, and saves `model_v5.pkl`.

```bash
python src/train_v5.py
```
*Estimated Runtime: 15-20 minutes.*

### Step 2: Generate Predictions (Fast CPU Inference)
Run the V7 hyper-optimized inference script. This will stream the 10 million test records, build the pruned blocking index, multiprocess the candidate evaluations, and output the final matches.

```bash
python src/predict_v7_final.py
```
*Estimated Runtime: 2.5 - 3 hours (for all 1.7M S1 entities).*

### Step 3: Validate Output
Validate the format of the output file against the challenge guidelines.

```bash
python "C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\utils\validate_submission.py" \
    --matching output/matching_results.tsv \
    --test-dir "C:\Projects\POCKET ROCKETS\extracted_resource_py\student_resource\dataset\test"
```

### Step 4: Package Submission
Automatically zip the source code, output files, and `.pkl` model into a submission-ready package.

```bash
python build_submission.py
```

---
*Built by Pocket Rockets.*
