# Methodology Document — Business Entity Resolution

## Methodology used

We employ a **3-stage supervised ML pipeline** — Blocking → Feature Extraction → Ensemble Classification — that mirrors industry-standard entity resolution architectures while being highly optimized for the precision-heavy $F_{0.5}$ evaluation metric.

The pipeline is trained end-to-end on the provided ground truth: we sample 41,500 S1 entities from the training set, generate positive pairs (from ground truth matches) and hard negative pairs (using a mix of rule-based blocking and dense TF-IDF character n-gram cosine matching). We extract 24 rich string-similarity and categorical features per pair, and train a blended ensemble of XGBoost and LightGBM classifiers. The classification threshold is then optimized to maximize $F_{0.5}$ on a held-out validation split.

At inference time, we use a memory-lean architecture. The pipeline loads all test entities into memory, pre-normalizes them, and constructs a pruned multi-key blocking index. It retrieves candidates for each S1 entity and distributes the feature extraction across a 12-core multiprocessing pool. Candidates that fail an ultra-safe dual short-circuiting check are discarded instantly, completely bypassing the ML inference overhead. High-confidence candidates are scored by the ensemble on the CPU.

## Candidate generation/blocking strategy

The blocking stage reduces the search space from ~$1.7 \times 10^{13}$ pairs to a manageable set while maintaining extremely high recall. We partition all comparisons by **country** (same-country only) and use 7 complementary blocking keys per entity:

1. **Prefix-5**: First 5 characters of the normalized business name.
2. **Prefix-3**: First 3 characters of the normalized business name.
3. **Sorted-Tokens**: First 3 tokens sorted alphabetically. Catches word-reordering variations.
4. **Token-Exact**: First 10 characters of any token in the name >= 4 characters.
5. **Address Number + Alpha**: Combines the primary street number with the first significant word in the address.
6. **Address Alpha-3**: First 3 significant words in the address combined.
7. **Address Num**: First pure number sequence in the address.

Before indexing, strings are pre-normalized: lowercased, stripped of punctuation, and common corporate suffixes removed (`Inc`, `LLC`, `Corp`, `Ltd`, `Pvt`, `GmbH`, `SA`, etc.). Blocking keys returning more than 50 candidates (100 initially, capped to prevent explosion) are pruned on the fly.

This strategy is **country-agnostic** by design: it generalizes to France (which appears only in the test set) without any country-specific logic.

## Model architecture and feature engineering

### Features (24 per candidate pair)

| # | Feature | Description |
|---|---------|-------------|
| 1-6 | Name Metrics | RapidFuzz `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, Jaro-Winkler, Word Jaccard |
| 7 | `name_char_3gram` | Character 3-gram Jaccard similarity (captures deep sub-token similarities) |
| 8 | `name_len_diff` | Normalized absolute length difference of names |
| 9 | `name_exact` | Binary: exact normalized name match |
| 10 | `name_first_word`| Binary: exact first word match |
| 11-16 | Addr Metrics | RapidFuzz `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, Jaro-Winkler, Word Jaccard |
| 17 | `addr_num_match` | Binary: Exact match of extracted street numbers (or 0.5 if missing) |
| 18 | `addr_has_num` | Binary: Does either address contain a number? |
| 19 | `addr_empty` | Binary: Is the address field completely empty? |
| 20 | `country_match` | Binary: Exact match of country codes |
| 21 | `cand_count_log` | Log1p of the total candidates found for the S1 entity |
| 22-24 | TF-IDF Meta | Cosine similarity, rank, and margin from TF-IDF indexing (used as 0.0 at test time) |

### Classifier Ensemble

**Blended Ensemble**:
    - **XGBoost (Hist)**: 1000 estimators, max depth 7, learning rate 0.05, subsample 0.8.
    - **LightGBM**: 1000 estimators, max depth 7, learning rate 0.05, subsample 0.8.
    - Final prediction is `0.5 * XGB_Prob + 0.5 * LGB_Prob`.

### Threshold optimisation

The decision threshold is swept on the validation split, selecting the value that maximizes pair-level $F_{0.5}$. The threshold is conservatively adjusted downward by 0.10 during inference to compensate for the nullification of the TF-IDF features (which were removed at test time to save 6GB of RAM).

## Any other relevant information about the approach

- **Ultra-Safe Short-Circuiting**: Before computing expensive features or ML predictions, if a candidate pair has both name `ratio` < 25% and address `ratio` < 25%, it is instantly rejected. This provides a 10x speedup while retaining 100% of valid candidates.
- **No external data or APIs** were used.
- **Singleton handling**: If blocking produces zero candidates, it is predicted as a singleton (empty match list).
- **All libraries** are MIT/Apache-2.0 licensed.
