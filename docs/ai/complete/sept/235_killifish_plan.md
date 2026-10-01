# Implementation Plan for Killifish Lifespan Benchmark

This document outlines the plan to implement the instructions in `docs/ai/audit/235_killifish.md` inside `src/benchmarks/lifespan/killifish_benchmark.py` and `tests/test_killifish_benchmark.py`.

## 1. Step 0: Data Audit Extension
- **Current state**: The script simply checks `L=70` and `L=100` and reports if any of them reach a count of 40.
- **Planned changes**:
  - Evaluate landmarks `L` in `{70, 100}`. Check for count `>= 40`.
  - If multiple pass, pick the one with more fish (break ties with `70`).
  - If both fail, evaluate `L` in `{50, 60}` and stop after printing the counts.
  - The chosen landmark `L` will be passed down to the rest of the script for feature extraction and evaluation.
  - Make sure the per-h5-file properties (`n_rows`, `columns` ending in `_m` or `_s`) are reported into `killifish_audit.json`.

## 2. Step 1: Feature Extraction
- **Block Averaging**:
  - In `compute_features_for_fish`, if a session has `> 50,000` rows, compute `block_size = ceil(n_rows / 50000)`.
  - Truncate the array to a multiple of `block_size`, then reshape and average across the blocks so the final shape is `(n_blocks, n_features)`. Record `block_size` for the outputs.
- **FFT-based Autocorrelation**:
  - Rewrite the `iact` loop in `compute_f1_f2`.
  - Vectorize the ACF computation across all features using `numpy.fft` or `scipy.fft`. Compute `acf = irfft(rfft(X, N) * conj(rfft(X, N)))` up to a lag of 500.
  - Calculate `iact` by summing `acf[1:k]` up to the first zero-crossing for each feature, then `iact = 1 + 2 * acf_sum`.
  - Delete the old Python `for` loop.
- **Caching**:
  - Implement a parquet cache at `output/benchmarks/lifespan/session_features.parquet`.
  - The cache will store the per-session features (F1 mean, F2 var, F2 lag1_acf, F2 iact) keyed by the absolute or relative h5 file path.
  - Before processing an h5 file, the script will check the cache and skip h5 loading/computation if present.

## 3. Step 2: Evaluation
- **Data Filtering**:
  - Filter fish alive at the selected landmark `L` with `>= 3` sessions before `L`.
  - Define `T = lifespan - L` and `E = 1` if dead else `0`.
- **Feature Sets**:
  - Assemble `F1` (mean and SD across sessions of F1 session features).
  - Assemble `F2` (mean across sessions and age-slope of F2 session features).
  - Assemble `F1+F2` by concatenating both.
- **Cross-Validation**:
  - Use 5-fold CV over fish, repeated 20 times (seeds 0-19).
  - Use `StandardScaler` and `PCA(n_components=min(10, n_features, n_train - 1))` fitted on the training split.
  - Train `CoxPHFitter(penalizer=0.1)` on the training split and predict the out-of-fold risk on the validation split.
  - Pool all out-of-fold predictions within a single repeat to compute *one* overall C-index per repeat.
- **Null Model**:
  - Per repeat, shuffle the `(T, E)` pairs across the fish indices and run the exact same `F1+F2` CV pipeline.
- **Metrics and Gates**:
  - Compute mean and 95% CI (2.5th to 97.5th percentiles) of the C-index across the 20 repeats for `F1`, `F2`, `F1+F2`, and `Null`.
  - Compute `Delta C = C(F1+F2) - C(F1)` for each repeat, then find the mean and CI.
  - Evaluate `G1`, `G2`, and `G3` and mark PASS/FAIL.

## 4. Step 3: Outputs
- **JSON & CSV**:
  - Output all numbers (counts, metric means, percentiles, gate results) to `killifish_results.json`.
  - Append rows for each feature set to `results/lifespan_benchmark.csv`.
- **Plots**:
  - `killifish_cindex.png`: Bar plot showing mean C-index and error bars (using the 95% CIs) for `F1`, `F2`, `F1+F2`, and `Null`. Draw a dashed line at `y=0.5`.
  - `killifish_km.png`: Kaplan-Meier plot using `lifelines`. Split fish into tertiles based on their out-of-fold risk from the `F1+F2` repeat-0 predictions. Include log-rank p-value in the title.
- **Cleanup**: Delete the unused `run_evaluation_with_bootstrap` stub.

## 5. Tests
Create `tests/test_killifish_benchmark.py`:
1. **FFT Autocorrelation Test**: Compare the output of the new FFT-based ACF/IACT function against a simple unvectorized nested-loop implementation on a random 2D array, asserting `allclose` to `1e-6`.
2. **Synthetic CoxPH Test (Positive)**: Generate synthetic fish whose lifespans are perfectly ordered by a single continuous feature (e.g., lag-1 ACF). Pass it through the CV pipeline and verify that Gate 1 (`G1: F1+F2 2.5th percentile > 0.5`) easily passes.
3. **Synthetic CoxPH Test (Null)**: Pass the same dataset with randomly shuffled outcomes and verify that the mean C-index is in `[0.45, 0.55]`.
