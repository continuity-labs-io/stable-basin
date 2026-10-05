# 249 Landmark Sweep

## Goal
Determine how much behaviour before age L tells us about remaining lifespan by sweeping L over [50, 60, 70, 80, 90, 100, 110]. 

## Directives
1. **Sweep Execution:**
   - Create a standalone runner script `src/benchmarks/lifespan/run_landmark_sweep.py` that iterates `L` over `[50, 60, 70, 80, 90, 100, 110]`.
   - Use the existing `score_landmark` path and functions.
   - Keep the features and model completely unchanged. No tuning after seeing the table.

2. **Metrics per L:**
   - Record `n_animals` (valid animals at that L) and `n_events` (deaths).
   - Calculate the C-index (C) for feature sets F1, F2, and F1+F2.

3. **Bootstrap Confidence Intervals (Strictly Animal-Level):**
   - Compute a 95% interval on each C by resampling **ANIMALS** with replacement (400 draws).
   - **Crucial:** You must refit the model inside each draw. Do *not* report the spread across CV repeats, as that shares the same animals and understates uncertainty.

4. **Noise Floor Estimation:**
   - Establish the noise floor at each L by shuffling the outcomes 200 times (refitting inside each shuffle).
   - Report the 2.5 and 97.5 percentiles (`null_lo`, `null_hi`).

5. **Gating (Per L):**
   - For each feature set at each L, determine the gate: output `SIGNAL` if the C-index's 2.5th percentile is strictly above the shuffle null's 97.5th percentile. Otherwise, output `NO SIGNAL`.

6. **Output Requirements:**
   - Output one combined table to the console (L by row).
   - Save a CSV containing: `(L, feature_set, n, events, C, lo, hi, null_lo, null_hi, signal)` per feature set.
   - Adhere strictly to the **Output Directory Structure** rule: output must be saved to a directory in `output/` that mirrors the script's path (e.g., `output/benchmarks/lifespan/`).
