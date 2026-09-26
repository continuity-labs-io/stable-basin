# Plan: Refactor Generic Metrics and Utilities out of 11_null_control.py

## Context
The `11_null_control.py` script currently contains several generic statistical tests, data splitting routines, and a computationally heavy `HessianTraceEvaluator` that are baked directly into the benchmarking script. As pointed out during code review, these functions are widely applicable across different experiments and should not be isolated to a single benchmark script. Abstracting them will prevent code duplication ("reinventing the wheel") and improve testability and maintainability for future papers and benchmarking suites.

## Objectives
Extract generic mathematical, statistical, and evaluation routines from `src/benchmarks/worm_gait/11_null_control.py` into shared `src/` modules. Update the benchmark script to import these shared functions instead.

## Execution Directives

1. **Extract General Statistical Tests**
   - **Target File:** `src/metrics/baseline_statistics.py` (or a newly created `src/metrics/hypothesis_testing.py`).
   - **Functions to Move:**
     - `hedges_g(a, b)`: Hedges' g calculation.
     - `unpaired_stats(a, b, rng, n_perm, n_boot)`: Unpaired permutation tests and bootstrap CIs.
     - `paired_stats(clean, degraded, rng, n_perm, n_boot)`: Paired significance tests and signflips.
     - `naive_timestep_ks(steps_a, steps_b)`: Kolmogorov-Smirnov timestep aggregation.
     - `resplit_stats(clean_s, cond_s, labels, n_splits, seed)`: Resplitting evaluation for false-positive rates.
   - **Constraint:** Ensure that any `scipy.stats` imports required (e.g., `ks_2samp`, `mannwhitneyu`) are migrated alongside these functions.

2. **Extract Data Utilities and Splitting Logic**
   - **Target File:** Create or use an existing module like `src/data/utils.py` or `src/metrics/data_splits.py`.
   - **Functions to Move:**
     - `zscore_fit(trajs)`: Global mean and standard deviation fitting.
     - `stratified_split(labels, rng, frac)`: Label-stratified data splitting.
     - `window_starts(T, seq_len, k)`: Trajectory window calculations.

3. **Extract General I/O Utilities**
   - **Target File:** `src/utils/io.py` (create if it doesn't exist).
   - **Functions to Move:**
     - `sha256(path)`: Standard file hashing.

4. **Abstract the Hessian Trace Evaluator**
   - **Target File:** `src/metrics/hessian_trace.py` or `src/echo/metrics/hessian.py`.
   - **Class to Move:** `HessianTraceEvaluator`.
   - **Rationale:** The evaluator uses Equinox/JAX to compute the trace of the Hessian of the energy landscape via forced unrolling. This is a core thermodynamic metric of the `Stable Basin` project and must be available universally, not just in `11_null_control.py`.

5. **Refactor the Benchmark Script**
   - **Target File:** `src/benchmarks/worm_gait/11_null_control.py`.
   - **Action:** Delete the local implementations of the extracted functions and class.
   - Add standard import statements pulling the functions from their respective new generic modules.

6. **Testing and Verification**
   - Add targeted unit tests for the extracted statistical functions in `tests/metrics/test_baseline_statistics.py` (ensuring the `ARRANGE`, `ACT`, `ASSERT` block structure is strictly followed).
   - Add unit tests for the `HessianTraceEvaluator` to ensure it integrates correctly outside of the `11_null_control` script.
   - Run `python -m src.benchmarks.worm_gait.11_null_control --stub --train-ts data/worm/EigenWorms_TRAIN.ts --test-ts data/worm/EigenWorms_TEST.ts` (or equivalent test) to verify that the plumbing still works seamlessly after refactoring.
