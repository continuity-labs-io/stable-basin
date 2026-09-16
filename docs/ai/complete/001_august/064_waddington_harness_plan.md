# Unified Benchmark Harness Implementation Plan

We will replace the duplicated, standalone script approach with a clean,
extensible command-line harness and a Makefile. This allows you to quickly
re-run all benchmarks, or specific ones, by simply typing `make run-all`.

## User Review Required

> [!IMPORTANT] The plan involves deleting the standalone scripts
> `01_train_synthetic_benchmark.py`, `02_extrapolation_benchmark.py`, and
> `03_imputation_benchmark.py` since their logic will be entirely consolidated
> into the new harness. Let me know if you want to keep them for historical
> purposes.

## Proposed Changes

### 1. Namespaced Harness Script (`src/harness/waddington_runner.py`)

#### [NEW] `src/harness/waddington_runner.py`

To support multiple papers and architectures in the future, we will namespace
this specific benchmark suite. We will build a unified python script tailored to
the Waddington sensor fusion architecture.

- It will parse arguments: `--task_name`, `--epochs`, `--train_seq_len`,
  `--test_seq_len`, and `--models` (a list of model strings to evaluate).
- It will include a generalized version of the `save_benchmark_plot` helper.
- Outputs will be saved as `output/data/{task_name}_results.png` and
  `output/data/{task_name}_results.csv`.
- Future benchmark suites can simply add their own runners (e.g.,
  `src/harness/vision_runner.py`).

### 2. Namespaced Push-Button Automation (`Makefile`)

#### [NEW] `Makefile`

We will create a root `Makefile` exposing push-button commands to trigger the
harness, heavily namespaced to prevent collisions with future papers:

- `make waddington-synthetic`: Runs the 500-length training/eval on baseline,
  transformer, and mask-aware.
- `make waddington-extrapolation`: Runs the 2000-length stress test on the
  original 3 models.
- `make waddington-imputation`: Runs the 2000-length stress test on all 5
  imputation baseline models.
- `make waddington-all`: Sequentially executes all three benchmarks above to
  completely regenerate all results for this specific suite.

### 3. Cleanup Legacy Experiments

#### [DELETE] `src/experiments/01_train_synthetic_benchmark.py`

#### [DELETE] `src/experiments/02_extrapolation_benchmark.py`

#### [DELETE] `src/experiments/03_imputation_benchmark.py`

These are now redundant and will be removed to prevent script sprawl.

## Verification Plan

### Automated Tests

N/A

### Manual Verification

1. Run `make waddington-imputation` (or `make waddington-all`) to ensure the
   harness correctly compiles, executes the training loops, and generates the
   consolidated artifacts in `output/data/` dynamically without errors.
2. **Regression Check**: After running the new harness, compare the newly
   generated CSVs (e.g., `04_arxiv_results.csv`, `03_stress_test.csv`)
   byte-for-byte against the existing outputs in `output/data/` to guarantee
   that the unification did not alter the mathematical behavior or the structure
   of the data.
