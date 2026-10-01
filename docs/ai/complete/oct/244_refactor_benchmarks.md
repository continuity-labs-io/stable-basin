# Plan: Refactor Benchmarks

## Overview
The goal of this refactor is to rename the `aging_resilience` benchmark track to `echo_resilience`, and to create a new `model_free_resilience` track (track 2) that contains the model-free resilience metrics (currently just the killifish benchmark).

## Step-by-Step Execution

### 1. Rename `aging_resilience` to `echo_resilience`
- Rename the directory `src/benchmarks/aging_resilience` to `src/benchmarks/echo_resilience`.
- Rename `configs/aging_resilience.yaml` to `configs/echo_resilience.yaml` (if it exists).
- Find and replace all string occurrences of `aging_resilience` with `echo_resilience` across the repository. This includes:
  - Imports in `src/benchmarks/echo_resilience/*.py`
  - Output directory paths (e.g., `output/benchmarks/echo_resilience`)
  - Target names and variables in `Makefile` (e.g., `aging-resilience-experiments` -> `echo-resilience-experiments`)

### 2. Create `model_free_resilience` track
- Rename the directory `src/benchmarks/killifish_lifespan` to `src/benchmarks/model_free_resilience`.
  - *Alternatively, create `src/benchmarks/model_free_resilience` and move `killifish_benchmark.py` into it.*
- Update imports in `tests/benchmarks/test_killifish_benchmark.py` to point to the new location (`src.benchmarks.model_free_resilience.killifish_benchmark`).

### 3. Update `Makefile`
- Update all `aging-resilience-*` targets to `echo-resilience-*`.
- Create a new section in the Makefile for "Track 2: Model Free Resilience".
- Add targets to run the model-free resilience benchmarks (e.g., `model-free-resilience-killifish`) which will point to `src/benchmarks/model_free_resilience/killifish_benchmark.py`.

### 4. Verification
- Run `make preflight` to ensure no imports are broken (TorchFix, Pytest).
- Run the new Makefile targets to ensure they execute correctly.
