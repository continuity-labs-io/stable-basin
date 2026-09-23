# 198: Codified ML Experiment Pipeline

## Goal
To codify a reproducible 6-phase machine learning experiment pipeline that prevents technical debt and cyclical dependencies.

## Implementation Plan

1. **Extract Shared Engine Logic**
   - Extract shared boilerplate (e.g., `build_graph`, `setup_experiment`, `simulate_sde`) from individual experiment scripts.
   - Place them into a new `src/benchmarks/worm_gait/core.py` module.
   - This removes the hacky `importlib` workarounds that were previously used for cross-script dependencies.

2. **Standardize Sequence (1-10)**
   - Rename and logically order all benchmark scripts from Phase 0 (Data Grounding) up to Phase 5 (Visualization):
     - `01_worm_gait_baseline_metrics.py` (was `04`)
     - `02_infer_biological_lambda.py` (was `10`)
     - `03_worm_gait_aging_ssm.py` (was `08`)
     - `04_worm_gait_aging_transformer.py` (was `09`)
     - `05_optune.py` (was `01`)
     - `06_worm_gait_aging_ebm.py` (was `02`)
     - `07_worm_gait_intervention.py` (was `03`)
     - `08_worm_gait_lambda_sweep.py` (was `06`)
     - `09_pharmacological_translation.py` (was `07`)
     - `10_animate_worm_gait.py` (was `05`)

3. **Output Path Synchronization**
   - Update internal variables in all 10 Python files to mirror the new prefix numbers (e.g. `02_worm_gait_decline_trained_engine.eqx` -> `06_...`).
   - Update `configs/worm_gait_intervention.yaml` to point to the correct weights.
   - Update all `\includegraphics` and LaTeX macros inside `paper/sharpening_the_tack/sharpening_the_tack.tex` and `.j2` template to point to the correct updated image files.

4. **Update Build System (`Makefile`)**
   - Reorder the `worm-gait-*` targets in strict chronological order from 01 to 10.
   - Update the `worm-gait-experiments` meta-target to execute the full pipeline sequentially.
   - Add a `reproduce-paper` target that executes the complete pipeline and then compiles the PDF using the `paper` target.

5. **Enshrine the SOP**
   - Append the "Codified ML Experiment Pipeline (SOP)" rule directly into the workspace's `.agents/AGENTS.md` file so that all future AI agents will follow this rule strictly when organizing experiments.

6. **Verification**
   - Run `python -m py_compile` and `make preflight` to ensure all tests pass and there are no syntax errors.
