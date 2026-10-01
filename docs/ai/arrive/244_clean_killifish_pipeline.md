# 242: Clean Killifish ML Pipeline Outputs

## Current State
The ML pipeline originally built for the Worm Gait aging resilience experiment (`src/benchmarks/aging_resilience/`) was recently extended to support the Killifish dataset. However, because the configuration files (such as `killifish_experiments.yaml`) still point their output paths to `output/benchmarks/aging_resilience/`, the killifish ML models, plots, and metrics are being dumped into the exact same output folder as the original worm gait results. This creates confusion and clutters the previously clean worm gait pipeline.

## Proposed Plan

1. **Separate Output Directories via Configuration**:
   - Update `configs/killifish_experiments.yaml` so that its output paths (`model_weights`, `output_plot`, `output_metrics`) point to a dedicated killifish directory, e.g., `output/benchmarks/killifish_ml/`.
   - Update `configs/catnap_experiments.yaml` similarly, routing its outputs to `output/benchmarks/catnap_ml/` to prevent future clutter.

2. **Migrate Existing Output Files**:
   - Move the existing killifish outputs (`killifish_intervention_metrics.json`, `killifish_intervention_rescue.png`, `killifish_trained_engine.eqx`) from `output/benchmarks/aging_resilience/` to their new dedicated output directory.

3. **Rename the Core Pipeline (Optional but Recommended)**:
   - Since `src/benchmarks/aging_resilience/` now acts as a generalized EBM physics engine and multi-dataset pipeline (handling worm gait, killifish, and catnap via `task_registry.py`), the directory name is misleading. We should rename it to a generic name like `src/benchmarks/core_aging_ml/` or `src/benchmarks/ebm_pipeline/`.
   - Update all corresponding Python import paths globally.

4. **Verification**:
   - Run the killifish and worm gait ML evaluation scripts sequentially to verify that their artifacts are saved in completely isolated directories without cross-contamination.
