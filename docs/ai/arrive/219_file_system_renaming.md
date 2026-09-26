# Prompt 5: The Great File System Renaming

Finally, now that the pipeline is completely dataset-agnostic, generalize the paths and filenames.

1. Rename the directory `src/benchmarks/worm_gait/` to `src/benchmarks/aging_resilience/`.
2. Rename all scripts inside to remove the `worm_gait_` prefix (e.g., `01_worm_gait_baseline_metrics.py` becomes `01_baseline_metrics.py`). Preserve the numbering.
3. Do a workspace-wide Find & Replace: Update all import paths from `src.benchmarks.worm_gait` to `src.benchmarks.aging_resilience`.
4. Update all hardcoded output directories inside the scripts from `output/benchmarks/worm_gait` to `output/benchmarks/aging_resilience`.
5. Update `configs/worm_gait_experiments.yaml`: 
   - Rename to `configs/aging_resilience.yaml`.
   - Add a `dataset` block at the top containing `name: "worm_gait"`.
   - Update `paths` inside it to point to `output/benchmarks/aging_resilience/`.
6. Update the `Makefile`: change `worm-gait-*` targets to `aging-resilience-*`, and update the execution paths to match the new filenames and directories.
