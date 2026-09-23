Reviewer 2 pointed out that our baseline script evaluates an N=1 cohort and that injecting white noise ruins AR(1) and MOU entropy estimations. They also requested we stop calling the synthetically degraded data "Aged".

1. **Reframing the Dataloader (`src/data/behavior/celegans_gait_dataset.py` & callers):**
   Rename the `is_aged` parameter to `inject_synthetic_degradation` everywhere in the codebase (including all instantiations in the `src/benchmarks/worm_gait/` folder). Update the dataset docstrings to clarify this is a synthetic positive control for testing the pipeline's detection capabilities, not real aged biology. 

2. **Fix N=1 and Swap Metrics (`src/benchmarks/worm_gait/01_worm_gait_baseline_metrics.py`):**
   - We currently hardcode `traj_young = ds_young.data[0]`. Rewrite the script to iterate over **all** trajectories in both the baseline and degraded datasets.
   - Delete the "TIME DOMAIN METRICS" (CSD) and "ENTROPY METRICS" (MOU) sections entirely, as they are invalidated by synthetic white noise.
   - Keep the "SPECTRAL METRICS" (Peak Frequency) and add `calculate_ksm` from `ThermodynamicMetrics` as the primary stability metric. Note that the True biological framerate is 25Hz, so change `sampling_rate=16.0` to `25.0`.
   - Calculate the metrics for every trajectory, then log the `np.mean()` and `np.std()` for the KSM and Peak Frequency across the entire cohort.

3. **Update Plot Labels (`src/benchmarks/worm_gait/05_worm_gait_aging_ebm.py` and `10_animate_worm_gait.py`):**
   Change plot titles, log statements, and legend labels from "Young (Day 1-3)" / "Old (Day 9+)" to "Clean Baseline" / "Synthetically Degraded".
