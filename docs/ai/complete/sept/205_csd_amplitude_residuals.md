Reviewer 2 proved that running Critical Slowing Down (CSD) on raw oscillating eigenworms just measures waveform smoothness. We need to evaluate CSD on the amplitude residuals of the gait limit cycle and fix our N=1 evaluation bias.

Please overhaul `src/benchmarks/worm_gait/01_worm_gait_baseline_metrics.py`:
1. **Fix N=1:** Iterate over *all* trajectories in `ds_young` and `ds_old` (now degraded), not just `ds_young.data[0]`.
2. **Update CSD:** Delete the old `ThermodynamicMetrics` CSD and MOU logic entirely. Import `amplitude_residual_stats` from `src.data.behavior.synthetic_aging`. For each trajectory, compute `stats = amplitude_residual_stats(traj.numpy(), pair=(0, 1))`.
3. **Log Correctly:** Accumulate `amp_ar1` and `amp_var` for both the clean and degraded cohorts, and log their `np.mean()` and `np.std()`. 
4. **Fix Sample Rate:** Update the PSD peak frequency calculation to use `sampling_rate=25.0` (the true video framerate) instead of 16.0.
