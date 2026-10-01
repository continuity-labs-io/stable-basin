Create src/benchmarks/aging_resilience/13_worm_positive_control.py. No JAX, no trained model.

- Load EigenWorms TRAIN/TEST with load_ts from 11_null_control (via importlib). Z-score with TRAIN
  statistics. Raise FileNotFoundError if files are missing; no synthetic fallback.
- For each TEST worm and slowdown s in {1.0, 1.25, 1.5, 2.0, 3.0}: y = slow_amplitude_relaxation(traj, s,
  pair=(0, 1)) (s = 1.0 returns the input). Record amplitude_residual_stats(y) (amp_ar1, amp_var) and raw
  lag-1 AR1 of channel 0.
- Noise look-alike arm: add_measurement_noise(traj, sd=0.5); record the same stats.
- Paired: for each s and the noise arm vs clean, use paired_stats from src.metrics.baseline_statistics
  on amp_ar1.
- Unpaired: 200 stratified splits (stratified_split from src.data.utils); group A clean, group B slowed.
  Power = fraction of splits with Mann-Whitney p < 0.05. Null false-positive rate = same with B clean.
- Outputs: 13_worm_positive_control.json and 13_worm_positive_control.png.
  Left panel: amp_ar1 vs s, one line per worm, plus the noise arm.
  Right panel: power vs s, with the null rate and a 0.8 reference line.
- Print the minimum detectable slowdown (smallest s with power >= 0.8) and the null false-positive rate.
- Figure title must say: "Known slowdown injected into real worm gait (test of the measure, not aged worms)".
