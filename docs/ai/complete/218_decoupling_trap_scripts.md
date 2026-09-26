# Prompt 4: Decoupling the "Trap" Scripts (01, 10, 11)

Refactor the domain-heavy scripts: `01_worm_gait_baseline_metrics.py`, `10_animate_worm_gait.py`, and `11_null_control.py`.

1. **In `01_worm_gait_baseline_metrics.py`**:
   - Remove the `amplitude_residual_stats` and `SpectralMetrics` imports.
   - Retrieve the dataset using `task.get_raw_datasets(config)`.
   - In the `evaluate_cohort` loop, remove the hardcoded AR1, variance, and peak frequency code. Simply call `metrics = task.compute_domain_metrics(traj.numpy())` and dynamically aggregate whatever keys the dictionary returns.

2. **In `10_animate_worm_gait.py`**:
   - Remove the `create_worm_gait_animation` import.
   - Retrieve data using `task.get_raw_datasets(config)`.
   - Call `task.render_animation(young_data, old_data, output_path, frames=500, fps=30)`.

3. **In `11_null_control.py`**:
   - Remove the `amplitude_residual_stats` import, but **KEEP** `slow_amplitude_relaxation` (this is the explicit degradation function we want to inject).
   - Instantiate `task = get_benchmark_task(config)`.
   - Down in the `severities` loop, apply the change via the higher-order function: `x = task.apply_dataset_change(traj, change_fn=slow_amplitude_relaxation, slowdown=s, pair=pair) if s != 1.0 else traj`.
   - Replace `amplitude_residual_stats(x, pair)` with `task.compute_domain_metrics(x)`. Ensure the script dynamically consumes the dictionary keys.
