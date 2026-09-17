**Context Files to Load / Modify:**
* `src/echo/benchmarks/06_worm_gait_aging_ebm.py`

**Core Objectives (`06_worm_gait_aging_ebm.py`):**

**1. Implement the 3-Loader Scientific Split:**
* In `main()`, replace the current dataset loading logic. Use a `try/except FileNotFoundError` block that attempts to load the real datasets, and falls back to instantiating three `SyntheticWormMockDataset`s if missing.
* Create exactly three loaders:
    1. `train_young_loader`: Uses `TRAIN.ts`, `is_aged=False`. (Passed to `runner.run()` to carve the basin).
    2. `eval_young_loader`: Uses `TEST.ts`, `is_aged=False`.
    3. `eval_old_loader`: Uses `TEST.ts`, `is_aged=True`.

**2. Update Evaluation & Plotting:**
* Update `get_macro_states()` calls: evaluate the "Young" traces exclusively on `eval_young_loader`, and the "Old" traces exclusively on `eval_old_loader`.
* **Panel A Fix:** Stop plotting the unreadable 17,000-step spaghetti trajectory. Change the plot call to slice the first 500 steps: `axes[0].plot(traj_young[:500, 0]...` (Make sure `traj_young` comes from the `eval_young` dataset).
* **Hessian Striding:** To get a better distribution histogram without OOM errors, stride the Hessian trace evaluation. Instead of `[:200]`, use `[::10][:1000]` on all four `tracker.batch_calculate_curvature` calls.

**Constraints:**
* Keep the dataset classes as short, dumb, and readable as physically possible.
* Do not change the underlying Equinox or Optax mathematical logic.
