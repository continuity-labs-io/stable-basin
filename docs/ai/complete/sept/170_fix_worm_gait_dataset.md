**Context Files to Load / Modify:**
* `src/data/behavior/celegans_gait_dataset.py`
* `src/echo/benchmarks/06_worm_gait_decline.py`
* `src/echo/benchmarks/07_insilico_reprogramming.py`

**Task: Nuke Dataset Object & Fix Pipeline Warts**
The `CElegansGaitDataset` class has become an over-engineered monolith. We must strip it down and separate it into two dead-simple classes. Additionally, we need to apply Z-score normalization to prevent EBM activation saturation, and fix a serialization mismatch between the two benchmark scripts.

**Core Objectives (`celegans_gait_dataset.py`):**

**1. Create `RealEigenwormDataset` (Delete the old `CElegansGaitDataset`):**
* **Responsibility:** Only loads real biological data. No synthetic fallbacks.
* **Initialization:** Accepts `data_path`, `seq_len`, and `is_aged`.
* **Logic:** 
    * Parse the `.ts`, `.npy`, or `.csv` file. If it doesn't exist, do NOT catch it. `raise FileNotFoundError`.
    * **Normalization (Critical Fix):** Immediately compute the global mean and standard deviation across all loaded trajectories. Apply `(x - mean) / (std + 1e-8)` to all data. This prevents the EBM from saturating.
    * If `is_aged=True`, apply the thermodynamic noise degradation (`traj * 0.5 + torch.randn_like(traj) * 0.2`).
    * Filter out trajectories shorter than `seq_len`.
* **Behavior:** `__getitem__` yields random, contiguous `seq_len` crops.

**2. Create `SyntheticWormMockDataset`:**
* **Responsibility:** Purely for CI smoke tests. No file loading.
* **Initialization:** Accepts `seq_len` and `num_samples`.
* **Logic:** Generates simple, deterministic 6D sine waves. No aging logic.

**Core Objectives (`06_worm_gait_decline.py`):**

**3. Dataset Instantiation & Plotting Warts:**
* Update imports to use `RealEigenwormDataset`.
* Wrap the dataset instantiation in a `try/except FileNotFoundError` block in the benchmark script. Try to load `"data/worm/EigenWorms_TRAIN.ts"`. If missing, log a warning and fall back to instantiating `SyntheticWormMockDataset`.
* **Panel A Fix:** Stop plotting the entire 17,000-step spaghetti trajectory. Change the plot call to slice the first 500 steps: `axes[0].plot(traj_young[:500, 0]...`
* **Hessian Striding:** To get a better distribution histogram without OOM errors, stride the Hessian trace evaluation. Instead of `[:200]`, use `[::10][:1000]` on all four `tracker.batch_calculate_curvature` calls.

**Core Objectives (`07_insilico_reprogramming.py`):**

**4. The Handoff & Reality Fix:**
* **Serialization Fix:** Change the `eqx.tree_deserialise_leaves` target path to exactly match script 06: `"output/echo/benchmarks/06_worm_gait_decline_trained_engine.eqx"`
* **Initial State Fix:** When instantiating the dataset to grab `x0`, use `RealEigenwormDataset(data_path="data/worm/EigenWorms_TEST.ts", seq_len=10, is_aged=True)`. (Use a `try/except` block to fall back to the synthetic dataset if the file is missing).

**Constraints:**
* Keep the dataset classes as short, dumb, and readable as physically possible.
