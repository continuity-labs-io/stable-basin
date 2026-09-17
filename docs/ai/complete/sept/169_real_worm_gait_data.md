**Context Files to Load / Modify:**
* `src/data/behavior/celegans_gait_dataset.py`

**Task: Paper 1 Data - Integrate Real Biological Eigenworm Kinematics**
The dataset currently relies on a synthetic limit cycle. We need to upgrade `celegans_gait_dataset.py` to seamlessly load *real* biological C. elegans eigenworm trajectories (6D behavioral kinematics) to serve as the ground-truth data for the final Paper 1 figures.

**Core Objectives:**

**1. File Loader Utility:**
* Update the `__init__` method to accept a `data_path` parameter (e.g., `"data/raw/eigenworms.npy"` or `.csv`).
* Implement logic to check if this file exists locally. 
* If it exists, load it. Assume the file provides continuous 6D trajectories. Write a helper logic to slice these long trajectories into contiguous `seq_len` chunks to form a PyTorch tensor of shape `[num_sequences, seq_len, 6]`.

**2. Aging Split (Thermodynamic Degradation):**
* Real datasets often lack perfectly labeled "Old" vs "Young" worms in the same file. We need a way to mathematically simulate the physiological aging split for the Old Worm evaluation.
* Add an `is_aged` boolean flag to the Dataset. If `is_aged=True`, apply a calibrated Ornstein-Uhlenbeck (OU) noise process (or heavy Gaussian noise) to the biological trajectories. This perfectly represents thermodynamic degradation and the loss of precision weighting in the limit cycle.

**3. Resilient CI Fallback:**
* If the `data_path` file does NOT exist, the `except` block MUST gracefully fall back to the existing `generate_synthetic_limit_cycle()` logic. 
* The CI pipeline must never crash just because the local biological data file is missing.

**Constraints:**
* **Crucial:** Do NOT import `owmeta` or use RDF semantic web libraries. We are dealing strictly with flat time-series kinematics.
* Keep external dependencies minimal (`numpy`, `pandas`, `torch`).
* **Logging:** Use `logger.info` to clearly indicate whether the dataset successfully loaded "Real Biological Data" or if it is using the "Synthetic Fallback".
