**Context Files to Load / Create:**
* `src/data/eeg/sleep_edf.py` (Create)
* `src/data/eeg/lemon.py` (Create)

**Task: Phase 4 The Human - Nonlinear EEG Entropy & The Decision Rule (Part 1/3)**
We have successfully validated the ECHO Harness on the low-dimensional biological limit cycle of worm gait. Now, we return to the human EEG datasets to determine if our nonlinear estimators capture non-Gaussian irreversibility (the true thermodynamic arrow of time) that the linear phase-lag models from Phase 1 missed.

**Core Objectives:**

**1. MNE Data Loaders for EEG:**
* Create `src/data/eeg/sleep_edf.py` and `src/data/eeg/lemon.py`. 
* Implement PyTorch `Dataset` classes that utilize `mne` to load raw EEG files.
* **PCA Reduction:** Raw high-density EEG (e.g., 64-channel LEMON data) is too noisy and high-dimensional for stable SDE training. The data loaders must perform a Principal Component Analysis (PCA) projection (e.g., keeping the top 5-10 components) before yielding the fixed-length `seq_len` crops.
* **Synthetic Fallback:** Ensure both data loaders have a built-in synthetic fallback (e.g., generating stationary AR process noise or Gaussian blobs resembling PCA components) if the actual `.edf` or `.vhdr` files are not found locally. This is strictly required so that CI tests never fail. If the fallback is used, log a warning stating that the data is synthetic.

**Constraints:**
* PyTorch handles the `mne` data loading and PCA. Ensure the DLPack bridge is used for the handoff to the EchoRunner.
* MNE can be quite verbose. Ensure you suppress standard MNE `INFO` logs within the data loaders to keep the terminal output clean.
* Use the standard Python `logging` module. Keep all log messages peaceful, precise, and practical. Avoid dramatic, capitalized, or emoji-laden print statements.
