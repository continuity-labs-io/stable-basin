**Context Files to Load / Create:**
* `src/echo/models/observer.py`
* `src/echo/harness/echo_runner.py`
* `src/metrics/ep_surrogates.py`
* `src/metrics/entropy_production.py` (For linear baselines)
* `src/data/eeg/sleep_edf.py` (Create)
* `src/data/eeg/lemon.py` (Create)
* `src/echo/benchmarks/07_nonlinear_eeg_entropy.py` (Create)

**Task: Phase 4 The Human - Nonlinear EEG Entropy & The Decision Rule**
We have successfully validated the ECHO Harness on the low-dimensional biological limit cycle of worm gait. Now, we return to the human EEG datasets to determine if our nonlinear estimators capture non-Gaussian irreversibility (the true thermodynamic arrow of time) that the linear phase-lag models from Phase 1 missed.

**Core Objectives:**

**1. MNE Data Loaders for EEG:**
* Create `src/data/eeg/sleep_edf.py` and `src/data/eeg/lemon.py`. 
* Implement PyTorch `Dataset` classes that utilize `mne` to load raw EEG files.
* **PCA Reduction:** Raw high-density EEG (e.g., 64-channel LEMON data) is too noisy and high-dimensional for stable SDE training. The data loaders must perform a Principal Component Analysis (PCA) projection (e.g., keeping the top 5-10 components) before yielding the fixed-length `seq_len` crops.
* **Synthetic Fallback:** Ensure both data loaders have a built-in synthetic fallback (e.g., generating stationary AR process noise or Gaussian blobs resembling PCA components) if the actual `.edf` or `.vhdr` files are not found locally. This is strictly required so that CI tests never fail.

**2. Implement Estimators D & E:**
* **Estimator D (Model-Free Classifier):** Inside the benchmark script (or a helper module), implement a lightweight Arrow-of-Time neural network (e.g., an LSTM, 1D-CNN, or MLP classifier in Equinox) trained via binary cross-entropy to distinguish forward time-windows from reversed time-windows. This provides a pure non-Gaussian EP lower bound.
* **Estimator E (Fitted NESS EBM):** Instantiate the `PredictiveCodingGraph` utilizing the `PrecisionWeightedEBM`. 
  * **Critical:** You *must* pass `use_blanket_topology=False`. Because EEG components are a unified sensor array rather than a partitioned cellular organism, the dissipative matrix $\Gamma$ must remain a full-rank, unmasked positive-definite matrix to preserve the fluctuation-dissipation theorem.

**3. The Decision Rule Orchestrator (`07_nonlinear_eeg_entropy.py`):**
* Build the final execution script that orchestrates the following comparison:
  * **Test A:** Run Estimator D on the real EEG data vs. the phase-randomized surrogates (generated via `phase_randomized_surrogate`). This isolates purely non-Gaussian irreversibility (as phase-randomization destroys non-Gaussian features but preserves the exact linear spectrum).
  * **Test B:** Train Estimator E (the ECHO harness) on the PCA-reduced EEG using `EchoRunner`. Evaluate its held-out likelihood compared to the linear MOU fit (Estimator B from Phase 1).
* **Logging:** Output a clean log summary detailing the Decision Rule. Did the nonlinear EBM learn a more accurate stationary distribution than the linear MOU? Did the Arrow-of-Time classifier successfully detect non-Gaussian irreversibility in the real data that vanished in the surrogates?

**Constraints:**
* PyTorch handles the `mne` data loading and PCA; JAX/Equinox handles the Estimator E model training and Estimator D classifier. Ensure the DLPack bridge is used for the handoff to the EchoRunner.
* MNE can be quite verbose. Ensure you suppress standard MNE `INFO` logs within the data loaders to keep the terminal output clean.
* Use the standard Python `logging` module. Keep all log messages peaceful, precise, and practical (e.g., `logger.info("Evaluating nonlinear Estimator E on PCA-reduced LEMON dataset.")`). Avoid dramatic, capitalized, or emoji-laden print statements.
