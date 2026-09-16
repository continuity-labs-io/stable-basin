**Context Files to Load / Create:**
* `src/metrics/ep_surrogates.py`
* `src/metrics/entropy_production.py` (For linear baselines)
* `src/echo/benchmarks/07_nonlinear_eeg_entropy.py` (Create)

**Task: Phase 4 The Human - Nonlinear EEG Entropy & The Decision Rule (Part 3/3)**
With both estimators and data loaders in place, we orchestrate the final decision rule to compare the nonlinear estimators against linear baselines and phase-randomized surrogates.

**Core Objectives:**

**1. The Decision Rule Orchestrator (`07_nonlinear_eeg_entropy.py`):**
* Build the final execution script that orchestrates the following comparison:
  * **Test A:** Run Estimator D on the real EEG data vs. the phase-randomized surrogates (generated via `phase_randomized_surrogate`). This isolates purely non-Gaussian irreversibility (as phase-randomization destroys non-Gaussian features but preserves the exact linear spectrum).
  * **Test B:** Train Estimator E (the ECHO harness) on the PCA-reduced EEG using `EchoRunner`. Evaluate its held-out likelihood compared to the linear MOU fit (Estimator B from Phase 1).
* **Logging:** Output a clean log summary detailing the Decision Rule. Did the nonlinear EBM learn a more accurate stationary distribution than the linear MOU? Did the Arrow-of-Time classifier successfully detect non-Gaussian irreversibility in the real data that vanished in the surrogates?

**Constraints:**
* Use the standard Python `logging` module. Keep all log messages peaceful, precise, and practical. Avoid dramatic, capitalized, or emoji-laden print statements.
