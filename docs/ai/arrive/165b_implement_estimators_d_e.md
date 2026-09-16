**Context Files to Load / Create:**
* `src/echo/models/observer.py`
* `src/echo/harness/echo_runner.py`
* `src/echo/benchmarks/07_nonlinear_eeg_entropy.py` (Create)

**Task: Phase 4 The Human - Nonlinear EEG Entropy & The Decision Rule (Part 2/3)**
Building upon the data loaders, we now implement the estimators to analyze the Arrow-of-Time and validate our nonlinear approach.

**Core Objectives:**

**1. Implement Estimators D & E:**
* **Estimator D (Model-Free Classifier):** Inside the benchmark script (or a helper module), implement a lightweight Arrow-of-Time neural network (e.g., an LSTM, 1D-CNN, or MLP classifier in Equinox) trained via binary cross-entropy to distinguish forward time-windows from reversed time-windows. This provides a pure non-Gaussian EP lower bound.
* **Estimator E (Fitted NESS EBM):** Instantiate the `PredictiveCodingGraph` utilizing the `PrecisionWeightedEBM`. 
  * **Critical:** You *must* pass `use_blanket_topology=False`. Because EEG components are a unified sensor array rather than a partitioned cellular organism, the dissipative matrix $\Gamma$ must remain a full-rank, unmasked positive-definite matrix to preserve the fluctuation-dissipation theorem.

**Constraints:**
* JAX/Equinox handles the Estimator E model training and Estimator D classifier.
* Use the standard Python `logging` module. Keep all log messages peaceful, precise, and practical. Avoid dramatic, capitalized, or emoji-laden print statements.
