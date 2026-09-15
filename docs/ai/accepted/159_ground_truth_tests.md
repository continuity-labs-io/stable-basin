**Context Files to Load:**
* `src/metrics/entropy_production.py`
* `src/metrics/ep_surrogates.py`

**Task: Implement Stage 0 Ground-Truth Tests**
Please write the test suite `tests/test_entropy_production.py` to validate our linear EEG entropy estimators. We are establishing our scientific ground truth before moving to complex models.

**Core Objectives:**
* **MOU Ground-Truth Verification:** Implement closed-form Multivariate Ornstein-Uhlenbeck (MOU) simulations[cite: 1]. Verify that our estimators perfectly recover the exact mathematical expectation for entropy production, specifically Φ = 2q²[cite: 1].
* **Surrogate Testing:** Assert that the surrogate generation functions (both reversible Gaussian and phase-randomized) reliably produce near-zero entropy production, ensuring our baseline controls are mathematically sound[cite: 1].
* **CI Readiness:** Structure the tests for continuous integration (CI) by utilizing strict random seeding for deterministic execution[cite: 1]. 
* **Logging & Tone Constraints:** Use the standard Python `logging` module. Keep all internal log messages peaceful, precise, and practical (e.g., `logger.info("Executing MOU closed-form validation.")`). Do not use dramatic, capitalized, or emoji-laden print statements anywhere in the test suite.
