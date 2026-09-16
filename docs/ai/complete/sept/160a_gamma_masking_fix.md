**Context Files:**
* `src/echo/architecture/observer.py` 

**The Γ (Gamma) Masking Bug (Fluctuation-Dissipation Violation):**
* **The Issue:** In `observer.py` (specifically within the continuous-time
  observer model / physics step), masking the dissipative matrix Γ
  element-by-element destroys its positive-definiteness. This mathematically
  invalidates the stationary density $e^{-E/T}$ and ruins the Entropy Production
  formula.
* **The Fix:** Introduce a `use_blanket_topology` boolean flag. 
    * If `use_blanket_topology=False` (unpartitioned systems like EEG): Disable
      the mask entirely. Guarantee Γ is parameterized as a full-rank, learned
      positive-definite matrix (e.g., via Cholesky decomposition $L L^T$).
    * If `use_blanket_topology=True` (partitioned systems like Worm Gait or
      Hierarchical Cells): Re-parameterize Γ so it is strictly positive-definite
      *within* the allowed block-diagonal pattern defined by the Markov
      partition. Do not use naive element-wise zero-masking on a dense matrix.

**Constraints:**
* Use the standard Python `logging` module. Keep all log messages peaceful,
  precise, and practical. Avoid dramatic or capitalized print statements.
