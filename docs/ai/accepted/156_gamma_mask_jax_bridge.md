**Context Files to Load / Create:**
* `src/echo/models/observer.py` (Create or load to define the `PredictiveCodingGraph` or core physics step)
* `src/echo/harness/pytorch_jax_bridge.py` (Create)

**Task: Phase 2 Engine Foundations - Physics Constraints & Hardware Bridge**
We are beginning Phase 2 (The ECHO Engine). Before building the main Optax training loop, we must establish the foundational physics constraints and the high-speed data transfer bridge to prevent structural breakdowns.

**Core Objectives:**

**1. The Γ (Gamma) Masking Bug (Fluctuation-Dissipation Violation):**
* **The Issue:** In `observer.py` (specifically within the continuous-time observer model / physics step), masking the dissipative matrix Γ element-by-element destroys its positive-definiteness. This mathematically invalidates the stationary density $e^{-E/T}$ and ruins the Entropy Production formula.
* **The Fix:** Introduce a `use_blanket_topology` boolean flag. 
    * If `use_blanket_topology=False` (unpartitioned systems like EEG): Disable the mask entirely. Guarantee Γ is parameterized as a full-rank, learned positive-definite matrix (e.g., via Cholesky decomposition $L L^T$).
    * If `use_blanket_topology=True` (partitioned systems like Worm Gait or Hierarchical Cells): Re-parameterize Γ so it is strictly positive-definite *within* the allowed block-diagonal pattern defined by the Markov partition. Do not use naive element-wise zero-masking on a dense matrix.

**2. The PyTorch-JAX DLPack Bridge (PCIe Bottleneck):**
* **The Issue:** Passing variable-length biological sequence data from PyTorch DataLoaders to JAX via NumPy causes a CPU-GPU roundtrip, severely bottlenecking PCIe bandwidth.
* **The Fix:** Create `pytorch_jax_bridge.py`. Implement a utility function (e.g., `torch_to_jax`) that intercepts batches from a PyTorch DataLoader and converts them to JAX arrays using strictly zero-copy memory transfers.
* **Implementation:** Use `torch.utils.dlpack.to_dlpack` and `jax.dlpack.from_dlpack`. Ensure tensors remain in GPU VRAM throughout the handoff.

**Constraints:**
* Strictly maintain the framework firewall: PyTorch handles data loading; JAX handles compute. Ensure JAX does not globally import PyTorch in a way that breaks headless TPU/GPU execution.
* Use the standard Python `logging` module. Keep all log messages peaceful, precise, and practical (e.g., `logger.debug("Executing zero-copy DLPack transfer.")`). Avoid dramatic or capitalized print statements.
