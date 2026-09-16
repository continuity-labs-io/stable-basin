**Context Files to Load / Create:**

- `src/echo/harness/pytorch_jax_bridge.py` (Create)

**Core Objective:**

**The PyTorch-JAX DLPack Bridge (PCIe Bottleneck):**

- **The Issue:** Passing variable-length biological sequence data from PyTorch
  DataLoaders to JAX via NumPy causes a CPU-GPU roundtrip, severely
  bottlenecking PCIe bandwidth.
- **The Fix:** Create `pytorch_jax_bridge.py`. Implement a utility function
  (e.g., `torch_to_jax`) that intercepts batches from a PyTorch DataLoader and
  converts them to JAX arrays using strictly zero-copy memory transfers.
- **Implementation:** Use `torch.utils.dlpack.to_dlpack` and
  `jax.dlpack.from_dlpack`. Ensure tensors remain in GPU VRAM throughout the
  handoff.

**Constraints:**

- Strictly maintain the framework firewall: PyTorch handles data loading; JAX
  handles compute. Ensure JAX does not globally import PyTorch in a way that
  breaks headless TPU/GPU execution.
- Use the standard Python `logging` module. Keep all log messages peaceful,
  precise, and practical (e.g.,
  `logger.debug("Executing zero-copy DLPack transfer.")`). Avoid dramatic or
  capitalized print statements.
