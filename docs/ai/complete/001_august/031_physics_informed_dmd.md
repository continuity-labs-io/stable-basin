Context: The `ThermodynamicMetrics` class currently calculates the Koopman
Stability Metric (KSM) and Local Lyapunov Exponent (LLE) using a manual, naive
Truncated SVD implementation. We need to make this robust against the heavy 1/f
noise of real biological tissue by integrating `pydmd`.

Task: Refactor the `calculate_ksm` and `calculate_lle` methods to use the PyDMD
library.

1. Remove the manual `torch.linalg.svd` blocks.
2. Instantiate `pydmd.OptDMD` (Optimized Dynamic Mode Decomposition) inside the
   sliding window loop. OptDMD is highly robust to sensor noise, which is
   critical for our HD-MEA telemetry.
3. Extract the continuous-time eigenvalues (ω) directly from the fitted OptDMD
   model.
4. Calculate the local linear operator Ã's maximum eigenvalue divergence to
   bound our KSM score [0, 1] exactly as before, but backed by the optimized
   solver.
5. Ensure data types are properly managed when moving between PyTorch tensors
   and PyDMD's expected NumPy arrays. Keep the fallback graceful so edge GPUs
   don't stall on highly stable, rank-deficient biological frames.
6. Ensure unit tests are extended to cover the new PyDMD integration and
   existing tests still pass.

For environment.yml (conda-forge):

- pip:
  - pydmd

For requirements.txt (pip):

pydmd
