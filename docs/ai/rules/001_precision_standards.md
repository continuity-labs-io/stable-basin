# Precision and Mathematical Standards

This document establishes the source of truth for how we handle tensors,
precisions, and complex mathematics in the Stable Basin Benchmark codebase. Adhering to
these standards ensures we prevent underflow/overflow crashes, maintain
cross-platform deterministic behavior, and avoid runtime panics when processing
noisy biological data.

## 1. Tensor Data Types (Dtypes)

- **Deep Learning Default**: `torch.float32`. All neural network weights,
  continuous state-space models (e.g., Mamba-2), and dataset loaders must output
  `float32` by default to ensure optimal performance and compatibility on edge
  GPUs.
- **Mixed Precision Exceptions**: `torch.bfloat16` may be used strictly inside
  `torch.autocast` context managers during training loops. It should **never**
  be used for thermodynamic metrics, matrix inversions, or distance
  calculations, as it will lead to catastrophic underflow errors.
- **Scientific Computing Default**: `float64` (Double Precision). For sensitive
  non-linear matrix operations (e.g., Singular Value Decomposition (SVD),
  Dynamic Mode Decomposition (DMD), pseudo-arc length continuation) where
  precision drops lead to rank collapse, inputs must be temporarily cast to
  `float64` for the calculation.

## 2. Handling Complex Numbers & Math Domain Errors

- **Complex Eigenvalues**: When extracting eigenvalues from biological
  time-series data (e.g., via DMD), the results are frequently complex.
  Developers must explicitly define how to collapse complex numbers to physical
  reals based on the specific metric:
  - For stability metrics (e.g., Lyapunov exponents, spectral radii), take the
    magnitude: `.abs()`
  - For phase/oscillation analysis, take the real/imaginary parts explicitly:
    `.real` or `.imag`
- **Epsilon Protections**: All divisions, square roots, and logarithms must
  include an epsilon guard (e.g., `math.log(x + 1e-7)`) to prevent `NaN` or
  `-Inf` propagation.

## 3. Fallbacks for Edge Cases (Biological Flatlining)

- **Rank-Deficient Signals**: Real-world biological sensors frequently encounter
  dead channels, artifacts, flatlines, or pure uncorrelated noise. Matrix
  solvers (like SVD or OptDMD) will fail to find optimal decomposition ranks on
  pure noise.
- **Rule**: Mathematical exceptions must never crash the pipeline during
  inference. Always wrap sensitive mathematical extraction operations in
  `try-except` blocks (or suppress specific warnings with
  `warnings.catch_warnings()`) and provide scientifically valid, graceful
  fallbacks. For example:
  - If biological frames are pure noise and DMD fails, set the Local Lyapunov
    Exponent (LLE) to `0.0`.
  - If a covariance matrix is singular, fallback to an identity matrix
    representation or a stability metric of `1.0`.

## 4. Dataset Normalization

Continuous-time engines (like H-SSMs) and SVD solvers are highly sensitive to
unscaled variance.

- All dataset iterators (Optical and Electrophysiology) must standardize their
  output sequence chunks (e.g., Z-scoring or Min-Max scaling per chunk) before
  they are yielded by the `DataLoader` to prevent runaway activations.
