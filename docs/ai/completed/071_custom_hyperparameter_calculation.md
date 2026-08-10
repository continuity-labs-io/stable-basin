Context:
Apparently the heart and the brain are important for human biology. Also we are playing around with Koopman Stability Metric which requires a hyperparameter to be set... so Koopman relies on DMD which relies on SVD compression... SVD compression strictly assumes that low variance equals noise, the entire algorithm hinges on exactly where you set the cutoff rank (r) for the singular values. 

so we are going to experiment with a implement a Gavish-Donoho Bounded, Log-Scaled Spectral Gap.

The Governor: Use Gavish-Donoho to establish the absolute mathematical "roof." This firmly isolates the Marchenko-Pastur thermal noise and prevents the matrix from blooming during a Waddington Crash.

The Target: Strictly inside that safe room, calculate the curvature of the biological modes to find the structural break.

The 1/f Fix: Because biological energy scales exponentially, you must calculate the Spectral Gap in log-space.

The Curvature Trigger: If the tissue is a pulsing cardiomyocyte, the structural cliff will produce massive positive curvature, and the Spectral Gap will dynamically tighten the rank around the master rhythm. If the tissue is a thinking brain organoid (smooth pink noise), the algorithm will detect the low curvature and automatically default to the safe G-D ceiling to preserve the neural complexity.

## Implementation Details

The hyper-optimized Python implementation `calculate_dynamic_rank` has been successfully integrated into the platform:

1. **Integrated Core Logic (`src/metrics/metrics.py`)**
   - The `calculate_dynamic_rank` function was directly appended to `metrics.py`. It is clearly demarcated with a warning block indicating that it is highly experimental ("SUPER VERY MUCH RESEARCH LAND!!").
   - The `calculate_ksm` method inside `ThermodynamicMetrics` was extended to accept a `rank_method` parameter. 
   - When `rank_method="dynamic"`, the engine explicitly performs a highly efficient Truncated SVD over the sliding latent state window to calculate the singular values. The singular values are then fed into the Gavish-Donoho bounded, log-scaled spectral gap calculation to dynamically determine the optimal cutoff rank $r$ for that precise temporal frame. 
   - PyDMD's `OptDMD` is then explicitly initialized with `svd_rank=r`.

2. **Benchmarking & Performance Evaluation (`src/metrics/ksm_method_benchmark.py`)**
   - The temporal lag/accuracy benchmark (`run_accuracy_benchmark`) has been explicitly wired up to use `rank_method="dynamic"` when detecting the structural variance collapse. 
   - The raw computational speed benchmark (`run_dmd_speed_benchmark`) was refactored. The legacy threshold rank logic was ripped out and cleanly replaced with a direct call to `calculate_dynamic_rank()`, ensuring the profiling accurately reflects the slight overhead of the Log-Scaled Spectral Gap derivative calculations.
   - *Architecture Fix:* Replaced the legacy `StateSpaceEngine` in the benchmark with the new, unified `MeldEngine(mask_aware=False)` since the former was purged in the previous directive. This allowed the Exact Jacobian PALC benchmark to continue running smoothly.

3. **Validation (`pytest`)**
   - All tests in `tests/metrics/` remain fully passing, ensuring that the legacy behaviors of PyDMD defaulting to optimal singular rank are mathematically preserved when `rank_method="default"`.
