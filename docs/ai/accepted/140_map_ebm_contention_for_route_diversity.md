# Map EBM Contention to Quantify Route Diversity

**Importance**: HIGH (Catastrophic Failure Modeling).

Most aging models fail because they assume linear decay (e.g., a cell loses 1%
of its function every day). Biological systems actually fail non-linearly due to
degeneracy (multiple pathways doing the same job). By mapping contention, Stable
Basin will accurately simulate the "Gompertz law of mortality"—where the system
seems perfectly healthy until a simultaneous dual-stressor exhausts the
redundancy, causing a sudden, catastrophic collapse.

## 🔧 CORE CHANGES (Infrastructure & Math)

- `src/echo/physics/thermostat.py` & `primitives/thermalizer.py`: The
  Euler-Maruyama SDE solver currently accepts a single noise vector $\omega$. It
  must be refactored to accept a matrix or list of simultaneous, orthogonal
  stochastic perturbations ($\omega_1, \omega_2, ..., \omega_n$) to accurately
  model concurrent demands.
- `src/echo/metrics/thermal_interpretability.py`: We must add a Degeneracy /
  Route Diversity Metric. While the Trace of the Hessian measures the total
  steepness of the basin, the spectrum (number of non-zero eigenvalues) measures
  the number of distinct topological paths back to homeostasis. We will track
  the Nullity of the Hessian.

## 🎬 SCRIPTS & DEMOS (The Proof)

- `src/echo/benchmarks/concurrent_contention_benchmark.py`:
  - **Phase A**: Inject stressor $\omega_1$ (e.g., simulated thermal shock). The
    system recovers via route A.
  - **Phase B**: Reset. Inject orthogonal stressor $\omega_2$ (e.g., pH
    imbalance). The system recovers via route B.
  - **Phase C (Contention)**: Inject $\omega_1 + \omega_2$ simultaneously.
  - **The Metric**: The script tracks the Degeneracy Metric. It successfully
    detects the non-linear "gradient collapse point" where the shared
    restorative pathways become over-allocated, causing the top-down prior to
    fail and the reachable set to fragment.
