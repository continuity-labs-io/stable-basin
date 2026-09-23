# Baseline Biological Metrics Evaluation

## Objective
Evaluate the raw biological datasets (Young vs. Old C. elegans worms) using the rigorous thermodynamic and spectral metrics developed in the `src/metrics/` package. The goal is to establish ground truth baselines and mathematically prove that these metrics successfully capture the expected physiological differences (e.g., increased entropy production and structural flattening in aging).

## Instructions
1. Create a new benchmark script `src/benchmarks/08_worm_gait_baseline_metrics.py`.
2. Load the `RealEigenwormDataset` for both `is_aged=False` and `is_aged=True`.
3. Compute the following metric classes on the raw trajectories:
   - **Time Domain Metrics:** Calculate Critical Slowing Down (CSD) via variance and lag-1 autocorrelation.
   - **Spectral Metrics:** Calculate phase volume structure and power spectral density.
   - **Entropy Metrics:** Calculate thermodynamic irreversibility using Multivariate Ornstein-Uhlenbeck (MOU) processes.
4. Output the results cleanly to the console, making sure to explicitly delineate the sections (Time Domain, Spectral, Entropy) so the reader understands the distinct paradigms being tested.
5. Serialize the complete results to `output/echo/benchmarks/08_worm_gait_baseline_metrics.json`.
6. Add the benchmark to the `Makefile` under a new target `worm-gait-baseline` and include it in `worm-gait-experiments`.

## Acceptance Criteria
- A clean, crash-free execution of the benchmark script.
- Console output that intuitively groups metrics into their respective scientific domains.
- A JSON payload containing the raw statistical comparisons.
