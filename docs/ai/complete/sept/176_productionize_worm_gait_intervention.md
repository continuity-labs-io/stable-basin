**Context Files to Load / Modify:**
* `src/echo/benchmarks/07_worm_gait_intervention.py`
* `configs/worm_gait_intervention.yaml` (To be created)

**Task: Productionize the Worm Gait Intervention Benchmark Script**
The `07_worm_gait_intervention.py` script is currently a monolithic experimental script with many hardcoded parameters and inline helper functions. We need to refactor it into a robust, production-ready benchmark.

**Core Objectives:**

**1. Configuration Extraction (YAML):**
* Create a new configuration file: `configs/worm_gait_intervention.yaml`.
* Move all hardcoded variables from `main()` into this file (e.g., `N_steps`, `num_runs`, `dt`, `lambda_A`, `lambda_B`, architectural sizes for Micro/Macro observers, random seeds).

**2. Script Refactoring:**
* Break up the giant `main()` function into clear, modular steps:
  - `setup_experiment(config)`: Load data and initialize the graphs and states.
  - `run_experiment(...)`: Execute the simulations (Runs A and B).
  - `calculate_metrics(...)`: Process the trajectories and calculate Hessian traces.
  - `plot_results(...)`: Handle all matplotlib generation.
  - `save_results(...)`: Handle output serialization.
* Move inline helper functions (like `get_traces()`) into the module scope.
* `main()` should become a simple, readable pipeline orchestrating these helper functions.

**3. Data Serialization (JSON):**
* Implement a `save_results` function to dump **all** generated data (not just the subsets used for plotting) into a JSON output file.
* This must include the full trace data for both the degraded (Run A) and therapeutic (Run B) trajectories, as well as the final calculated metrics (means, std deviations).

**4. Make Target:**
* Ensure a `worm-gait-intervention` target is added to the `Makefile` to run the updated script with the new configuration.

**Constraints:**
* Do not change the core Euler-Maruyama SDE (`simulate_sde`) mathematics or the experimental assumptions. We are only refactoring for modularity and production readiness.
