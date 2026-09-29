Now, let's create the simulator script that interprets this configuration, constructs the RMR2 arms, and predicts the lifespan extension.

1. Create a new file `src/benchmarks/aging_resilience/12_rmr2_leaderboard.py`.
2. Add imports: `os`, `json`, `yaml`, `argparse`, `logging`, `jax`, `jax.numpy as jnp`, `equinox as eqx`, `numpy as np`, `pandas as pd`, `matplotlib.pyplot as plt`, `from scipy.stats import energy_distance`.
3. Import from our framework: `from src.benchmarks.aging_resilience.core import setup_experiment`, `from src.benchmarks.aging_resilience.task_registry import get_benchmark_task`, `from src.echo.architecture.predictive_coding_graph import PredictiveCodingGraph`, `from jaxtyping import PRNGKeyArray`.
4. Create a new function `@eqx.filter_jit def simulate_rmr2_sde(graph: PredictiveCodingGraph, x0: jax.Array, lambda_Pi: float, lambda_Gamma: float, lambda_T: float, lambda_Q: float, N: int, dt: float, key: PRNGKeyArray) -> jax.Array:`
   - Inside, replicate the logic of `simulate_sde` from `core.py`, but apply the pharmacological multipliers:
     - `Q_full = jax.scipy.linalg.block_diag(Q_micro, Q_macro) * lambda_Q`
     - `Gamma_full = jax.scipy.linalg.block_diag(Gamma_micro, Gamma_macro) * lambda_Gamma`
     - Recompute `evals, evecs` using the new `Gamma_full` to get `S_full`.
     - `def energy_fn(x): return lambda_Pi * ff.joint_energy_fn(x[:d_micro], x[d_micro:])`
     - When calculating diffusion in the scan step, apply noise reduction: `diffusion = jnp.sqrt(2.0 * (0.05 / lambda_T) * dt) * (S_full @ dW)`
     - Keep the rest of the Euler-Maruyama loop and return the trajectory.
5. In `main()`:
   - Load the YAML config. Set up `logging`.
   - Load the Catnip Young Evaluation dataset to use as the healthy ground truth target `Y_true` (flatten the sensory dimensions across the batch as done in `08_lambda_sweep.py` using `task.get_dataloaders()`).
   - Call `graph, x0, key = setup_experiment(config)` to get the physics engine and the degraded initial state.
   - Run `vmap_simulate = eqx.filter_jit(jax.vmap(simulate_rmr2_sde, in_axes=(None, None, None, None, None, None, None, None, 0)))`
   
   - Iterate over `config["rmr2_pharmacology"]["arms"]`. For each arm:
     - Initialize `L_Pi = 1.0`, `L_Gamma = 1.0`, `L_Q = 1.0`, `L_T = 1.0`.
     - If `base_treated` is True, add the `baseline_shift` values from the config to the multipliers.
     - For each drug in the arm's `drugs` list, look up its `mechanism` and `effect` in the config. Add the `effect` to the corresponding `L_*` multiplier.
     - Simulate using `vmap_simulate` over `num_runs` keys.
     - Extract the sensory flat array, compute `dist_arm = energy_distance(Y_true, sensory_flat)`.
     - Store the distance. Keep track of `dist_untreated` from the "00_Control_Aged" arm.
   
   - For each arm, calculate Thermodynamic Rescue: `R_arm = 1.0 - (dist_arm / dist_untreated)`.
   - Actuarial Bridge: Convert $R$ into Predicted Lifespan Extension (Months). Let's assume max rescue adds 16.0 months: `predicted_months = max(0.0, R_arm * 16.0)`.
6. Sort results by `predicted_months` descending. Save to a CSV (`paths.output_dir/12_rmr2_leaderboard.csv`).
7. Create a horizontal bar chart (`ax.barh`) plotting the Predicted Lifespan Extension. Save to `paths.output_dir/12_rmr2_leaderboard.png`.
