Reviewer 2 caught a validation data leak and a tautological ablation in our EBM benchmark. We need to implement a mathematically honest baseline.

1. **Create an Honest Baseline (`src/echo/primitives/ebm.py`):**
   Comparing `PrecisionWeightedEBM` to `GaussianEBM` confounds "precision weighting" with "nonlinear MLP vs parabola". Please create a new class in `ebm.py` called `IdentityPrecisionEBM`. 
   - It should be an exact structural copy of `PrecisionWeightedEBM` (keeping the exact same MLP backbone and `energy_head`).
   - Remove the `precision_head` entirely.
   - In its `__call__` method, it must return the exact same non-linear `energy` as `PrecisionWeightedEBM`, but for the precision matrix, it must simply return `jnp.eye(self.d_state, dtype=jnp.float32)`.

2. **Swap the Ablation & Fix the Leak (`src/benchmarks/worm_gait/05_worm_gait_aging_ebm.py`, `src/benchmarks/worm_gait/04_worm_gait_optune_ebm_architecture.py`, and `src/benchmarks/worm_gait/core.py`):**
   - In scripts 04 and 05, replace all imports, instantiations, and logging strings of `GaussianEBM` with `IdentityPrecisionEBM`.
   - In `05_worm_gait_aging_ebm.py`'s `plot_ablation_results`, update the titles and legends for Panel B to compare "Frozen Identity Precision" instead of "Laplace Flatline".
   - In `src/benchmarks/worm_gait/core.py`, locate the `run_worm_gait_experiment` function. `EchoRunner` is currently being passed `train_young_loader` twice (`runner.run(graph, train_young_loader, train_young_loader, key, dt=dt)`). Change the second argument to `eval_young_loader`.
