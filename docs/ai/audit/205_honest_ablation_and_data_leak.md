Reviewer 2 pointed out that comparing our EBM to a Gaussian EBM is a rigged ablation (Gaussian Hessians are mathematically constant). They also found a validation data leak.

1. **Honest Ablation (`src/echo/primitives/ebm.py`):**
   - Create a new class `IdentityPrecisionEBM` that exactly mirrors `PrecisionWeightedEBM` (same MLP backbone and `energy_head`).
   - Remove the `precision_head`. Its `__call__` method must return the exact same non-linear energy, but the precision matrix must be hardcoded to `jnp.eye(self.d_state, dtype=jnp.float32)`.
   - Update `04_worm_gait_optune_ebm_architecture.py` and `05_worm_gait_aging_ebm.py` to use `IdentityPrecisionEBM` instead of `GaussianEBM`. Update plot legends in script 05 to say "Frozen Identity Precision".

2. **Fix the Data Leak (`src/benchmarks/worm_gait/core.py`):**
   - In `run_worm_gait_experiment`, `EchoRunner` is currently passed `train_young_loader` twice: `runner.run(graph, train_young_loader, train_young_loader, ...)`. Change the second argument to `eval_young_loader`.
   