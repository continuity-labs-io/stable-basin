We now need to wire the new Structured EBM and Hutchinson Curvature Estimator into the configuration and factory pipeline.

1. **Update `src/benchmarks/aging_resilience/core.py` (The EBM Factory)**:
   - In `build_graph(ebm_class, key, config)`, remove the `ebm_class` argument (update all call sites in scripts 04, 05, 06, 07, etc., to stop passing it).
   - Add `from src.echo.primitives.ebm_structured import StructuredPrecisionEBM` and `from src.echo.primitives.ebm import IdentityPrecisionEBM, PrecisionWeightedEBM` to your imports.
   - Inside `build_graph`, extract the EBM types:
     `micro_ebm_type = config["observer"]["micro"].get("ebm_type", "dense")`
     `macro_ebm_type = config["observer"]["macro"].get("ebm_type", "dense")`
   - Create a local helper function:
     ```python
     def get_ebm_instance(ebm_type, d_state, hidden_size, depth, k, cfg):
         if ebm_type == "dense":
             return PrecisionWeightedEBM(d_state=d_state, hidden_size=hidden_size, depth=depth, key=k)
         elif ebm_type == "structured_low_rank":
             return StructuredPrecisionEBM(d_state=d_state, hidden_size=hidden_size, depth=depth, key=k, rank=cfg.get("precision_rank", 4))
         elif ebm_type == "identity":
             return IdentityPrecisionEBM(d_state=d_state, hidden_size=hidden_size, depth=depth, key=k)
         else:
             raise ValueError(f"Unknown ebm_type: {ebm_type}")
     ```
   - Update `micro = eqx.tree_at(...)` and `macro = eqx.tree_at(...)` to use this `get_ebm_instance` helper, passing the respective types, parameters, and configuration blocks.

2. **Update `compute_full_trace` in `src/benchmarks/aging_resilience/core.py`**:
   - Change the signature to: `def compute_full_trace(energy_fn, states, config, key=None, batch_size=1000):`
   - Read the curvature settings from the config:
     ```python
     eval_cfg = config.get("evaluation", {})
     estimator = eval_cfg.get("curvature_estimator", "exact_hessian")
     n_probes = eval_cfg.get("hutchinson_probes", 15)
     ```
   - Pass these into the `curvature_over_states` call:
     ```python
     res = curvature_over_states(
         energy_fn, states, chunk_size=batch_size, nonfinite="drop",
         estimator=estimator, hutchinson_key=key, hutchinson_probes=n_probes
     )
     ```
   - Ensure `run_aging_experiment` passes `config` and a `key` to `compute_full_trace`.

3. **Update `src/echo/harness/echo_runner.py`**:
   - In `EchoRunner.validate()`, read the estimator strategy:
     ```python
     eval_cfg = self.config.get("evaluation", {})
     estimator = eval_cfg.get("curvature_estimator", "exact_hessian")
     n_probes = eval_cfg.get("hutchinson_probes", 15)
     ```
   - When calling trace functions for the subset, import `batch_hutchinson_trace` from `src.echo.metrics.energy_landscape`. 
   - If `estimator == "exact_hessian"`, use `batch_hessian_trace`. If `estimator == "hutchinson"`, split the validation `key` to get a `hutch_key`, and use `batch_hutchinson_trace(energy_fn, x_subset, key=hutch_key, n_probes=n_probes)`.

4. **Update your YAML configs (`configs/aging_resilience.yaml` and `configs/killifish_experiments.yaml`)**:
   - Under `observer.micro` and `observer.macro`, add:
     ```yaml
     ebm_type: "dense"  # Use "structured_low_rank" for killifish/high-dim
     precision_rank: 4
     ```
   - Add a new root-level block:
     ```yaml
     evaluation:
       curvature_estimator: "exact_hessian" # Use "hutchinson" for high-dim
       hutchinson_probes: 15
     ```

5. **Verify `05_aging_ebm.py` (formerly `05_worm_gait_aging_ebm.py`)**:
   - Because we no longer pass EBM classes directly to `run_aging_experiment`, update the ablation test. Instead of passing `IdentityPrecisionEBM` and `PrecisionWeightedEBM`, you should deepcopy the `config`, modify `config_A["observer"]["micro"]["ebm_type"] = "identity"` and `config_A["observer"]["macro"]["ebm_type"] = "identity"`, and pass the modified config dicts to the experiment runners.
