The `calculate_curvature` answers *how* to measure curvature at a point. My `HessianTraceEvaluator` answers *which* points to measure: it runs the model over fixed windows, discards burn-in, and reuses the same random noise for clean and degraded worms. The trouble is that it also snuck in its own one-line Hessian. There was a third copy too: the chunking loops in `core.compute_full_trace` and `01_waddington_collapse`. So the fix is one rule. `energy_landscape.py` owns how curvature is measured, and harnesses only decide which states get measured.

**What's in the files below:**

- **`energy_landscape.py`** replaces yours. Every existing call keeps working, including old lambda-style calls. It adds:
  - a trace-only path, since trace doesn't need the eigen-decomposition: 115 ms vs 448 ms per 2,048 states on your real `PrecisionWeightedEBM`;
  - `curvature_over_states`, which chunks at a fixed shape so it compiles once, and handles NaN by drop, keep, or raise, never by filling in a value;
  - a small `ScalarEnergy` wrapper, explained below.

  Both paths now define the trace as `jnp.trace(H)`, so they report the same number.
- **`trace_evaluator.py`** is my evaluator moved to `src/echo/harness/`, with its private Hessian deleted. It now calls your module. Delete `src/echo/metrics/hessian.py` and change the one import in `11_null_control.py`.
- **`predictive_coding_graph.patch`** moves `JointEBM` out of the `ebm` property to module level. It passes `git apply --check` against this repo.
- **`test_energy_landscape.py`**: 8 tests pass here against your real EBM. The ninth needs `torx`, which I don't have, so run that one yourself.

**Why the `ScalarEnergy` wrapper matters.** Under `eqx.filter_jit`, anything that isn't an array is treated as fixed and cached by identity. I measured what that does:

- **A fresh lambda recompiles on every call.** `echo_runner.validate` defines `energy_fn` inside the batch loop, so it recompiles on every validation batch.
- **A reused lambda can return stale results.** When the model behind it changed, it silently kept the *old weights'* answer. I didn't find that live in your repo, but it's one refactor away.
- **`graph.ebm` defines a new class on every access**, which forces a recompile even when the model is passed in properly. That's what the patch fixes.
- **Passing an `eqx.Module` instead** compiles once and always uses the current weights. `ScalarEnergy(ebm)` behaves exactly like `lambda x: ebm(x)[0]`, but as a module.

**Migrating each call site:**

| Where | Change |
|---|---|
| `core.compute_full_trace` / `compute_metrics` | Use `curvature_over_states(ScalarEnergy(graph.ebm), states)`. Delete `nan_to_num(nan=0.0)`: zero curvature means perfectly flat, so every imputed NaN counts toward your hypothesis. Log `n_nonfinite` instead. |
| `echo_runner.validate` | Build `ScalarEnergy(model.ebm, model.hull)` once, above the loop, and use `batch_hessian_trace`. Note it's evaluated at `x_init`, which is near the origin, not at visited states. Fine as a training diagnostic, not as a basin number. |
| `01_waddington_collapse` | Use `ScalarEnergy(graph.flow_factor.macro_ebm)` with `curvature_over_states`. Drop the manual chunk loop and `nan_to_num(nan=1.0)`. |
| `03_concurrent_contention` | Keep the full-spectrum call, since it uses nullity. Line 90 references `tracker`, which is never defined, so it will crash with a NameError. Use `calculate_curvature(energy, final_A)`. |
| `05_clinical_workflow_demo` | Replace both lambdas with `ScalarEnergy(...macro_ebm)`. |

`07` does the same NaN-to-zero fill on simulated trajectories. A diverged run should be counted, not moved to the origin.

**One thing to check.** The trace sums *signed* eigenvalues. On an untrained EBM at random states, every Hessian was indefinite. That says nothing about your trained model, but the new `n_negative` field makes it a one-line check. If a large share of your real states are indefinite, "trace = basin steepness" doesn't hold there.
