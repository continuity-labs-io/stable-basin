ECHO Worm Gait: The Harness + the Gaussian Baseline

Context: Prompt 1 built the data layer (CElegansGaitDataset,
SyntheticWormGaitDataset, split_by_age_cohort). This prompt builds the
ECHO Training Harness itself and the GaussianEBM baseline, so that
"which EBM to use" becomes a one-line config change. Prompt 3 (the
benchmark script) comes after this.

Read these existing files first — this prompt depends on their exact
current behavior, not an approximation of it:
  - src/echo/architecture/observer.py     (MarkovBlanketObserver — you
    will make one small addition here, see step 1)
  - src/echo/architecture/hierarchy.py    (PredictiveCodingGraph —
    confirms it does NOT expose micro/macro observers after
    construction; you get them from the standalone objects you build
    before wrapping them, exactly as step 3 below does)
  - src/echo/primitives/ebm.py            (PrecisionWeightedEBM — the
    (energy, precision) interface contract and the Cholesky-SPD trick
    to mirror in GaussianEBM)
  - src/echo/physics/dissipative.py       (same Cholesky trick, second
    reference point)
  - src/echo/metrics/thermal_interpretability.py (HessianCurvatureTracker
    — confirm you do NOT need to change this file; it already works
    unmodified against any eqx.Module with a matching __call__)
  - src/echo/primitives/thermalizer.py    (ForcedTorxThermalizer.__call__
    — READ THE step_fn CAREFULLY: at scan step t, seq[t] is injected
    into the sensory slice at position injection_start_idx, THEN the
    physics step runs. So trajectory[t]'s sensory slice is the model's
    prediction for seq[t+1], not seq[t]. Loss pairs are
    (sensory_slice(trajectory[:-1]), seq[1:]).)
  - src/echo/benchmarks/01_waddington_collapse.py (shows the pattern
    you should follow: build standalone micro/macro MarkovBlanketObserver
    objects, keep a reference to macro.ebm directly for the Hessian
    tracker, THEN wrap both in PredictiveCodingGraph. Also note the
    random-init scale of 0.1 and how injection_start_idx =
    micro.hull.d_internal.)
  - src/harness/trainer.py                (StableBasinTrainer — mirror
    its naming/logging style for EchoTrainer, it's the PyTorch analog)
  - src/harness/clinical_diagnostic_runner.py (--config YAML loading
    convention to mirror; ignore the Ray/W&B parts, not needed here)

Build the following:

1. src/echo/architecture/observer.py — ONE small, backward-compatible
   change to MarkovBlanketObserver.__init__: add a new keyword-only
   parameter `ebm: eqx.Module | None = None`. If provided, use it
   directly as self.ebm (skip constructing PrecisionWeightedEBM
   internally). If not provided, fall back to exactly the current
   behavior (build PrecisionWeightedEBM from ebm_hidden_size/ebm_depth).
   Do not reorder or remove any existing parameters —
   01_waddington_collapse.py's existing call must keep working
   completely unmodified after this change. This is the only edit to
   any existing architecture file in this prompt.

2. src/echo/primitives/gaussian_ebm.py — GaussianEBM(eqx.Module)
   The Laplace-assumption baseline from the design doc, Section 5.1.
   Same (energy, precision) return contract as PrecisionWeightedEBM:
   __call__(x) -> Tuple[scalar energy, (d_state,d_state) SPD precision].
     - Learned mean `mu: (d_state,)`.
     - Learned unconstrained `W: (d_state, d_state)`, with
       Pi = tril(W) @ tril(W).T + epsilon * I  (identical trick to
       DissipativeFriction.Gamma and PrecisionWeightedEBM's
       precision_head — reuse the same jnp.tril + jitter pattern).
     - energy(x) = 0.5 * (x - mu) @ Pi @ (x - mu), squeezed to shape ().
     - Pi does NOT depend on x — this is the entire point of the class.
       Constructor signature: (d_state: int, key: PRNGKeyArray,
       epsilon: float = 1e-4). No hidden_size/depth args — it has no
       MLP.

3. src/echo/harness/__init__.py — new empty package.

4. src/echo/harness/pytorch_jax_bridge.py
     - to_jax(tensor: torch.Tensor) -> jax.Array : detach -> .cpu() ->
       .numpy() -> jnp.asarray. This is the ONLY place a .numpy() call
       for this harness should live (the Firewall Rule from the
       original ECHO design doc — no scattering .numpy() calls through
       training code).
     - to_jax_batch(batch: dict) -> dict : apply to_jax to every
       torch.Tensor value, pass through everything else (e.g. the
       age_days int) unchanged.

5. src/echo/harness/echo_trainer.py — EchoTrainer
   Pure JAX/Equinox. Operates on ONE sequence at a time (no batch
   dimension) — this matches forced_unroll's existing unbatched
   signature; don't introduce jax.vmap-across-batch here, that's future
   scope, not needed for this shakedown.
     - __init__(self, optimizer: optax.GradientTransformation, dt: float,
       sensory_start_idx: int, sensory_dim: int)
       (optimizer is expected to already be composed with gradient
       clipping via optax.chain(optax.clip_by_global_norm(...), 
       optax.adamw(...)) — build that composition in echo_runner, not
       here, so EchoTrainer stays a thin wrapper around plain optax.)
     - loss_fn(self, graph, x_micro_init, x_macro_init, seq, key) ->
       scalar: call graph.forced_unroll(key, x_micro_init, x_macro_init,
       self.dt, seq=seq) to get trajectory [seq_len, d_micro+d_macro].
       Extract predicted sensory slice:
       trajectory[:-1, sensory_start_idx : sensory_start_idx+sensory_dim]
       and compare (MSE) against seq[1:] — per the frame-alignment note
       above. Do NOT compare trajectory[:, ...] against seq[:,...]
       directly (that's the off-by-one bug this prompt is warning you
       about).
     - make_step(self, graph, opt_state, x_micro_init, x_macro_init,
       seq, key) -> (graph, opt_state, loss), wrapped in
       @eqx.filter_jit + eqx.filter_value_and_grad(self.loss_fn). Rely
       on eqx.filter_value_and_grad's default filter (differentiates
       floating-point array leaves only) rather than hand-rolling
       eqx.filter — static hull dims/topology are already marked
       eqx.field(static=True) in the existing modules, so this should
       Just Work without extra plumbing.
     - fit(self, graph, opt_state, sequences: list[jax.Array],
       x_micro_init, x_macro_init, epochs: int, key, log_fn=print) ->
       (graph, opt_state, loss_history: list[float]). Loop epochs, loop
       sequences, call make_step, average loss per epoch, call
       log_fn(epoch, avg_loss).

6. src/echo/harness/echo_runner.py — module-level functions (no class
   needed unless you find one clearly useful):
     - build_graph(config, key) -> dict with keys:
       {"graph": PredictiveCodingGraph, "macro_ebm": macro.ebm,
       "sensory_start_idx": int, "sensory_dim": int}
       Steps: read config["architecture"], compute d_state_micro /
       d_state_macro from the four hull dims each, construct the micro
       and macro EBM via a small _build_ebm(arch, d_state, key) helper
       that dispatches on arch["ebm_type"] ("gaussian" -> GaussianEBM,
       "mlp" -> PrecisionWeightedEBM(hidden_size=arch["ebm_hidden_size"],
       depth=arch["ebm_depth"])), construct the two MarkovBlanketObserver
       instances passing ebm=<constructed ebm> (using the step-1
       change), keep the standalone `macro` object around (do not let
       it get lost inside PredictiveCodingGraph — see
       01_waddington_collapse.py's pattern), then build
       PredictiveCodingGraph(micro, macro, n_steps=1, key=...).
       sensory_start_idx = arch["d_internal_micro"],
       sensory_dim = arch["d_sensory_micro"].
     - load_dataset(config) -> torch Dataset, dispatching on
       config["data"]["type"]:
         "celegans_gait"      -> CElegansGaitDataset(...)   (Prompt 1)
         "synthetic_worm_gait"-> SyntheticWormGaitDataset(...) (Prompt 1)
         "pharmacological"    -> PharmacologicalShockDataset(...)
                                   (existing file, unchanged) — this is
                                   the "prove it's dataset-agnostic"
                                   requirement from the design doc.
       Raise ValueError on unknown type.
     - run_burn_in(bundle, dataset, config, key) -> (bundle, opt_state,
       loss_history): before training, ASSERT
       bundle["sensory_dim"] == dataset[0]["x_raw"].shape[-1], with a
       clear error message — this is the check that would have caught
       the injection-index misalignment risk from step 2 above at
       config-time instead of silently training garbage. Then build the
       optax optimizer (chain of clip_by_global_norm +
       config["training"]["clip_grad_norm"], then adamw with
       config["training"]["learning_rate"]), bridge every dataset item's
       "x_raw" to jax via to_jax, build small random x_micro_init /
       x_macro_init (scale 0.1, same pattern as
       01_waddington_collapse.py), run EchoTrainer.fit for
       config["training"]["epochs"].
     - run_validation_hook(bundle, cohort_sequences: dict[str,
       list[jax.Array]], eval_key) -> dict[str, jax.Array] (one
       Hessian-trace array per cohort label). Attach
       HessianCurvatureTracker to bundle["macro_ebm"] directly (exactly
       like 01_waddington_collapse.py does — do not try to pull it back
       out of bundle["graph"], it isn't accessible there). Use the SAME
       x_micro_init/x_macro_init and the SAME eval_key across every
       cohort and every ebm_type variant within one comparison run —
       don't re-randomize per call, or you confound the comparison with
       initialization noise instead of measuring what you intend to.
     - run(config) -> dict: the single public entrypoint.
       key -> split into graph/train/eval keys. build_graph ->
       load_dataset -> split_by_age_cohort(dataset,
       config["data"]["young_age_days"], config["data"]["old_age_days"])
       -> run_burn_in on the YOUNG cohort ONLY (never the old cohort —
       this is the design doc's core experimental control) ->
       bridge eval sequences from both cohorts to jax ->
       run_validation_hook -> return {"loss_history":...,
       "hessian_traces": {"young":..., "old":...},
       "graph": trained_graph, "config": config}.

7. configs/echo_worm_gait.yaml — create exactly:

   seed: 0
   data:
     type: synthetic_worm_gait   # switch to celegans_gait once Prompt 1's
                                   real data is downloaded
     strain: N2
     seq_len: 1500
     size: 20                    # only used by synthetic_worm_gait
     young_age_days: [1, 3]
     old_age_days: [9, 13]
   architecture:
     ebm_type: mlp                # "mlp" or "gaussian" — the whole point
     d_internal_micro: 8
     d_sensory_micro: 6
     d_active_micro: 8
     d_external_micro: 8
     d_internal_macro: 8
     d_sensory_macro: 4
     d_active_macro: 2
     d_external_macro: 2
     ebm_hidden_size: 32
     ebm_depth: 2
     temperature: 1.0
     dt: 0.01
   training:
     epochs: 50
     learning_rate: 1.0e-3
     clip_grad_norm: 1.0
   evaluation:
     output_plot: output/echo/worm_gait_decline.png
     eval_seed: 999

8. Tests (pytest, follow this repo's existing test layout):

   a) test_gaussian_ebm_constant_hessian_trace — THE critical test from
      the design doc's Section 5.2. Construct a GaussianEBM, sample ~20
      random states spread well beyond the origin (not just near zero),
      run them through HessianCurvatureTracker.batch_calculate_curvature,
      assert every hessian_trace value is equal to the first within a
      small tolerance (this is a mathematical identity, not a
      probabilistic result — it should pass exactly, tolerance is only
      for floating-point noise).
   b) test_precision_weighted_ebm_varying_hessian_trace — negative
      control: confirm a randomly-initialized PrecisionWeightedEBM's
      Hessian trace is NOT constant across the same kind of sample set.
      Without this, test (a) passing could just mean the test itself is
      trivially satisfied by any EBM.
   c) test_build_graph_ebm_dispatch — build_graph with ebm_type="gaussian"
      and ebm_type="mlp" configs, assert bundle["macro_ebm"] is an
      instance of GaussianEBM / PrecisionWeightedEBM respectively.
   d) test_run_burn_in_sensory_dim_mismatch — deliberately set
      d_sensory_micro to the wrong value and confirm run_burn_in raises
      a clear assertion error rather than training silently.
   e) test_load_dataset_dispatch — "synthetic_worm_gait" and
      "pharmacological" both return a working Dataset;
      "celegans_gait" raises FileNotFoundError gracefully when the real
      database isn't present (catch it, don't fail CI on it).
   f) test_echo_runner_run_smoke — call run(config) end-to-end with the
      yaml above but epochs reduced to ~3 and size reduced to ~6 for
      speed, using data.type="synthetic_worm_gait". Assert the result
      dict has the expected keys, loss_history has no NaNs and is
      non-increasing on average, and hessian_traces["young"] /
      ["old"] are non-empty arrays of the expected length. Run this
      smoke test with BOTH ebm_type values.
   g) Confirm (by reading the diff, or by re-running any existing test
      for it if one exists) that step 1's change to observer.py is
      purely additive and 01_waddington_collapse.py still runs
      unmodified.

Non-goals for this prompt: do not build
src/echo/benchmarks/06_worm_gait_decline.py or any plotting code — that
is Prompt 3, and it should call run() twice (once per ebm_type) and do
nothing else architectural. Do not wire real Zenodo data into any test.
