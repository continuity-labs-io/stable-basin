DESIGN DOCUMENT: ECHO / Worm Gait — Locomotor Limit-Cycle Decline
1. Objective
This is our first ECHO experiment, and it's deliberately a toy. The point is not to produce a publishable aging result from worm gait — it's to answer a cheap, falsifiable question before we commit real effort to partial cellular reprogramming (Option A):
Does letting the energy function be a learned, potentially multimodal landscape (E_θ) capture anything about biological aging that a plain Gaussian/Laplace-assumption potential can't?
C. elegans locomotor decline is the right toy for this because:
Gait is a literal limit cycle (a sinusoidal undulation), not a metaphor — so Γ (dissipative settling) and Q (solenoidal circulation) have an obvious, visually checkable target in a low-dimensional space.
The state space is ~6-dimensional ("eigenworm" posture coordinates), not ~1024-channel voltage — cheap enough to train two EBM variants side-by-side and compare them directly, rather than trusting a single run.
Public data already exists: the Open Worm Movement Database (Yemini et al. 2013, A database of Caenorhabditis elegans behavioral phenotypes) contains thousands of single-worm recordings, spanning wild-type worms across the aging trajectory (locomotor decline is well documented starting ~day 2–3 of adulthood), with posture already commonly reduced to a small set of dominant eigenworm modes.
Building the ECHO Training Harness (echo_trainer.py / echo_runner.py, previously specced but not yet built) is the means to run this comparison — and it's built dataset-agnostic from day one so it's not thrown away regardless of which way the comparison goes.
This is explicitly allowed to fail. If the Gaussian variant does just as well as the learned EBM on worm gait, that's a real and useful result: it tells us this particular signal is too simple to need multimodal machinery, and that the interesting test of the Waddington-landscape idea is Option A, where cells plausibly do occupy genuinely distinct discrete attractors. Either outcome is data we take into the next phase.
2. Separation of Concerns (Infra vs. Demo)
Per your standing preference, this doc keeps a hard line between:
Infra (reusable library code, dataset-agnostic where possible, imported by anything): dataset loader, EchoTrainer, EchoRunner, the PyTorch↔JAX bridge.
Demo/benchmark script (one file, thin): wires infra together for this benchmark specifically and produces the plot/report. It contains no training loop, no gradient code, no dataset parsing — only orchestration and output generation.
Note that 01_waddington_collapse.py currently blurs this line (the benchmark function is the physics wiring). This project is the chance to establish the clean split from scratch, rather than retrofitting it — Prompt 3 of the original ECHO doc can later point 01_waddington_collapse.py at the same EchoRunner once it exists.
This split is also what makes the Gaussian-vs-multimodal comparison (Section 5) cheap: the EBM is just one pluggable component of MarkovBlanketObserver. Swapping it doesn't touch the hull, the thermostat, Γ, Q, or the thermalizer at all — only which infra module build_graph instantiates.
3. Data
3.1 Source
Open Worm Movement Database (public, .mat/.hdf5 per-worm feature files). Each recording ships pre-computed:
worm.posture.eigenProjection — projection of body posture onto the first 6 dominant eigenworm shape modes (Stephens et al. 2008 basis). These 6 modes capture >95% of postural variance and the first 2 alone trace out the crawling limit cycle.
Per-recording metadata: strain (we use wild-type N2), age (days of adulthood), and experiment/plate ID.
This gives us, for free, exactly the kind of "sensory channel that is a biological oscillator" the ECHO doc's HD-MEA voltage channel played for the electrophysiology case — except low-dimensional and already dimensionality- reduced by domain experts, so no encoder is needed before the EBM.
3.2 New Infra: src/data/behavior/celegans_gait_dataset.py
class CElegansGaitDataset(Dataset):
    """
    PyTorch Dataset over Open Worm Movement Database eigenworm projections.

    __getitem__ returns a [seq_len, 6] float32 tensor of eigenworm
    coordinates for one worm recording, plus its age-in-days label.
    """
    def __init__(self, base_path=None, age_range=None, strain="N2", seq_len=1500):
        ...
    def __getitem__(self, idx) -> dict:
        return {"x_raw": eigenworm_seq, "age_days": age, "mask": mask}
Mirrors the existing PharmacologicalShockDataset / MullerBrownDataset pattern: resolve paths relative to PROJECT_ROOT (Ray-safe), open the file handle inside __getitem__ for multiprocessing safety, raise FileNotFoundError cleanly if the raw database hasn't been downloaded yet so the benchmark script can fall back gracefully (as 01_waddington_collapse.py already does).
A held_out_ages split helper groups recordings into young (day 1–3, peak movement) vs. old (day 9+, post-decline) cohorts — this is the axis the benchmark compares across, not a train/test split in the usual ML sense.
4. Architecture Mapping
Reuse MarkovBlanketObserver / PredictiveCodingGraph unchanged — no new physics primitives needed (Γ, Q, the hull, the thermostat are all identical to the electrophysiology case). Only the dimensions, the training target, and — per Section 5 — the EBM component itself change:
Component
Waddington Collapse (existing)
Worm Gait (this doc)
Sensory input
1024-ch HD-MEA voltage
6-dim eigenworm projection
d_internal_micro / d_active_micro / d_external_micro
64 / 64 / 64
8 / 8 / 8 (low-dim oscillator doesn't need a wide micro hull)
Macro observer
condensed 16/8/4/4 latent
condensed 8/4/2/2 latent
Training regime
none (weights hand-mutated in the old benchmark)
proper burn-in via EchoTrainer on young-cohort sequences only
Validation hook
Hessian trace vs. rolling voltage variance
Hessian trace, young-trained model rolled forward on held-out young vs. old sequences (no retraining on old data)

The key experimental design point: we train once, on healthy young-worm gait, and never fine-tune on old-worm data. The model is asked to explain old-worm sequences using the basin it learned from young ones. If the architecture is doing something real, forced_unroll should show the macro state either (a) failing to lock onto as clean a limit cycle, or (b) locking on but with a measurably shallower/flatter Hessian trace, when fed old-worm eigenworm sequences vs. young ones. That contrast is the aging signal — directly analogous to Route 4 ("Rejuvenation Hysteresis") in the README's philosophy, but here it's decline rather than rescue.
5. The Comparison: Laplace Assumption vs. Learned Multimodal EBM
This is the actual experiment. Everything else in this doc is scaffolding to make this comparison possible and legible.
5.1 The two variants
Both share the identical (energy, precision) interface consumed by MarkovBlanketObserver and HessianCurvatureTracker, so either can be constructed and dropped in without touching any other module.
PrecisionWeightedEBM (exists today) — E_θ(x) is an MLP. Can in principle represent a landscape with multiple basins, ridges, asymmetric wells — anything.
GaussianEBM (new, Prompt 1 below) — the Laplace assumption made literal: E(x) = ½(x − μ)ᵀ Π (x − μ) for a learned mean μ and a learned constant SPD matrix Π (same Cholesky-factorization trick as DissipativeFriction and the existing precision head: Π = LLᵀ + εI). Π does not depend on x — this is the whole point. It's a single basin, full stop.
5.2 The built-in sanity check
Because GaussianEBM's energy is exactly quadratic, its Hessian is Π everywhere — a constant matrix, independent of the state. So the GaussianEBM variant's Hessian-trace curve over any trajectory must come out as a flat line at trace(Π), whether the input is young-worm or old-worm data. This isn't a hoped-for empirical result, it's a mathematical identity — which makes it a free correctness test: if HessianCurvatureTracker ever shows a wiggling trace for the GaussianEBM variant, that's a bug in the harness, not a finding about worms. (Worth asserting directly in a unit test in Prompt 2, not just eyeballed on the final plot.)
5.3 The actual question
Everything interesting is on the PrecisionWeightedEBM side: does its Hessian trace vary across the trajectory, and does it separate young from old gait more than the flat baseline trivially "separates" them (it doesn't — trace(Π) is one number, same for both cohorts by construction)? Concretely, per cohort and per variant, EchoRunner.run() should report:


GaussianEBM
PrecisionWeightedEBM
Burn-in loss (young data)
expected: fits a single elliptical orbit fine — gait is close to periodic
expected: fits at least as well, possibly overfits without regularization
Hessian trace, young cohort
flat, = trace(Π)
? — the open question
Hessian trace, old cohort
flat, same value (Π doesn't see the input distribution shift)
? — does it drop, and does it separate from young?
Phase portrait
fixed ellipse (shape set by Γ, Q; E only sets where it's centered)
free-form — can look like anything the training data supports

If the PrecisionWeightedEBM trace is also roughly flat and doesn't separate the cohorts, the honest conclusion is: this signal doesn't need multimodal machinery, and the interesting test of the Waddington-basin idea is Option A, not gait. If it does separate them, that's a genuine (if small) positive result worth carrying forward. Both outcomes are useful — this section exists so we don't quietly skip running the boring baseline.
6. New Infra: The ECHO Harness Itself
This is the part carried over from the original ECHO design doc, built for real now, and deliberately not worm-specific — any future ECHO dataset (single-cell reprogramming included) should be able to reuse both files untouched.
A. src/echo/harness/echo_trainer.py — EchoTrainer
Pure JAX/Equinox, functional. Takes a PredictiveCodingGraph, an optax.GradientTransformation (e.g. optax.adamw), and exposes:
make_step(graph, opt_state, batch, dt) — an @eqx.filter_jit-wrapped, @eqx.filter_value_and_grad-wrapped update. Filters trainable EBM/Γ/Q weights from static hull topology/dims via eqx.filter. Applies optax.clip_by_global_norm before the optimizer step.
loss_fn(graph, batch, dt, key) — teacher-forced sensory MSE: feed batch.x_raw into forced_unroll's seq argument, take MSE between the sensory slice of the predicted next state and the true next frame.
fit(graph, opt_state, dataloader, epochs, dt, log_fn) — epoch loop, calls log_fn per epoch (kept generic so W&B is an injected callback, not a hard dependency of the trainer itself).
B. src/echo/harness/echo_runner.py — EchoRunner
Orchestration only — no gradient math lives here.
build_graph(config, key) — constructs micro/macro MarkovBlanketObserver instances and the PredictiveCodingGraph from a config dict (dimensions, ebm_hidden_size, temperature, dt). Dispatches on config["architecture"]["ebm_type"] ("gaussian" → GaussianEBM, "mlp" → PrecisionWeightedEBM) — this one branch is what makes Section 5's comparison a config change instead of a fork.
load_dataset(config) — dispatches on config["data"]["type"] (a new "celegans_gait" branch alongside the existing "pharmacological" one) and returns a PyTorch DataLoader.
run_burn_in(graph, dataset, config) — instantiates EchoTrainer, runs fit, returns the trained graph + loss history.
run_validation_hook(graph, eval_sequences, macro_ebm) — attaches HessianCurvatureTracker to graph.macro's EBM, runs forced_unroll per eval sequence, returns {cohort_label: hessian_trace_array}.
run(config) — the single public entrypoint: load → build → burn-in → validate → return a plain result dict. This is the only function the demo script calls.
C. src/echo/harness/pytorch_jax_bridge.py
Small, dependency-isolating helper: to_jax(tensor) -> jnp.Array and to_numpy_batch(dataloader_batch) -> dict[str, np.ndarray]. Enforces the Firewall Rule from the original doc in one place instead of scattering .numpy() calls through training code.
D. configs/echo_worm_gait.yaml
data:
  type: celegans_gait
  strain: N2
  seq_len: 1500
  young_age_days: [1, 3]
  old_age_days: [9, 13]
architecture:
  ebm_type: mlp  # "mlp" (PrecisionWeightedEBM) or "gaussian" (GaussianEBM) — see Section 5
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
7. Demo Script: src/echo/benchmarks/06_worm_gait_decline.py
Thin by construction — this is everything it's allowed to do:
Parse --config, load YAML (the config's ebm_type sets one variant; the script runs the script's main() twice, once per ebm_type, by overriding that one field — it does not know or care how either variant is built internally).
Call EchoRunner(config).run() twice (once per ebm_type) → get back, per variant: trained graph + per-cohort (young/old) Hessian trace arrays + loss history. No physics, no gradients, no EBM-swapping logic here — that all lives in build_graph.
Generate the output, laid out so the Section 5 table is visible at a glance:
Panel 1: macro-state phase portrait (first 2 macro dims), 2×2 — {gaussian, mlp} × {young, old} — from forced_unroll. This is the direct, literal picture of a fixed ellipse (gaussian) vs. a free-form orbit (mlp), and of the orbit tightening or degrading with age.
Panel 2: Hessian trace over time, both cohorts, both variants overlaid (4 lines) — the gaussian pair should visibly be two flat, overlapping lines per Section 5.2; the interesting line is whatever the mlp pair does.
Panel 3: burn-in loss curve, both variants (sanity check that both actually converged before the comparison is treated as meaningful).
Print the Section 5.3 table populated with real numbers: mean Hessian trace per {variant × cohort}, and whether the mlp variant's young/old gap is (a) non-trivial and (b) in the expected direction.
Save plot + a small JSON summary to output/echo/, matching the existing output/harness/ convention.
If the dataset isn't downloaded yet, catch FileNotFoundError exactly as 01_waddington_collapse.py does today and skip gracefully rather than crashing the smoke test.
8. Phased Implementation Plan (Micro-Prompts)
Prompt 1 — Data infra. CElegansGaitDataset + a download_openworm.sh or documented manual-download step (the Open Worm Movement Database isn't pip install-able) + a young/old cohort split helper. Include a synthetic fallback generator (sinusoid + phase-noise + amplitude decay, seeded) so the harness can be unit-tested without the real database present — same role MullerBrownDataset plays for the toy physics demos.
Prompt 2 — The Harness + the Gaussian baseline. echo_trainer.py, echo_runner.py, pytorch_jax_bridge.py, configs/echo_worm_gait.yaml, and GaussianEBM (Section 5.1) alongside the ebm_type dispatch in build_graph. Include the Section 5.2 flat-Hessian-trace identity as an actual unit test (construct a GaussianEBM, sample several random states, assert hessian_trace is constant across all of them within numerical tolerance) before wiring anything worm-specific — that test alone tells you if GaussianEBM is implemented correctly. Also retroactively pass the config-driven PharmacologicalShockDataset branch through EchoRunner.load_dataset, so the harness is proven dataset-agnostic on day one rather than only ever being exercised by worm data.
Prompt 3 — The Benchmark. 06_worm_gait_decline.py per Section 7, wired purely through two calls to EchoRunner.run() (one per ebm_type). No new physics, training, or EBM-construction code should appear in this file — if it does, that's a sign something belongs back in echo_runner.py instead.
9. Open Questions / Risks
Recording length variance. Open Worm Movement Database recordings vary in duration; seq_len truncation/padding policy needs to be decided in Prompt 1 (likely: truncate to the shortest usable window across the cohort rather than pad, to avoid teaching the model an artificial "freeze" at sequence end).
Confound: age vs. plate/batch effects. Locomotor decline correlates with age, but recordings are also batched by experiment day. Worth stratifying the young/old split across multiple plates/dates rather than pulling both cohorts from a single session, or the "aging" signal could partly just be a batch artifact.
A "no difference" result is ambiguous between two causes: (a) worm gait genuinely doesn't need multimodal structure, or (b) the mlp variant just didn't train well enough to find it (undersized ebm_hidden_size, too few epochs, bad learning rate). Before treating Section 5.3's table as a real answer, confirm the mlp burn-in loss actually converged (Panel 3) and consider a quick learning-rate/hidden-size sweep on the mlp variant only — a negative result is only informative if the positive result had a fair chance to show up.
Relatedly: GaussianEBM's flat trace is a fixed, near-zero-cost baseline regardless of hyperparameters — there's no equivalent "did it converge" question on that side, which is exactly why it belongs in the comparison as a floor rather than being skipped.
