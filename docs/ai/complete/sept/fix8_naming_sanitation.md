TASK: Remove overclaiming names and text from the Stable Basin repo, and add a naming rule to the
agent instruction file. This is a rename-and-rewrite task. Do not change any computation, default
parameter, or numeric output, except the two fake values listed in Step 3.

STEP 0: SETUP
- Create branch cleanup/names.
- Use git mv for every file rename so history is kept.
- Run pytest before editing and save the pass/fail counts.
- Scope: src/ (including src/demo and src/icebox), configs/, Makefile, README.md, ISSUES.md.
  Do not edit paper/ or tools/. List any violations you see there in the report.

STEP 1: ADD THE RULE
Add this section verbatim to the agent instruction file at the repo root (AGENTS.md, CLAUDE.md
or GEMINI.md, whichever exists; if none exists, create AGENTS.md):

    ## Naming and documentation rule

    Names, docstrings, comments, log messages, plot labels, file names, README and ISSUES
    describe what the code does today. They do not describe what we hope it will show.

    1. Name things by the operation: what goes in, what is computed, what comes out.
       Example: MetricThresholdMonitor, not RejuvenationFlightController.
    2. Do not use clinical or outcome words (rescue, therapy, therapeutic, dose,
       pharmacological, clinical, patient, rejuvenation, infusion, cure) for simulations or
       model parameters. Use them only for code that processes real data from that setting.
    3. Label data by where it came from, everywhere it appears (variable names, legends,
       output files). Synthetic or transformed data says so: "synthetic slowdown x3", not
       "old". Real cohorts state their definition: "<= 6 months", "adult day 18".
    4. No certainty or superlatives the code has not earned: exact, guaranteed, proves,
       ultimate, successfully detects, defeats. If a comment says "guarantees", a test
       checks it.
    5. Physics words (entropy, thermodynamic, basin, Markov blanket, homeostasis) are used
       only when the code computes or enforces that quantity. Variance is "variance", not
       "entropy".
    6. Mock, placeholder and estimated values are labeled as such where produced and where
       shown. Never hardcode a number that looks like a measurement.
    7. A result stated in README or docs names the script and output file that produced it,
       and whether it passed its pre-registered gate.
    8. Dropped or iceboxed code is marked at the top of the file and is not presented as
       current.
    9. When you add or rename anything, check it against this rule. If you notice a
       violation outside your current task, list it in your summary; do not fix it silently.

STEP 2: RENAMES
Update every import, reference, Makefile target, config file name, output file name, JSON
key, and wandb run/artifact name that depends on each rename.

| Old | New | What the code actually does |
|---|---|---|
| src/core/rejuvenation_controller.py, RejuvenationFlightController | src/core/threshold_monitor.py, MetricThresholdMonitor | Applies fixed thresholds to KSM/CSD/PLV and logs a status |
| _actuate_iv_pump | _log_decision | Writes a log line; controls no hardware |
| EMERGENCY_ABORT / MAINTAIN_INFUSION / STATE_BIFURCATION_DANGER | ALARM / OK / STATE_ALARM | A threshold was crossed |
| src/echo/clinic/ | src/echo/model_probes/ | Model-parameter edits and a gradient probe |
| DigitalTwinAnnealer.anneal_twin | ParameterScaler.scale_friction_and_precision | Multiplies friction and precision-head weights by constants |
| DigitalTwinInterrogator.ping_and_measure | GradientResponseProbe.measure | Adds a perturbation to the micro state, reports gradient norms |
| 06_infer_biological_lambda.py, 06_inferred_biological_lambda.json, key biological_lambda | 06_fit_lambda.py, 06_fitted_lambda.json, key fitted_lambda | Fits lambda to the eval data (synthetic for worm) |
| 07_intervention.py | 07_lambda_comparison.py | Compares simulations at two lambda values |
| 09_pharmacological_translation.py, 09_clinical_translation_metrics.json, 09_pharmacological_curve.png, keys Optimal_Lambda / Max_Rescue_R | 09_lambda_argmax.py, 09_lambda_argmax.json, 09_lambda_argmax.png, keys argmax_lambda / max_R | Argmax of R over a lambda grid |
| Makefile: aging-resilience-infer-lambda / -intervention / -pharmacology | aging-resilience-fit-lambda / -lambda-comparison / -lambda-argmax | Follow the scripts |
| src/harness/clinical_diagnostic_runner.py, configs/clinical_diagnostic.yaml, target clinical-diagnostic, outputs clinical_diagnostic_* | ksm_threshold_runner.py, configs/ksm_threshold.yaml, target ksm-threshold, outputs ksm_threshold_* | Finds the first frame where KSM drops below a threshold |
| ThermodynamicDiagnosticEngine.generate_diagnostic | AttributionSummary.summarize | Summarizes attribution magnitudes |
| src/models/vessel/active_inference_agent.py, ActiveInferenceAgent | chemotaxis_reaction_diffusion.py, ChemotaxisReactionDiffusion | Reaction-diffusion with chemotaxis; no free-energy term |
| src/data/sim2real/ | src/data/synthetic/ | Every module generates simulated data |
| ThermodynamicMetrics (src/metrics/time_domain.py) | TimeSeriesStabilityMetrics | Variance, AR1, DMD eigenvalues, LLE |
| extract_fedichev_macrostates; keys Z_entropic_damage / z0_volatility / epsilon_0_ksm | extract_path_metrics; keys cumulative_path_divergence / csd / ksm | Names the computation |
| TelemetryLogger: log_fedichev_macrostates; rerun paths consciousness_manifold/*, early_warning_radar/*, fedichev_macrostates/* | log_path_metrics; paths latent/points_3d, metrics/* | Paths name the logged array |

STEP 3: DOCSTRINGS, COMMENTS, LABELS (no behavior change except the two fake values)
- MarkovHull docstring: replace "Mathematically enforces a Markov Blanket" and "Enforces the
  fundamental law..." with: "Partitions the state into internal/sensory/active/external index
  ranges and provides a mask that zeroes internal-external entries of Q and Gamma. This does
  not make internal and external states conditionally independent: the energy is a dense MLP
  over the full state, so the drift of internal states still depends on external states."
- EchoTrainer docstring: "Trains by minimizing one-step-ahead MSE on the sensory dims under
  teacher forcing, backpropagating through the SDE unroll."
- PrecisionWeightedEBM / IdentityPrecisionEBM: replace "guarantee the landscape is a
  positive-definite basin" with: "Adds 0.0005*||x||^2 so energy grows at large |x|. This does
  not make the Hessian positive-definite; the MLP term can make it indefinite."
- celegans_gait_dataset.py: inject_synthetic_degradation docstring and inline comment ->
  "Slows gait-amplitude relaxation 3x via synthetic_aging.slow_amplitude_relaxation. Synthetic
  positive control, not aged worms." SyntheticWormMockDataset docstring -> "Noisy
  Stuart-Landau oscillator in channels 0-1, Gaussian noise in channels 2-5."
- ForcedTorxThermalizer controller comment: remove "homeostasis" and "drag ... to safety";
  replace with "q_ext = -q_gain * state * mask: proportional pull toward zero on masked dims."
- MeldLoss docstring: remove the glucose-perfusion and Landauer text. Describe the math:
  forecast MSE, penalty relu(||delta_y||_2 - L*delta_x), reconstruction MSE.
- mamba_lrp.py class docstring: replace "Mathematically exact" with "Approximate LRP-epsilon:
  routes relevance back through time with a fixed retention factor of 0.98 instead of the
  model's state transition, so relevance is not exactly conserved."
- time_domain.py: remove "successfully detects the Waddington bifurcation point" and "directly
  quantify the thermodynamic stability". CSD docstring -> "alpha * rolling variance + beta *
  lag-1 autocorrelation over a sliding window."
- diagnostic_engine.py (FAKE VALUES, remove):
  (a) confidence_score default 0.98 is not computed; make it a required argument with no
      default, or drop the field.
  (b) the "mechanism" field maps feature-name prefixes Psi/Omega/Sigma that never match the
      real feature names, so it always returns "Unknown anomaly"; remove the field.
  Change status "CRITICAL_FAILURE_PREDICTED" to "attribution_summary".
- PharmacologicalShockDataset and HDMEADataset docstrings: remove "MVM Proof", "ultimate",
  "proves", "defeating". State the data source, format and array shape.
- hardware_monitor.py: where the CPU path returns formula-based VRAM numbers, log
  "estimated, not measured" and say so in the docstring.
- Plots and logs in the aging_resilience scripts and core.py:
  - "Therapeutic Rescue R(lambda)" -> "R(lambda) = 1 - D(young, lambda) / D(young, lambda_A)"
  - "Clinical Dose-Response Sweep" -> "R(lambda) across lambda (inverse-temperature scaling)"
  - "Optimal Dose" -> "argmax lambda"
  - "Degraded Pathology" / "Therapeutic Rescue" panel titles -> "lambda = {value}"
  - delete "Figure 4 Beacon Plot"; replace "precision_injection_gain" in logs with "lambda"
  - x0 labels and the setup_experiment log "pathological initial state" -> "x0: random normal
    (scale 2) with sensory dims set from one eval-old frame"
- Cohort labels: add a cohort_labels property to AgingBenchmarkTask returning
  (label_a, label_b): worm ("clean", "synthetic slowdown x3"); killifish ("<= 50% of
  lifespan", "> 50% of lifespan"); catnap ("<= 6 months", ">= 24 months"). Use it for every
  legend and log line in 01, 02, 03, 05, 07, 10 that currently says young/old.
- 10_animate / animation.py: the body shape uses a synthetic Fourier basis. Add "Body shape
  drawn from a synthetic basis, not the measured eigenworm basis" to the figure. Replace
  "Attractor" and "(Limit Cycle)" titles with "Eigenworm channels 0-2".
- VesselBaseline plot: "Internal Variance (Entropy)" -> "Internal variance".
- Remove stale "suggested location" and "replaces ..." header lines left from earlier handoffs
  (synthetic_aging.py, 11_null_control.py, trace_evaluator.py).

STEP 4: README AND ISSUES
README.md, rewrite to:
- Summary: "Stable Basin is research code. It fits energy-based stochastic dynamical models
  (JAX/Equinox, built on Torx) to physiological time series and computes stability metrics:
  Hessian trace of the learned energy, critical-slowing-down indicators, entropy-production
  estimators, DMD eigenvalues."
- Status: what runs, what passed a gate (name the script and output file), what has not been
  tested. State plainly: no result yet on real aged data; worm-gait results use a synthetic
  slowdown as a positive control; killifish and CATNAP loaders exist and their cohort
  construction is being fixed.
- A table: directory -> what it does.
- "Earlier work, not current": SSM/transformer sensor-fusion experiments on synthetic data;
  Mamba-2 and MambaLRP are iceboxed.
- Keep a "Route" only if a script implements it, and name that script. Move the rest to
  "Ideas, not implemented".
- Remove "Defeating Entropy", "flight computer", "brutal", "ultimate", "If you want to solve
  aging", the MaskAwareMamba reference-architecture section, and "O(1) VRAM" unless a
  benchmark in the repo measures it.
- If OPEN_PROBLEMS.md does not exist, point that link to ISSUES.md.
- Citation title -> "Stable Basin: energy-based dynamical models and stability metrics for
  physiological time series (research code)".
ISSUES.md:
- Rejuvenation Gymnasium: remove "reward function is the Trace of the Hessian", "Bio-Blade
  hardware" and "first AI-driven longevity controller". Add: "Do not use the Hessian trace of
  the model energy as a reward. Scaling the energy by lambda raises the trace by
  construction, so an agent can maximize it with no change in the data. A reward must be
  computed from measured data against a held-out reference." Remove the 11_ratchet_simulator
  reference if that file does not exist.
- Quintet tensor: replace "prove that the engine can maintain a coherent biological limit
  cycle" with a plain statement of the engineering task.
- Flyte/Ray tickets: update stale script names.

STEP 5: SWEEP
Run:
rg -n -i -e rescue -e therap -e "\bdose" -e pharmacolog -e clinical -e patient -e rejuvenat \
  -e infusion -e "iv pump" -e "\bcure" -e flight -e ultimate -e "\bprove" -e "\bproof" \
  -e "\bexact" -e guarantee -e defeat -e conscious -e homeostasis -e patholog -e entropy \
  -e thermodynamic -e fedichev -e waddington -e "biological lambda" -e successfully \
  -e sim2real -e "MVM" -e "bio-blade" src configs Makefile README.md ISSUES.md
For each hit: fix it if it breaks the rule; otherwise leave it and record a one-line reason
in the report (for example, "entropy" in entropy_production.py is an entropy estimator).

STEP 6: REPORT AND GATES
Write docs/naming_cleanup_report.md containing:
- a table of old name -> new name, file, one-line reason;
- sweep hits left in place, each with a reason;
- violations seen in paper/ and tools/ (not edited).
Gates:
- G1: pytest pass count equals the Step 0 count (update tests only for renamed names).
- G2: make -n succeeds for every Makefile target.
- G3: every renamed module imports: python -c "import <module>" for each.
- G4: the Step 5 sweep returns no hits except those listed in the report as kept.
- G5: in the report, list every diff hunk that changes a non-string, non-comment line other
  than a rename or the two fake values. The expected list is empty.
Stop after the report. Do not push.
