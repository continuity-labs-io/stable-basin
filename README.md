# Stable Basin

**Open benchmarks for one question: does function measured repeatedly over time
predict how long an animal will live, better than a single snapshot of the same
animal?**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Status as of September 30, 2026: early research code. The lead benchmark
(killifish lifespan) is under construction and has no result yet. See
[What has been tested](#what-has-been-tested) for what holds and what does not.

---

## The question

Aging clocks and most omics biomarkers measure an organism's state at one moment.
Testing an aging intervention against survival means waiting for animals to die,
which takes years in mice.

Stable Basin tests a different kind of readout: how an organism behaves and
recovers over time, recorded without harming it. If such recordings predict
remaining lifespan in animals no model has seen, they become a candidate early
readout (a surrogate endpoint) for intervention studies.

**Working hypothesis.** An organism keeps its essential variables within livable
ranges against constant disturbance. Aging is the gradual loss of that capacity:
corrections get slower, targets drift, and the livable range narrows. If that is
right, loss of capacity should show up in time-series features such as recovery
time, variance and autocorrelation, before it shows up as death.

The benchmark tests this hypothesis rather than assuming it. Two cautions:

- Passive recordings may show loss of capacity late. Measuring recovery after a
  disturbance may be needed, not just watching fluctuations.
- The direction of change can differ by system. Some human brain-imaging studies
  find that aging shortens autocorrelation rather than lengthening it. The
  benchmark reports the direction it finds.

---

## Roadmap: model organisms toward mouse

Each stage asks the same question in a longer-lived organism. Each stage has a
pass/fail gate written before the run.

| Stage | Organism | Data | Question | Status |
|---|---|---|---|---|
| 1. Methods check | *C. elegans* | EigenWorms posture recordings (real gait) with a known slowdown injected | Do the features detect a known slowdown, and ignore added measurement noise? | Planned |
| 1b. Real aging | *C. elegans* | Open Worm Movement Database, ERIBA aging series (public; by day of adulthood, no lifespan labels) | Do the features change with age in real aging worms? | Planned |
| 2. **Lead benchmark** | Killifish | [Bedbrook et al., *Science* 2026](https://zenodo.org/records/17238217): 20 fps pose tracking from puberty to death, individual death dates, CC BY 4.0 | Do behavior dynamics before a landmark age predict remaining life in held-out fish, beyond static summaries? | **In progress** |
| 3. Mouse, observational | Mouse | JAX video frailty data; Calico home-cage physiology with survival (by request) | Does the signal hold in a mammal? | Not started |
| 4. Mouse, intervention | Mouse | LEV Foundation RMR2 smart-cage monitoring (partnership needed) | Does the readout track a rejuvenation intervention months before survival data? | Not started |

The end goal is stage 4: a readout that tells a mouse intervention study, within
months, whether a treatment is changing the course of decline.

---

## Lead benchmark: killifish lifespan

Code: [`src/benchmarks/lifespan/`](src/benchmarks/lifespan/)

**Task.** Pick a landmark age L (70 or 100 days). Using only recordings made
before L, predict each fish's remaining life after L. Fish still alive at the end
of the study are kept as censored, never dropped.

**Features**

| Set | What it contains |
|---|---|
| F1, static | Per-session mean of each pose feature; per fish, the mean and SD across sessions |
| F2, dynamic | Per session, after linear detrending: lag-1 autocorrelation, variance, integrated autocorrelation time. Per fish: the mean across sessions and the slope against age |
| F1 + F2 | Both |

**Evaluation.** Penalized Cox model with standardization and PCA fitted inside
each training fold. 5-fold cross-validation split by fish, repeated 20 times.
Score: concordance index (C-index). Null: the same pipeline on shuffled outcomes.

**Gates**

| Gate | Pass condition |
|---|---|
| G0, data | At least 40 fish alive at L, each with at least 3 sessions before L |
| G1, signal | F1 + F2 C-index, lower 2.5% bound above 0.5 |
| G2, dynamics add information | C(F1 + F2) − C(F1) has a mean of at least 0.03 and a lower 2.5% bound above 0 |
| G3, null | Shuffled-outcome C-index between 0.45 and 0.55 |

**Current state.** Fully implemented. (Gate 0 passes when data is present).

**Outputs** go to `output/benchmarks/lifespan/` and one row per feature set is
appended to `output/lifespan/lifespan_ledger.csv`.

---

## Tracks

From [`tracks.md`](tracks.md). Every track serves the lifespan benchmark or is
parked.

| Track | Question it answers | Status |
|---|---|---|
| 1. Lifespan benchmark | Do time-series features predict remaining lifespan beyond static summaries, in held-out animals? | Active, lead |
| 2. Model-free resilience metrics | Recovery time, lag-1 autocorrelation, variance, entropy production from raw data | Active |
| 3. Echo model (energy-based SDE) | Does a fitted dynamical model give better predictors or interpretable resilience measures? | Parked |
| 4. Validation code | Null controls, positive controls, cohort splits, leakage checks | Active, always |
| 5. MEA tissue QC | Spike sorting, drift, longitudinal comparability on MaxWell HD-MEA | Active, separate business decision |
| 6. Sequence models | SSMs, Mamba, transformers, sensor fusion | Parked |
| 7. Thermodynamic hardware | Torx / Extropic compatibility | Parked |
| 8. Control and interventions | Controllers, a Gymnasium environment | Parked; needs a validated readout first |

---

## What has been tested

| Test | Result | What it means |
|---|---|---|
| Synthetic check of the slowdown features (`synthetic_aging.py`) | Slowing the restoring dynamics raised variance and lag-1 autocorrelation together. Adding measurement noise raised variance but lowered autocorrelation. | The features can tell genuine slowing from added noise. |
| Raw eigenworm autocorrelation | Lag-1 autocorrelation stays near 0.99 whatever the injected slowdown. | Raw channels measure the waveform, not resilience. Measure on amplitude residuals instead. |
| Echo structure checks (`12_structure_checks.py`, 20 held-out worms) | Boundary test failed (r ≈ 0.23; gate < 0.01). Timescale-separation test failed (ratio 1.17, 95% CI 0.99–1.35; gate: lower bound > 2). Third test void: the gate was written incorrectly. | The fitted Echo model does not show the boundary (Markov blanket) or two-level hierarchy it was designed with. It should not be described as having either until a test passes. |

**Withdrawn draft.** The draft paper in `paper/sharpening_the_tack/` (built by
`make paper` and `make reproduce-paper`) is withdrawn pending a rewrite. Its
"aged" cohort was synthetic noise added to young worm data, the model comparison
was decided by construction, the rescue measure was circular, and each cohort was
one worm. Its effect sizes do not measure aging and should not be cited.

---

## Quickstart

### Setup

```bash
git clone https://github.com/continuity-labs-io/stable-basin.git
cd stable-basin
# Install dependencies: TODO confirm the install command for this repo.
make preflight        # lint and run the test suite
```

### Data

| Dataset | Where to put it | Source |
|---|---|---|
| Killifish lifelong behavior | `data/killifish/` | [Zenodo 10.5281/zenodo.17238217](https://zenodo.org/records/17238217) (about 14.5 GB) |
| EigenWorms | `data/worm/EigenWorms_TRAIN.ts`, `data/worm/EigenWorms_TEST.ts` | UEA/UCR time series classification archive |
| Calico CATNAP features | `data/catnap/trace_features.h5` | By request |

The worm pipeline raises an error when real data is missing. Synthetic data is
used only when you ask for it with an explicit flag (for example `--mock` or
`--use_synthetic`) or `make run-synthetic-aging`.

### Run

```bash
# Lead benchmark: killifish lifespan
make lifespan

# Worm pipeline, null and positive control for the Echo curvature measure
python -m src.benchmarks.aging_resilience.11_null_control

# Echo structure checks (boundary, timescale separation, macro closure)
python -m src.benchmarks.aging_resilience.12_structure_checks

# Full numbered worm pipeline (scripts 01-11)
make run-worm-gait
make run-synthetic-aging    # synthetic data only, no W&B
```

Individual pipeline steps: `make aging-resilience-<step> DATASET=<config>`, where
`<config>` names a file in `configs/`. See the Makefile for the step list.

---

## Repository map

| Path | What it holds | Track |
|---|---|---|
| `src/benchmarks/lifespan/` | Lifespan benchmark | 1 |
| `src/benchmarks/aging_resilience/` | Numbered worm-gait pipeline: baselines (01–03), Echo tuning and training (04–05), λ fitting and comparison (06–09), animation (10), null control (11), structure checks (12). Dataset adapters in `tasks/`. Its "old" cohort is real gait with a known slowdown injected, not aged worms. | 2, 3, 4 |
| `src/data/behavior/` | Worm, killifish and Calico CATNAP loaders; `synthetic_aging.py`, degradations with known ground truth for testing detectors | 1, 2, 4 |
| `src/data/ephys/` | MaxWell, HD-MEA, spike and LFP loaders | 5 |
| `src/data/synthetic/` | Simulated multimodal data for plumbing tests. Not biological measurements. | 6 |
| `src/echo/` | Echo model: architecture, energy functions, training harness, Hessian curvature (`metrics/energy_landscape.py`) | 3 |
| `src/metrics/` | Time-series stability metrics (variance, autocorrelation, Koopman/DMD, Lyapunov estimate), spectral metrics, entropy production estimators, permutation and bootstrap statistics | 2, 4 |
| `src/models/` | Sequence models: SSMs, transformer, GRU-D, ODE-RNN | 6 |
| `src/harness/` | Training and sweep runners for the sequence-model suite | 6 |
| `src/core/` | Device setup; `MetricThresholdMonitor`, which raises an alarm when metrics cross thresholds | 6, 8 |
| `tracks.md`, `ISSUES.md` | Track definitions; engineering tickets and parked ideas | — |
| `AGENTS.md` | Naming conventions and working rules for contributors and coding agents | — |

### Parked: sequence-model suite (track 6)

The original sensor-fusion experiments remain runnable but are not part of the
lifespan work: `make quickstart`, `make ssm-experiments`, `make ksm-threshold`.
They run on simulated data.

---

## Rules for results

- Every experiment is a pass/fail gate, written down before the run.
- Train/test splits are by animal, never by time window within an animal.
- A null control (shuffled outcomes or no effect) must score near chance, and a
  positive control with known ground truth must pass, before any claim is made.
- Trained models: at least 10 seeds, and convergence verified for every model
  being compared.
- Report effect sizes with intervals, not p-values alone.
- Names describe what the code does today. Simulations are not called clinical,
  data is labeled by origin, and physics terms are used only for quantities that
  are actually computed. See `AGENTS.md`.

---

## Contributing

Open work, in priority order:

1. **Finish the killifish benchmark** (features, evaluation, figures, tests).
2. **Worm positive control:** detect a known slowdown injected into real gait, with
   the false-positive rate on clean data reported.
3. **Recovery after disturbance:** the killifish are fed seven times a day.
   Measure how behavior returns to baseline after each feeding.
4. **Data with hidden outcomes:** lifespan cohorts not yet public (killifish, worm
   WorMotel or Lifespan Machine, mouse). A fair test needs outcomes no modeler has
   seen.

Open an issue before starting a large change.

### Toward an open challenge

The killifish benchmark is the seed of a proposed open challenge: predict
remaining lifespan from function measured over time, scored on hidden test sets,
with tracks for killifish, mouse and a live intervention cohort. Existing aging
biomarker challenges score single samples. If you hold a lifespan cohort, run a
challenge series or fund open datasets, please get in touch.

---

## Citation

```bibtex
@misc{stable_basin_2026,
  title        = {Stable Basin: Open Benchmarks for Predicting Lifespan from Function Measured over Time},
  author       = {McCall, Ryan J. and {Continuity Labs}},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/continuity-labs-io/stable-basin}}
}
```

If you use the killifish data, also cite Bedbrook et al., *Science* (2026).

## License

MIT.
