Refactor the lifespan work into one benchmark that runs a suite of datasets. Use only pandas, numpy,
scipy, scikit-learn and lifelines in this path: no torch, JAX, W&B or Ray. Do not touch
src/benchmarks/echo_resilience/.

LAYOUT
src/benchmarks/lifespan/{__init__,contract,adapters,landmark,evaluate,run}.py
src/features/{__init__,static,dynamics}.py
configs/lifespan/suite.yaml
tests/lifespan/

CONTRACT (contract.py)
- animals: one row per animal. animal_id (str, unique), lifespan_days (float), died (0/1, 1 = natural
  death), group (str), plus optional metadata columns.
- series: one row per animal x time bin. animal_id, age_days (int), bin (int, position within the
  day), then the measured variables (float).
- validate(animals, series, variables) raises with a clear message if: animal_id is not unique; a
  series animal_id is missing from animals; a variable is missing or non-numeric; any of these is
  listed as a variable: lifespan_days, died, status, prognosis, prognosis_fraction, group, cohort,
  sex, feeding, genotype, table, hatch_date; or age_days exceeds lifespan_days + 1.

ADAPTERS (adapters.py; a plain dict ADAPTERS = {name: function(cfg) -> (animals, series, variables)})
- killifish_bedbrook: port loading and filtering from
  src/benchmarks/model_free_resilience/killifish_benchmark.py. animal_id = full_fish_name;
  group = cohort; died = (status == 'd'); bin = minute_of_day // 10. Apply cfg['filters'].
  Option negative_is_missing (default true): negative variable values become NaN. A bin is valid if
  all variables are present. Keep animals in order of first appearance in the source file (as the
  old script does) so cross-validation folds match. Cache to data/cache/killifish_bedbrook.parquet.
- synthetic_positive: 80 animals, 120 days, 144 bins/day, 4 variables, fixed seed. Each animal has a
  relaxation time tau_i; each variable is an Ornstein-Uhlenbeck process with time constant tau_i plus
  a shared 24 h profile. All lifespans exceed 60 days, and remaining life after day 60 is shorter for
  longer tau_i. Static means are the same across animals. Censor about 30% at random. Tune the effect
  so dynamic features give C about 0.75. Reuse src/data/behavior/synthetic_aging.py where it fits.
- synthetic_after_landmark_only: same generator, but tau changes only in the last 20 days before
  death, and every lifespan is at least 80 days. Data before day 60 carry no information about
  lifespan.

LANDMARK (landmark.py)
- cut_at_landmark(animals, series, variables, L, min_valid_days, min_valid_bins=120)
  -> (pre_series, outcomes). Keep animals with lifespan_days >= L and at least min_valid_days valid
  days with age_days < L. pre_series = valid days with age_days < L only;
  assert pre_series.age_days.max() < L. outcomes = DataFrame(animal_id, T = lifespan_days - L,
  E = died, group).
- This is the only function that touches both animals and series.

FEATURES (src/features/)
- Port F1, F2a and F2b from the old script with identical formulas:
  static(rows, variables) -> pd.Series; dynamics(rows, variables) -> pd.Series.
  rows = one animal's pre_series. The 24 h profile for within-day residuals comes from the same rows.
  Feature functions never receive animals or outcomes.

EVALUATE (evaluate.py)
- Port run_evaluation_pipeline and run_loco_pipeline unchanged: seed 42, folds random_state =
  seed + repeat, the same folds for every feature set, CoxPH penalizer 0.1, PCA up to 10 components
  inside folds.

RUN (run.py; CLI: --suite PATH [--only NAME])
- Controls always run first, even with --only.
- For each dataset and landmark: adapter -> validate -> cut_at_landmark -> features (static,
  dynamics, both) -> evaluate -> gates G0-G3 as before.
- Write output/lifespan/<dataset>/L<landmark>/<run_id>/ where run_id = YYYYMMDD-HHMM-<short git
  hash>, containing: config.yaml (resolved), git_commit.txt, audit.json (animals and events by group
  and by died, valid-bin rule, negative-value counts, variables), features.parquet, results.json,
  cindex.png, km.png.
- Append one row per dataset x landmark x feature set to results/lifespan_ledger.csv: run_id, date,
  git_commit, dataset, role, landmark, feature_set, n_animals, n_events, c_mean, c_lo, c_hi,
  G1, G2, G3, valid.
- Controls: synthetic_positive must reach dynamics C >= 0.65; synthetic_after_landmark_only must
  have "both" C within [0.4, 0.6]. If either fails, write valid = false on every row in this run and
  exit non-zero.

SUITE (configs/lifespan/suite.yaml)
cv: {folds: 5, repeats: 20, seed: 42}
model: {penalizer: 0.1, pca_components: 10}
datasets:
  - {name: synthetic_positive, role: positive_control, landmarks: [60], min_valid_days: 14}
  - {name: synthetic_after_landmark_only, role: negative_control, landmarks: [60], min_valid_days: 14}
  - name: killifish_bedbrook
    role: data
    landmarks: [70, 100]
    min_valid_days: 14
    path: data/killifish/data/b0_20250415/df_reformat_b0_10_20250416.csv
    filters: {genotype: wt, sex: male, feeding: al}

TESTS (tests/lifespan/)
- test_landmark_invariance: on synthetic data, compute features through cut_at_landmark and the
  feature functions; overwrite every row with age_days >= L with random values and recompute; then
  drop those rows and recompute. Features must be identical (np.array_equal) in all three.
- test_contract: validate() raises on a forbidden variable and on a duplicate animal_id.
- test_controls: the two control thresholds above.
- test_port: killifish_bedbrook at L=70 with negative_is_missing=false reproduces the F1, F2 and
  F1+F2 mean C in output/benchmarks/model_free_resilience/killifish_results_precut.json to within
  0.01. Skip if the data file is absent.

MAKEFILE
- lifespan: python -m src.benchmarks.lifespan.run --suite configs/lifespan/suite.yaml $(if $(ONLY),--only $(ONLY))
  (use ONLY, not DATASET, which the echo targets already use)
- lifespan-test: pytest tests/lifespan -v
- Move the echo-resilience targets under a "# Parked: track 3 (Echo)" heading.
- After test_port passes: delete the model-free-resilience-killifish target and
  src/benchmarks/model_free_resilience/.

DOCS
- tracks.md: set track 3's status to "Parked".
- README.md: update the code paths, the run command (make lifespan) and the repository map.

GATE A: make lifespan-test passes, including test_port; make lifespan exits 0.
Paste the ledger rows from this run and audit.json for killifish_bedbrook at L=70.
