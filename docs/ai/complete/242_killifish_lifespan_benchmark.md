Context: Stable Basin repo. Build the first entry of a lifespan-prediction benchmark on the
public killifish data (Bedbrook et al., Science 2026; Zenodo 10.5281/zenodo.17238217).
Question: do dynamic features of behavior before a landmark age predict remaining lifespan
better than static summaries of the same behavior, in held-out fish? New module:
src/benchmarks/lifespan/. Do not change existing model code. Seed everything from one
--seed argument.

STEP 0: DATA AUDIT (gate before any modeling)
- Using the metadata CSV and kinematics files that KillifishContinuousDataset reads, load ALL
  fish, not only status == 'd'. Treat natural death ('d') as an event; treat any other status
  as censored at its recorded lifespan. Report the status values found and their counts.
- Per fish: number of recording sessions, age in days of each session, lifespan, event flag.
  Check how age is parsed from directory names and print 5 examples next to their paths.
- Write output/benchmarks/lifespan/killifish_audit.json.
- Gate 0: for landmark ages L = 70 and L = 100 days, count fish alive at L with >= 3 sessions
  before L. PASS if >= 40 fish at a landmark. If the loaded files are only a subset of the
  Zenodo record and fail this gate, stop and report which files are missing.

STEP 1: FEATURES (use only sessions with age < L; no information from after L)
- F1 static: per fish, mean and SD across sessions of each kinematic feature's session mean.
- F2 dynamic: per session, compute each feature's lag-1 autocorrelation, variance, and
  integrated autocorrelation time (sum of ACF to its first zero crossing) after linear
  detrending. Per fish, take the mean across sessions and the slope versus age across
  sessions.
- F1+F2 combined.
- Reduce each feature set with PCA fit inside each training fold only (10 components).

STEP 2: MODEL AND EVALUATION
- Same model for every feature set: penalized Cox proportional hazards (lifelines
  CoxPHFitter, penalizer 0.1). Target: remaining lifespan (lifespan - L) with event flag.
- Repeated 5-fold CV split by fish, 20 repeats, identical folds for every feature set.
  Metric: Harrell's C-index on held-out fish.
- Report per feature set: mean C and 95% CI over repeats; paired delta C versus F1 with a
  bootstrap 95% CI over fish.
- Null check: permute lifespans across fish and rerun F1+F2. Report its C-index.
- Write output/benchmarks/lifespan/killifish_results.json. Append rows to
  results/lifespan_benchmark.csv with columns: dataset, landmark_days, feature_set, n_fish,
  n_events, c_index, c_ci_low, c_ci_high, delta_vs_static, delta_ci_low, delta_ci_high,
  script, date.

GATES (fixed before running), reported PASS/FAIL at each landmark
- G1: behavior predicts at all: F1 or F2 has a C-index lower CI bound > 0.5.
- G2: dynamics add information: F1+F2 minus F1 has delta C >= 0.03 with lower CI bound > 0.
- G3: null check C-index between 0.45 and 0.55.
No p-values. Stop after the report.
