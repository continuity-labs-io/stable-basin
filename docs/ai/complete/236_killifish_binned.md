Rewrite src/benchmarks/lifespan/killifish_benchmark.py to use the 10-minute binned lifespan files. Delete all .h5 / 20 Hz code and the --kinematics-dir argument. Keep run_evaluation_pipeline, the gates and the outputs, with the changes below.

STEP 0: DATA AND AUDIT
- Add --source {auto,kinematic,syllable}, default auto.
  kinematic: data/killifish/data/b0_20250415/df_reformat_b0_10_20250416.csv
  syllable:  data/killifish/data/a1_20241119/26441580/df_reformat_10_20241119_join_edit.csv
             (fall back to df_reformat_10_20241119.csv if that name exists instead)
  auto = kinematic if the file exists, else syllable. Record the source in all outputs.
- Print the column list and 3 rows before anything else. Read only needed columns,
  in chunks, and cache once to output/benchmarks/lifespan/killifish_10min_<source>.parquet.
- Feature columns: kinematic = snout_velocity, snout_acceleration, disp, body_length,
  count_snout, active, inactive, sleep (if the file is long-format with a 'feature'
  column, pivot it); syllable = state_0..state_99.
- FORBIDDEN as features: prognosis, prognosis_fraction, lifespan, status, hatch_date,
  full_fish_name, fish_number, cohort, table, sex, feeding, genotype. Assert that no
  forbidden column is in the feature matrix; raise if one is.
- Fish key = full_fish_name. Print whether fish_number maps 1:1 to full_fish_name.
- Age = age_days. Ignore any age parsed from file or folder names.
- Primary cohort: genotype 'wt', sex 'male', feeding 'al'. Print fish counts for every
  sex x feeding group.
- Outcome: T = lifespan - L; E = 1 if status == 'd', else 0 ('j', 'r', 's' censored).
  Print counts by status.
- A day is valid if at least 120 of its 144 bins are present and non-negative.
- Gate 0: for L in {70, 100}, count fish with lifespan >= L and >= 14 valid days with
  age_days < L. Pass if >= 40; choose the L with more fish (tie -> 70). If both fail,
  report L in {50, 60} and stop. Write killifish_audit.json.

STEP 1: FEATURES (rows with age_days < L only)
- F1 static: per feature, the daily mean; per fish, mean and SD across valid days.
- F2 dynamic, per feature:
  a) Across days: daily-mean series, linear detrend, then variance, lag-1
     autocorrelation, and integrated autocorrelation time (sum ACF to first zero
     crossing, max lag 30 days).
  b) Within day: subtract the fish's mean 24 h profile (from its own pre-L days),
     then lag-1 autocorrelation of the 10-min residuals per day; per fish, the mean
     across days and the slope versus age.
- Print feature counts for F1, F2, F1+F2.

STEP 2: EVALUATION (as now, plus)
- If a Cox fit fails, do NOT fill with random numbers. Count failures per feature set,
  report them, and mark that repeat invalid.
- Secondary, reported but not gated: leave-one-cohort-out C-index for F1 and F1+F2.

STEP 3: OUTPUTS
- Same JSON, CSV rows and two figures, with 'source' added everywhere.

TESTS (tests/benchmarks/test_killifish_benchmark.py)
- Synthetic 10-min data: 60 fish, 100 days, 144 bins/day, lifespan driven by the
  across-day lag-1 autocorrelation of one feature -> G1 passes.
- Same data with shuffled outcomes -> C-index in [0.45, 0.55].
- Feature matrix that includes 'prognosis' -> raises.

Run with --source auto. Paste killifish_audit.json, killifish_results.json and both figures.
