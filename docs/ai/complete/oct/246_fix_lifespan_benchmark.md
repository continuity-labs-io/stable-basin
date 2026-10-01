Fix the lifespan controls so the Step A control gate can fail. Touch only src/benchmarks/lifespan/,
tests/lifespan/ and configs/lifespan/suite.yaml.

1. REPRODUCIBLE SYNTHETIC DATA (adapters.py)
- ou_process uses an unseeded np.random.default_rng() when no rng is passed, so the synthetic
  controls change on every run. In _generate_synthetic: rng = np.random.default_rng(cfg.get('seed', 42));
  use rng for every random draw (replace np.random.uniform / normal / rand); pass rng=rng to every
  ou_process call; remove np.random.seed.

2. STRONGER NEGATIVE CONTROL (replace synthetic_after_landmark_only)
- L = cfg.get('landmark', 60); n = cfg.get('n_animals', 100).
- lifespan = rng.uniform(L + 20, L + 80, n).round(); died = rng.random(n) < 0.7; group 'synth'.
- 4 variables per animal, 144 bins/day: ou_process(n_bins, 1.0, 2 * 144, 1.0, rng=rng)
  + sin(2*pi*bin/144) + v*10. Same tau and noise for every animal.
- For rows with age_days >= L, add 3.0 * sqrt(0.5 * 2 * 144) * z_i to every variable, where z_i is
  the animal's lifespan standardized across animals.
- Docstring: "Data before the landmark carry no information about lifespan. A score above chance
  means post-landmark data reached the features."

3. ONE SOURCE FOR THRESHOLDS (new src/benchmarks/lifespan/controls.py)
- POSITIVE_MIN_C = 0.65   (positive control, dynamic features)
- NEGATIVE_BAND = (0.40, 0.60)   (negative control, each of F1, F2, F1+F2)
- LEAK_MIN_C = 0.75   (negative control, F1+F2, under the leak mutation)
- check_controls(results) -> list of failure messages. run.py and all tests import from here.

4. FAIL CLOSED (run.py, contract.py)
- Move per-landmark scoring into score_landmark(animals, series, variables, L, cfg,
  cut_fn=cut_at_landmark) -> per feature set: C array and Cox failure count. main() calls it.
- The run is invalid (valid = false on every ledger row, non-zero exit) if: the suite does not have
  exactly one positive_control and one negative_control; a control has zero animals after the cut;
  a control's C array is empty or any of its repeats failed; or check_controls returns a failure.
- contract.py: hardcode FORBIDDEN_VARIABLES = {lifespan_days, died, status, prognosis,
  prognosis_fraction, group, cohort, sex, feeding, genotype, table, hatch_date, animal_id, age_days}.
  validate() always uses it; the suite may only add names.
- At the end, print one line: positive F2 C; negative F1 / F2 / F1+F2 C; PASS or FAIL.

5. TESTS (tests/lifespan/, repeats=5)
- test_synthetic_reproducible: each synthetic adapter called twice with the same seed gives identical
  series (pd.testing.assert_frame_equal); a different seed gives different series.
- test_controls_across_seeds: seeds 0, 1, 2: positive F2 C >= POSITIVE_MIN_C; negative F1, F2 and
  F1+F2 each inside NEGATIVE_BAND.
- test_negative_control_detects_leak: define leaky_cut in the test (like cut_at_landmark but keeps
  every valid day, including age_days >= L). Seeds 0, 1, 2: score_landmark(..., cut_fn=leaky_cut)
  gives negative F1+F2 C >= LEAK_MIN_C.
- test_fail_closed: a suite without the negative control exits non-zero; a negative control with
  zero animals marks the run invalid.
- Remove the 0.35-0.65 band from test_controls.py.
- Confirm test_landmark_invariance and test_contract exist and pass.

GATE: make lifespan-test passes. Run make lifespan twice; the control rows in the ledger are identical
across the two runs. Paste the control summary line from both runs and the pytest summary.
