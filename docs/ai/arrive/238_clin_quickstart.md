Load the following files into your context:
- `src/harness/clinical_diagnostic_runner.py`
- `src/metrics/diagnostic_engine.py`
- `src/benchmarks/aging_resilience/11_null_control.py`
- `Makefile`
- `README.md`

Please execute the following updates to fix the demo pipelines, remove hardcoded reporting hacks, and clean up the documentation. Adhere to subdued, professional language and logging (no emojis).

1. **Honest Clinical Quickstart Runner:**
In `src/harness/clinical_diagnostic_runner.py`:
- In the `__main__` block, add a `--no-ray` boolean flag. If this flag is passed, bypass `tune.Tuner` entirely and just run `evaluate_model` in a standard Python loop for the models defined in the config.
- In `evaluate_model`, remove the circular noise injection hack (`telemetry[:, crash_frame_true - 50 : crash_frame_true, 120:130] = ... + 5.0`). The attribution should find organic deviations, not artificially planted spikes.

2. **Honest Diagnostic Reporting:**
In `src/metrics/diagnostic_engine.py`:
- Modify `generate_diagnostic` so it evaluates if a crash actually occurred. If the system never crossed the KSM threshold, change the `"status"` to `"NOMINAL"` and do not hardcode the `"confidence_score"` to `0.98` (dynamically compute it or default to a baseline probability).
- Update the `mechanism` mapping logic. The current logic checks for prefixes like `"Psi"` or `"Omega"`, which don't match the actual feature names (e.g., `"RNA_TP53"`, `"VoltGrn"`). Map them correctly based on `self.feature_names`.

3. **Synthetic Aging Configuration & Fallback:**
- Create a new file at `configs/synthetic_aging.yaml`. Copy the structure of `configs/aging_resilience.yaml` but configure it for an ultra-fast smoke test: set `dataset.name: "worm_gait"`, `dataset.seq_len: 100`, `experiment.optuna_n_trials: 1`, `optimization.max_epochs: 2`, and `dataset.batch_size: 2`.
- In `src/benchmarks/aging_resilience/11_null_control.py`, implement a safe fallback to `SyntheticWormMockDataset` if `args.train_ts` or `args.test_ts` do not exist locally, rather than raising a `FileNotFoundError`. Make sure to initialize the "Old" mock cohort with `degraded=True`.

4. **Makefile Updates:**
In `Makefile`:
- Add a new target `quickstart:` that executes:
  `WANDB_MODE=disabled python -m src.harness.clinical_diagnostic_runner --config configs/clinical_diagnostic.yaml --use_synthetic --no-ray`
- Add a new target `run-synthetic-aging:` that executes:
  `WANDB_MODE=disabled $(MAKE) aging-resilience-experiments DATASET=synthetic_aging`

5. **README Cleanup:**
In `README.md`:
- Replace all references of `MaskAwareMamba` with `MaskAwareSSM`.
- Remove the phrase "Mathematically exact" when describing the LRP implementation, as it relies on an epsilon-stabilized approximation.
- Fix the citation block URL formatting (change `\url{[https://...](https://...)}` to simply `\url{https://github.com/continuity-labs-io/stable-basin}`).
- Remove references to the missing `OPEN_PROBLEMS.md` file, replacing it with a generic call to check the GitHub Issues tab.
