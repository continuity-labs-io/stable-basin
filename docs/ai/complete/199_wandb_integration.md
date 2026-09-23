# 199: Weights & Biases Integration for Worm Gait Pipeline

We will integrate `wandb` into the codified 10-script pipeline to capture all metrics, hyperparameters, plots, and models as tracked artifacts. This replaces local JSON saving and provides complete data provenance.

## Proposed Changes

### Configuration Updates
- Add a shared `logging.wandb_project: "worm_gait"` config value to all YAML configuration files so runs appear in a unified dashboard.

### Script-by-Script Updates

#### 1. Baselines (`01`, `02`, `03`, `04`)
- Inject `wandb.init()` at the start of these scripts.
- Log the final baseline metrics using `wandb.log()`.
- Log the baseline plots (`.png`) via `wandb.Image()`.

#### 2. Hyperparameter Optimization (`05_worm_gait_optune_ebm_architecture.py`)
- Import and use `optuna.integration.WeightsAndBiasesCallback`.
- Log the optimal `best_params` configuration as an artifact or run summary.

#### 3. EBM Training (`06_worm_gait_aging_ebm.py`)
- The `EchoRunner` natively supports W&B metrics logging, but we need to extend the script to log the final trained weights (`06_worm_gait_decline_trained_engine.eqx`) as a `wandb.Artifact(type='model')`.
- Log the `06_worm_gait_decline_ablation.png` via `wandb.Image()`.

#### 4. Inference & Dose-Response Sweeps (`07`, `08`, `09`, `10`)
- Wrap each script with a `wandb.init()` context.
- In the intervention script (`07`), log the `_rescue.png` trace comparisons.
- In the Lambda Sweep (`08`), log the dose-response trace plots and the numerical sweep metrics dictionary.
- In Pharmacological Translation (`09`), log the final calculated `EC50` biological parameter and the pharmacological translation curve.
- Ensure any loaded `.eqx` weights are recorded as a *used artifact* (e.g., `run.use_artifact()`) to establish the lineage from training -> inference.

## Questions

- Should we keep writing the local `.json` and `.png` files to the filesystem in addition to `wandb`, or completely replace the filesystem output with `wandb` exclusively? Yes, this is good for human verification.
- For the `wandb_project` name, should we unify all these under `"worm_gait"` or split them into sub-projects like `"worm_gait_baselines"` and `"worm_gait_training"`? Let's keep them all under `"worm_gait"`.
