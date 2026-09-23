# Worm Gait Benchmarks

## 01: Baseline Metrics (`01_worm_gait_baseline_metrics.py`)
Calculates naive baseline statistical metrics and geometric properties directly
from the raw `EigenWorms_TEST.ts` dataset.

## 02: Reviewer 2 Baseline - SSM (`02_worm_gait_aging_ssm.py`)
Evaluates a continuous-time sequence prediction model (`BaselineSSM` wrapped in
`SensorFusionPredictor`) on the worm dataset. Proves that standard MSE-based
sequence forecasting models fail to meaningfully detect the thermodynamic phase
transitions of biological aging (yielding negligible effect sizes compared to
the EBM).

## 03: Reviewer 2 Baseline - Transformer (`03_worm_gait_aging_transformer.py`)
Evaluates the absolute industry standard discrete-time `BaselineTransformer`.
Further validates that standard deep learning architectures are fundamentally
blind to the continuous thermodynamic collapse that defines biological aging.

## 04: Hyperparameter Optimization (`04_worm_gait_optune_ebm_architecture.py`)
An automated `optuna` tuning script designed to search the hyperparameter space
for the optimal network dimensions (e.g., hidden sizes and depths) for both the
micro and macro observers in the `PredictiveCodingGraph`. Saves the best
configuration to
`output/benchmarks/worm_gait/04_worm_gait_ebm_best_params.json`.

## 05: Worm Gait Decline (`05_worm_gait_aging_ebm.py`)
The ultimate integration benchmark operating on the *C. elegans* gait dataset,
orchestrated via `configs/worm_gait_experiments.yaml`. It trains a full
predictive coding graph using a joint Energy-Based Model (EBM). This script
proves that the system successfully converges onto non-linear biological limit
cycles and conclusively demonstrates our ability to measure the thermodynamic
flattening (decline in the Hessian Curvature trace of the Joint EBM) of a
Waddington basin caused by biological aging.

## 06: Infer Biological Lambda (`06_infer_biological_lambda.py`)
Infers the latent biological precision parameter ($\lambda$) that governs the
thermodynamic curvature of the 'Old Worm' state relative to the trained 'Young
Worm' engine. Provides the empirically grounded baseline parameter for the
downstream intervention sweep.

## 07: In-Silico Reprogramming & Thermodynamic Rescue (`07_worm_gait_intervention.py`)
Serves as the scientific climax of the project (Figure 4), driven by
`configs/worm_gait_experiments.yaml`. It demonstrates mathematically that an
erratic "old" biological trajectory can be rescued by applying a synthetic
control force (Precision Injection). By directly modifying the precision
topology of the Markov blanket during an Euler-Maruyama SDE rollout (via
`jax.lax.scan`), it artificially steepens the Waddington basin, forcing the
degraded system back into a youthful, tight biological limit cycle. The script
serializes summary statistical metrics (KS-statistic, Wasserstein distance,
Cohen's d) alongside 3D phase-space plots to precisely quantify the
thermodynamic restoration.

## 08: Lambda Sweep (`08_worm_gait_lambda_sweep.py`)
Performs a dose-response sweep across various values of the precision injection
parameter $\lambda$. It systematically calculates the resulting mean Hessian
trace and Cohen's $d$ effect sizes for each discrete intervention intensity.

## 09: Pharmacological Translation (`09_pharmacological_translation.py`)
Bridges the gap between abstract thermodynamic geometry and standard clinical
pharmacology. By fitting a 4-parameter logistic (4PL) Hill equation to the
dose-response trace data from step 08, it exacts the $EC_{50}$ intervention
threshold and computes the equivalent macroscopic Gibbs Free Energy ($\Delta G$)
shift required to restore the biological limit cycle.

## 10: Visualization (`10_animate_worm_gait.py`)
Renders a visual GIF animation comparing the raw physical young and old worm
gait data.
