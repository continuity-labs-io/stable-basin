# Echo Benchmarks

This directory contains a sequence of physics-based benchmarks that validate the mechanical properties, diagnostics, and clinical workflows of the Active Inference models under various physiological failure modes.

## 01: Waddington Collapse (`01_waddington_collapse.py`)
Simulates the collapse of the Waddington epigenetic landscape during biological aging. This benchmark measures the Energy Basin Escape Time (EBET) by tracking the degradation of the `HessianCurvatureTracker` prior to a total electrical crash, proving that thermodynamic structural collapse precedes electrical failure.

## 02: Silent Drift (`02_silent_drift_benchmark.py`)
Evaluates hierarchical blindness in the predictive coding graph. This benchmark demonstrates how extreme structural decoupling (e.g. aging/trauma) causes micro-level prediction errors to be absorbed silently without perturbing the macro-level belief state, rendering top-down homeostasis impossible.

## 03: Concurrent Contention (`03_concurrent_contention_benchmark.py`)
Tests the mathematical stability of the continuous-time physics engine under severe antagonistic conditions. It pits competing thermodynamic forces and non-equilibrium steady states against each other to ensure the `Thermostat` and `SolenoidalFlow` do not explode into numerical singularities during contention.

## 04: Decidability Diagnostic (`04_decidability_diagnostic.py`)
A rigorous physical diagnostic that differentiates between two catastrophic biological failures:
- **Policy Observability Failure (Patient A)**: The mechanical brakes remain intact, and external actuation can successfully restore homeostasis.
- **Reachability Collapse (Patient B)**: The internal friction has evaporated. The system is structurally unrecoverable regardless of external interventions.

## 05: Clinical Workflow Demo (`05_clinical_workflow_demo.py`)
An end-to-end demonstration of the 5-step clinical lifecycle on a simulated aging patient:
1. **Listen**: Synthetically degrade the patient's structural integrity.
2. **Measure Geometry**: Calculate the flattened Waddington basin curvature.
3. **Ping**: Inject a bioelectric shock to detect Silent Drift (Hierarchical Discordance).
4. **Compute Counterfactual**: Mathematically anneal the Digital Twin in silico to forge an optimal reference point.
5. **Actuate**: Extract the exact exogenous hardware energy (`Q_actuation`) required to push the patient back into a youthful limit cycle.

## 06: Worm Gait Decline (`06_worm_gait_decline.py`)
The ultimate integration benchmark operating on the *C. elegans* gait dataset. It orchestrates full predictive coding graph training via dual configurations (Laplace Baseline vs Precision-Weighted EBM). This script proves that the system successfully converges onto biological limit cycles and conclusively demonstrates our ability to measure the thermodynamic flattening (Hessian Curvature decline) of a Waddington basin caused by biological aging.

## 07: Nonlinear EEG Entropy (`07_nonlinear_eeg_entropy.py`)
Validates the ECHO Harness on human EEG datasets to determine if our nonlinear estimators capture non-Gaussian irreversibility (the true thermodynamic arrow of time) that linear phase-lag models miss. This script orchestrates an Arrow-of-Time neural network (Estimator D) and a Fitted NESS EBM (Estimator E) using a `PredictiveCodingGraph` to detect irreversibility in real vs phase-randomized surrogate data.

## 08: In-Silico Reprogramming & Thermodynamic Rescue (`08_insilico_reprogramming.py`)
Serves as the scientific climax of the project (Figure 4). It demonstrates mathematically that an erratic "old" biological trajectory can be rescued by applying a synthetic control force (Precision Injection). By directly modifying the precision topology of the Markov blanket during an Euler-Maruyama SDE rollout (via `jax.lax.scan`), it artificially steepens the Waddington basin, forcing the degraded system back into a youthful, tight biological limit cycle.
