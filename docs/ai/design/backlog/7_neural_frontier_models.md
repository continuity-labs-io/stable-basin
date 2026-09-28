# Executive Design Document: stable-basin Continuous Dynamics Engine

## 1. Executive Summary

This document details the architectural integration of physical evaluation metrics extracted from the open-source brain-gen repository into the stable-basin architecture. The primary objective is to advance the surrogate endpoints for the LEV Foundation Robust Mouse Rejuvenation (RMR2) combinatorial longevity study.

https://github.com/ricsinaruto/brain-gen

Traditional aging metrics frequently serve as trailing indicators. To evaluate complex, multi-modal interventions in the RMR2 study, stable-basin requires a high-resolution, real-time metric capable of measuring the structural decay of systemic biological resilience. The brain-gen codebase contains highly optimized, rigorously validated mathematical evaluation infrastructure designed for high-bandwidth continuous physical dynamics. By extracting these thermodynamic and spectral evaluation metrics, stable-basin can directly quantify the degradation of the Waddington attractor landscape, yielding a real-time Biological Stability Index for the aging murine cohorts.

## 2. Theoretical Framework: Aging as a Thermodynamic Phase Transition

The integration leverages the extracted codebase to model aging phenomena mathematically, transitioning from static biomarkers to continuous thermodynamic evaluation.

### 2.1. Loss of Fractal Complexity
* The Physics: Highly resilient, adaptable biological networks display 1/f scale-free complexity (pink noise), indicating a system operating near criticality. Systemic aging degrades this complexity into rigid brown noise or chaotic white noise.
* The Extraction: The repository provides optimized estimators for Detrended Fluctuation Analysis (DFA) and Hurst Exponents.
* RMR2 Application: Processing RMR2 telemetry through these functions will yield a continuous Fractal Resilience Score. Effective rejuvenation therapies should mathematically restore the Hurst exponent toward youthful 1/f baselines.

### 2.2. Critical Slowing Down
* The Physics: As the Waddington attractor basin flattens due to systemic aging, recovery from micro-perturbations slows. This manifests as localized spikes in variance and temporal autocorrelation, acting as a wobble preceding a saddle-node bifurcation.
* The Extraction: The sliding-windows module calculates Covariance Entropy and Amplitude Kurtosis across rolling time horizons.
* RMR2 Application: Sudden spikes in covariance entropy across physiological channels will enable stable-basin to detect impending healthspan cliffs in RMR2 control groups prior to visible frailty.

### 2.3. Rollout Divergence
* The Physics: Quantifying the phase-space divergence between an aging baseline trajectory and an interventional trajectory.
* The Extraction: The generative versus ground-truth divergence evaluation logic.
* RMR2 Application: stable-basin can calculate the mathematical distance between a synthetic youthful baseline rollout and the actual telemetry of treated cohorts, accurately quantifying Rejuvenation Hysteresis.

## 3. Component Extraction and Integration Strategy

The integration strictly follows an extract-and-discard methodology, bypassing all discrete constraints to maintain focus on continuous-time biological physical models.

### 3.1. Discarded Architecture
* All vector quantization codebooks and discrete tokenization logic.
* Cross-entropy Next-Token Prediction paradigms.
* Natural language model adapters and embeddings.

### 3.2. Subsystem Mapping
The extracted components will interface natively with the stable-basin metrics engine utilizing standard PyTorch and NumPy operations.

* eval/rollout_metrics.py maps to src/metrics/complexity.py
  * Purpose: Calculates DFA, Hurst, and 1/f decay to measure systemic network rigidity.
* eval/rollout_sliding_windows.py maps to src/metrics/wobble_detection.py
  * Purpose: Computes rolling covariance and autocorrelation to detect Critical Slowing Down.
* eval/eval_runner.py maps to src/harness/rmr_evaluator.py
  * Purpose: Quantifies the phase-space divergence between aging controls and rejuvenated cohorts.
* models/ntd.py maps to src/physics/ness_noise.py
  * Purpose: Simulates Fristonian Non-Equilibrium Steady State baseline noise using optimized Cholesky-banded Ornstein-Uhlenbeck processes.

## 4. Data Flow Pipeline for RMR2 Analysis

1. Ingestion: Continuous telemetry from the RMR2 enclosures flows into the stable-basin ingest layer.
2. State Projection: Data passes through the continuous-time state-space architecture to establish the biological latent state.
3. Execution:
  * src/metrics/wobble_detection.py processes the data for variance and covariance spikes over sliding windows.
  * src/metrics/complexity.py scores the fractal decay over the longitudinal horizon.
4. Output: stable-basin fuses these physical measurements into the Biological Stability Index, providing researchers with a rigorous measurement of biological resilience uncoupled from chronological time.

## 5. Execution Phasing

### Phase 1: Deep Extraction
* Isolate the mathematical evaluation modules from the repository.
* Prune all dependencies on discrete codebooks.
* Sanitize all logging to standard, calm, and professional outputs.
  * Example: ```logger.info('Computing DFA exponent over continuous horizon.')```

### Phase 2: Refactoring and Testing
* Refactor tensor shapes to align with continuous RMR2 telemetry formats.
* Optimize extracted matrix operations for the specific hardware compute environment.
* Implement native test environments enforcing the execution of ```pytest tests/``` and actively prohibiting conda run commands.

### Phase 3: RMR2 Deployment
* Process historical baseline control mice to map standard aging degradation curves using DFA and Covariance Entropy.
* Deploy the evaluation harness against active RMR2 combinatorial treatment arms to determine if interventions are successfully restoring Waddington basin depth.


