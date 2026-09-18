Master Design Document: Stable Basin Project
Scope: EEG Entropy (Linear & Nonlinear), ECHO Training Harness, and Worm Gait Validation

Status: Unified Execution Blueprint

# 1. Executive Summary & Engineering Dependency Graph
This document consolidates three parallel research and engineering vectors—EEG Entropy Biomarkers, the ECHO Training Harness, and the C. elegans Worm Gait Benchmark—into a single, sequentially dependent roadmap.

Does the sequencing matter?
Crucially, yes. While the scientific conceptualization progresses logically from linear EEG models to complex nonlinear EEG models, the computational execution sequence is dictated by a strict engineering dependency graph. We must build and validate our "rulers" before our "engines," and we must test our engines on a simple biological organism before pointing them at complex human data.

We will execute this build in four interleaved phases:

Phase 1: The Rulers (EEG Stages 0–2). Built purely on NumPy/SciPy linear estimators. Requires zero JAX training infrastructure, provides immediate ground-truth validation, and answers clinical questions about sleep and aging.

Phase 2: The Engine (ECHO Harness). The PyTorch-to-JAX Optax training pipeline (echo_trainer.py, echo_runner.py). This must be built before any Energy-Based Model (EBM) can be trained.

Phase 3: The Organism (Worm Gait Validation). Acts as the ultimate integration test for the Harness. Before throwing complex, noisy human EEG data at the EBM, we must prove the training loop converges natively on a clean, low-dimensional (6D) biological limit cycle.

Phase 4: The Human (EEG Stage 3). With the Harness and EBMs fully validated, we apply them back to the human EEG data to compute nonlinear entropy production (EP).

# 2. Critical Technical Refinements & Constraints
Before freezing this architecture into code, the following systemic watchouts are hardcoded into the pipeline to prevent structural breakdowns and hardware bottlenecks:

The Γ Masking Bug (Fluctuation-Dissipation Violation):

The Issue: Masking the dissipative matrix Γ element-by-element destroys its positive-definiteness. This breaks the fluctuation-dissipation theorem, ruining the stationary density e^{-E/T} and making the EP formula mathematically invalid.

The Resolution: For unpartitioned systems like the EEG application (Estimator E), we will introduce a use_blanket_topology=False flag to PredictiveCodingGraph to disable the mask. For partitioned systems (Cells/Worm Gait), we will re-parameterize Γ so it is strictly positive-definite within the allowed block-diagonal pattern.

The PyTorch-JAX Bridge (PCIe Bottleneck):

The Issue: Moving tensors from PyTorch (GPU) → NumPy (CPU) → JAX (GPU) creates an unacceptable PCIe bandwidth bottleneck.

The Resolution: We will strictly use DLPack (jax.dlpack.from_dlpack) for zero-copy memory transfers directly in the GPU VRAM. This strictly honors the framework firewall (PyTorch for data, JAX for compute) while keeping execution blisteringly fast.

Variable Sequence Lengths vs. JAX JIT:

The Issue: JAX @jit requires static shapes; passing variable-length sequences from the Open Worm Database will cause JAX to expensively recompile the XLA graph on every single batch.

The Resolution: PyTorch DataLoaders (e.g., CElegansGaitDataset) must extract random, contiguous, fixed-length crops (e.g., seq_len=500) during training.

# 3. Phase 1: The Rulers (Linear EEG Entropy, Stages 0–2)
We will measure entropy production (EP), the degree to which brain dynamics are irreversible in time, from scalp EEG. This phase relies on computationally lightweight linear (multivariate Ornstein–Uhlenbeck) models.

## 3.1 Estimator Definitions (Pure NumPy/SciPy)
EP (Φ) is carried entirely by the solenoidal flow Q. For a linear process, Φ_MOU = −Tr(Γ^{-1} QΣ^{-1} Q) ≥ 0.

Estimator A (Lag-τ Pairwise): A fast lower-bound trace comparison of forward vs. backward joint distributions.

Estimator B (MOU Fit): Computes transition matrix A, dissipation Γ, and solenoidal flow Q explicitly. Provides topographical maps of nodal irreversibility.

Estimator C (Spectral - Primary): Exact EP of the Gaussian process based on cross-spectral matrix. Band-resolved EP spectral density (φ(f)) requires zero Markov assumptions.

## 3.2 Datasets & Execution Plan
Stage 0 (Ground Truth): Closed-form MOU simulations in CI verifying the estimators and surrogate generation (reversible Gaussian, phase-randomized) recover exact math (Φ=2q^2).

Stage 1 (Calibration): Verify that EP drops from Wakefulness to deep sleep (N3). Uses Sleep-EDF (2-channel smoke test) and ANPHY-Sleep (83-channel calibration).

Stage 2 (Aging): Test if EP differs between young and older adults, and if it adds predictive power to age-prediction models beyond standard spectral features. Uses MPI-LEMON and the Dortmund Vital Study.

# 4. Phase 2: The Engine (ECHO Training Harness)
The generic infrastructure to train the PredictiveCodingGraph to autonomously discover biological limit cycles via Free Energy minimization.

## 4.1 Theoretical Strategy: Partially Observable BPTT
In biological datasets, we only observe sensory data (EEG voltages or worm posture); the internal and macro states remain latent.

Teacher-Forced Sequence Prediction: The engine is forced to inject true data s_true(t) into the sensory slice of the state. It computes the physics step to predict the full state x(t+1).

The Loss: Mean Squared Error (MSE) between the sensory slice of predicted x(t+1) and the actual next data frame. Backpropagation Through Time (BPTT) forces the latent Waddington basin to align to the observed dynamics.

## 4.2 Core Architecture
Trainer (echo_trainer.py): Pure functional JAX/Equinox. Uses optax.adamw. Applies eqx.filter to separate trainable weights (E_θ, Γ, Q) from static topologies. Uses highly optimized @eqx.filter_jit update steps with optax.clip_by_global_norm to prevent SDE gradient explosions.

Bridge (pytorch_jax_bridge.py): The DLPack zero-copy tensor handoff separating PyTorch data loaders from JAX arrays.

Orchestrator (echo_runner.py): Orchestrates the loop. Integrates Ray Tune and Weights & Biases (W&B). Attaches the HessianCurvatureTracker during the validation hook to log and visualize basin steepening.

# 5. Phase 3: The Organism (Worm Gait Integration)
Before pointing the JAX harness at high-dimensional human EEG, we run an integration test on C. elegans locomotor decline.

## 5.1 The Task & Dataset
Data: 6-dimensional eigenworm posture projections capturing a literal biological limit cycle.

DataLoader (celegans_gait_dataset.py): Yields seq_len=500 static crops. Includes a seeded synthetic 6D limit cycle fallback for CI testing.

Target: Train only on healthy young worms (Days 1–3); evaluate generalizability and basin deterioration on old worms (Day 9+).

## 5.2 The EBM Ablation Comparison
Does a learned, multimodal landscape (E_θ) capture biological aging better than a single rigid basin? We compare two variants side-by-side:

GaussianEBM (The Laplace Baseline): E(x) = 1/2 (x−μ)^⊤ Π(x−μ). The Hessian is exactly Π. Its Hessian trace will mathematically be a flat line, serving as a strict proof of harness correctness.

PrecisionWeightedEBM (Multimodal MLP): Capable of free-form asymmetric wells. We test if its Hessian trace naturally captures biological decline (shallower basins in older worms) without ever being trained on old data.

# 6. Phase 4: The Human (Nonlinear EEG, Stage 3)
With the ECHO Harness validated on worm gait, we bring it back to Phase 1's EEG pipeline to answer: Do nonlinear estimators capture non-Gaussian irreversibility (e.g., asymmetric waveforms, sharp rises/slow decays) that linear phase-lag estimators miss?

## 6.1 The Nonlinear Estimators
Estimator D (Model-Free Classifier): Trains an Arrow-of-Time or NEEP neural network to distinguish forward vs. time-reversed windows, providing a pure non-Gaussian EP lower bound.

Estimator E (Fitted NESS EBM): Applies the PredictiveCodingGraph to the PCA-reduced components of human EEG. Set use_blanket_topology=False so Γ remains full-rank. EP is computed via Monte Carlo sampling over the stationary distribution of the fitted model.

## 6.2 Decision Rule
Run Estimator D on real data vs. phase-randomized surrogates. We adopt the nonlinear models as primary metrics only if Estimator D isolates non-Gaussian EP, Estimator E reliably beats the MOU model on held-out likelihood, or they yield significantly larger effect sizes for Sleep/Age differences than Estimator C. Otherwise, we publish that linear phase-lag is sufficient.

# 7. Execution Roadmap (The 8 Prompts)
This document is now ready to be transformed into code. To prevent XLA compilation nightmares and maintain strict test coverage, we will execute exactly one prompt at a time:

## Phase 1: The Rulers (EEG Entropy Stages 0–2)
Prompt 1: Write src/metrics/entropy_production.py (Estimators A, B, C) and src/metrics/ep_surrogates.py.

Prompt 2: Write tests/test_entropy_production.py (Stage 0: Synthetic MOU ground-truth tests).

## Phase 2: The Engine (ECHO Training Harness)
Prompt 3: Fix the Γ masking bug in observer.py (add use_blanket_topology logic) and write src/echo/harness/pytorch_jax_bridge.py using DLPack.

Prompt 4: Write src/echo/harness/echo_trainer.py (Optax loop, @eqx.filter_jit, gradient clipping, and the sensory MSE teacher-forcing loss).

Prompt 5: Write src/echo/harness/echo_runner.py (Orchestration, W&B logging) and configs/echo_training.yaml.

## Phase 3: The Organism (Worm Gait Integration)
Prompt 6: Write src/data/behavior/celegans_gait_dataset.py (with static seq_len=500 crops and seeded synthetic 6D fallback) and GaussianEBM (in primitives/ebm.py).

Prompt 7: Write src/echo/benchmarks/06_worm_gait_decline.py to run the ablation and output the 3-panel plot and Hessian trace comparison.

## Phase 4: The Human (EEG Nonlinear)
Prompt 8: Build the MNE data loaders (sleep_edf.py, lemon.py), integrate them with EchoRunner, and execute Stage 3 (Nonlinear EEG Entropy).
