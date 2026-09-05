# DESIGN DOCUMENT: The ECHO Training Harness

## 1. Objective

To build a scalable, distributed, JAX-native training pipeline that integrates flawlessly with your existing Ray Tune and Weights & Biases (W&B) infrastructure. This harness will train the PredictiveCodingGraph to autonomously discover and lock onto biological limit cycles using Free Energy minimization (predictive coding), proving the architecture works on real empirical data.

## 2. Architectural Philosophy: The PyTorch-JAX Firewall

The greatest risk in this pipeline is framework collision.

PyTorch is our Data layer. It handles multi-processing, disk I/O, and batch yielding via torch.utils.data.DataLoader (e.g., MullerBrownDataset, PharmacologicalShockDataset).

JAX/Equinox is our Compute layer. It handles the physics, the EBMs, and the continuous-time unrolling via XLA compilation.

The Firewall Rule: PyTorch tensors must NEVER cross into JAX compute logic, and JAX arrays must NEVER cross into PyTorch dataset logic. We will enforce a strict numpy() conversion bridge inside the training loop to prevent PyTorch from holding onto CUDA memory while JAX is trying to compile.

## 3. The Theoretical Strategy: Partially Observable BPTT

In biological datasets, we do not observe the entire universe ($d_{state}$). We only observe the sensory data (e.g., HD-MEA voltage spikes). The Internal ($\mu$) and Macro states are latent variables.

We will train the network using Teacher-Forced Sequence Prediction:

* We feed the true biological sequence into the seq argument of forced_unroll.
* At step $t$, the engine is forced to inject the true data $s_{true}(t)$ into the sensory slice of the state.
* The engine computes the physics step, producing the full predicted state $x(t+1)$.
* We define the Loss Function as the Mean Squared Error (MSE) between the sensory slice of $x(t+1)$ and the actual next data frame $s_{true}(t+1)$.

By minimizing this sensory surprisal via Backpropagation Through Time (BPTT), the gradients flow into the unobserved latent variables, forcing the EBM ($E_\theta$) to carve out a Waddington basin that explains the sensory data.

## 4. Core Components

To maintain strict separation of concerns, we will place the new components in the src/echo/harness/ directory.

### A. The JAX Trainer (src/echo/harness/echo_trainer.py)

This is the functional, pure-JAX equivalent to your StableBasinTrainer.

* The Optimizer: We will use optax (e.g., optax.adamw).
* State Management: In Equinox, the model is the state. The trainer will cleanly filter trainable parameters (EBM weights, $\Gamma$, $Q$) from static configurations (Topology masks, dimensions) using eqx.filter.
* The Update Step (make_step): A highly optimized @eqx.filter_jit function that wraps @eqx.filter_value_and_grad. Crucially, it will apply gradient clipping—unrolling SDEs can cause gradients to explode, and clipping keeps the training stable.

### B. The Orchestrator (src/echo/harness/echo_runner.py)

This script mirrors the structure of clinical_diagnostic_runner.py but orchestrates the JAX training loop.

* Initialize Ray & W&B: Setup distributed logging.
* Load Data: Instantiate the requested PyTorch Dataset and DataLoader.
* Initialize Architecture: Build the PredictiveCodingGraph and Optax optimizer.
* The Burn-In Phase (Training): Iterate over the DataLoader for $N$ epochs. Use the PyTorch-JAX bridge. Log the training loss to W&B.
* The Validation Hook (The Proof): At the end of an epoch, pass a validation sequence through the graph. Attach the HessianCurvatureTracker to the Macro-EBM and log the hessian_trace to W&B. You will literally watch the attractor basin steepen on your dashboard as the model learns!

### C. The Configuration (configs/echo_training.yaml)

A declarative YAML config specifically for ECHO runs.

* Data Settings: Dataset type, seq_len, batch_size.
* Architecture Settings: d_internal, ebm_hidden_size, temperature, dt.
* Training Settings: learning_rate, epochs, clip_grad_norm.

## 5. Phased Implementation Plan (The Micro-Prompts)

To keep the coding agent focused and prevent XLA compilation nightmares, we will execute this in three granular prompts:

* Prompt 1: The JAX Trainer (echo_trainer.py)
  Focus: Equinox JIT compilation, Optax state management, gradient clipping, and defining the partially observable loss function.
* Prompt 2: The Ray Tune Orchestrator (echo_runner.py & YAML)
  Focus: Safely piping PyTorch DataLoaders into JAX arrays, defining the hyperparameter search space, and W&B/Ray integration.
* Prompt 3: The Waddington Benchmark Upgrade
  Focus: Strip the hacky weight-mutations out of 01_waddington_collapse.py and replace it with a call to the new EchoTrainer to pre-train the model on healthy baseline data before running the diagnostic crash.
