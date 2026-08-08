# MeldBenchmark

MeldBenchmark is a research repository for benchmarking continuous-time
multiscale biological datasets using State Space Models (SSMs). The project
focuses on fusing high-frequency electrophysiological data with lower-frequency
optical imaging, orthogonalizing hardware artifacts, and performing real-time
biological anomaly detection using self-supervised predictive coding.

## Core Concepts

### The "Drowning Signal" Environment

The `ToyBiologicalEnvironment` simulates a realistic, complex multiscale
biological recording:

- **GEVI (Genetically Encoded Voltage Indicator) Data:** Sampled at 20kHz,
  containing sparse 1ms biological spikes (action potentials).
- **Optical Data:** Sampled at 100Hz, representing a continuous macro-biological
  state.
- **Artifacts:** Both modalities are corrupted by a massive 2Hz sine wave
  representing a mechanical pump vibration (the "Drowning Signal").
- **Anomalies:** The environment supports injecting "Corrosion" (hardware
  failure causing baseline drift) and "Toxic Shock" (biological crashes causing
  variance explosions).

### State Space Engine (Fusion Core)

The core architecture (`src/models/state_space_engine.py`) uses a **Mamba SSM**
to model the continuous kinetic trajectory of the biological state:

1. **Edge Compression:** A 1D Convolution processes the 20kHz GEVI data into a
   lower-dimensional latent representation.
2. **Fusion:** The compressed GEVI latents are fused with the 100Hz optical
   stream.
3. **Forward Predictive Coding:** The Mamba SSM performs self-supervised forward
   prediction, outputting a "Surprise" metric (Cosine Distance) between
   predicted and actual future states.
4. **Orthogonal Veto:** The network is trained on homeostasis data to
   mathematically isolate biological spikes and orthogonalize (veto) the massive
   2Hz pump artifact.

### Thermodynamic Metrics & Latency Benchmarks

The repository evaluates the thermodynamic stability and phase space geometry of
the biological manifold (`src/metrics/thermodynamics.py`):

- **PALC (Pseudo-Arclength Continuation):** Exact Jacobian-based phase space
  volume tracking.
- **DMD (Dynamic Mode Decomposition):** A sliding-window approximation of the
  Koopman operator.
- **Latency Benchmarks:** `src/metrics/bifurcation_benchmark.py` compares the
  latency of exact Jacobian computation (~730ms) vs. DMD (~1.3ms), validating
  DMD as a viable proxy for 100Hz live-streaming applications.

### Interpretability (SPD)

The repository includes experimental integration with the Goodfire AI
**Stochastic Parameter Decomposition (SPD)** library
(`src/metrics/spd_interpreter.py`). It applies structural interpretability
techniques specifically to the internal 1D Convolutional components of the Mamba
architecture to decompose and understand its feature isolation capabilities.

## Getting Started

### Datasets

**HD-MEA NEUROPulse Dataset** For raw 3Brain `.brw` data files (BrainWave
format) recorded from 4,096-channel HD-MEAs, you can use the HD-MEA NEUROPulse
Dataset on Zenodo. This repository was published by researchers from the
University of Pavia and the IRCCS Mondino Foundation. It includes spontaneous
baseline activity and evoked responses. URL:
[https://zenodo.org/records/13908319](https://zenodo.org/records/13908319) You
can place a downloaded `.brw` file into the appropriate directory (e.g.,
`data/ephys/example.brw`) for the dataloaders to use.

### Installation

Ensure you have PyTorch and the required dependencies installed. You should
install the project in editable mode from the repository root:

```bash
pip install -e .
```

_(Note: Mamba SSM has known precision issues on Apple Silicon (MPS). The demo
scripts automatically default to CPU for the Mamba training loop to prevent NaN
instabilities.)_

### Running the Demos

The repository contains four main demonstration scripts:

#### 1. Bio-Blade Engine (`01_bio_blade_engine.py`)

This script demonstrates the Bio-Blade engine, which simulates high-throughput processing of biological data (like electrophysiology). It shows how the system can ingest raw telemetry and score it (e.g., KSM/CSD scores) to detect biological events at high speeds.

```bash
python src/demo/01_bio_blade_engine.py
```

#### 2. Indestructible Edge (`02_indestructible_edge.py`)

This script demonstrates the fault-tolerance and self-healing routing capabilities of the network. It simulates catastrophic hardware failures (e.g., sensor dropouts) and shows how the architecture seamlessly maintains representation and inference despite severe input corruption.

```bash
python src/demo/02_indestructible_edge.py
```

#### 3. Multimodal Autopsy (`03_multimodal_autopsy.py`)

This script demonstrates the Layer-wise Relevance Propagation (LRP) causality engine. It traces back from a catastrophic failure event (Structural Collapse) to uncover the latent root cause (an earlier RNA Stress Alarm) across complex, multimodal sequence data.

```bash
python src/demo/03_multimodal_autopsy.py
```

#### 4. Masked State Space Model (`04_masked_state_space_model.py`)

This script demonstrates the core Masked State Space Model (SSM) architecture. It shows how the model handles sparse, intermittent multimodal sensor data by dynamically gating the continuous state transitions based on sensor availability masks.

```bash
python src/demo/04_masked_state_space_model.py
```

### Running the Experiments

The repository also includes two primary experiments:

#### 1. Train Synthetic Benchmark (`01_train_synthetic_benchmark.py`)

This script trains and evaluates three models (Baseline SSM, Mask-Aware SSM, and a Causal Transformer) on a synthetic Waddington landscape dataset. It establishes the baseline capability of these architectures to learn latent dynamics from gated sensor data.

```bash
python src/experiments/01_train_synthetic_benchmark.py
```

#### 2. Length Extrapolation Stress Test (`02_extrapolation_benchmark.py`)

This script tests the out-of-distribution (OOD) generalization of the models trained on the Waddington dataset on vastly longer sequences. It proves the stability of State Space Models (SSMs) outfitted with the Dual-Lock Stasis system against ghost noise integration over sparse inputs.

```bash
python src/experiments/02_extrapolation_benchmark.py
```

### Running the Latency Benchmark

To compare the execution speed of exact Jacobian metrics vs. DMD approximations:

```bash
python src/metrics/bifurcation_benchmark.py
```
