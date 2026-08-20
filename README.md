# Stable Basin 🥣

> **Benchmarking continuous-time AI on its ability to maintain biological homeostasis against entropic decay.**

The objective is to maintain the biological latent state inside the youthful homeostatic attractor basin, evaluated by Time-in-Basin (TiB) against entropic decay and simulated hardware failures.

Stable Basin Benchmark is a research repository for benchmarking continuous-time
multiscale biological datasets using State Space Models (SSMs). The project
focuses on fusing high-frequency electrophysiological data with lower-frequency
optical imaging, orthogonalizing hardware artifacts, and performing real-time
biological anomaly detection using self-supervised predictive coding.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Traditional AI leaderboards rank models on *Mean Squared Error* or *Next-Token Prediction*. **Stable Basin** ranks models on **Survival**.

Stable Basin is an open-source evaluation suite designed to stress-test infinite-horizon State Space Models (SSMs) and continuous sequence architectures. It measures a model's ability to lock onto and maintain a "youthful" biological attractor basin against entropic decay, systemic shocks, and catastrophic hardware failures.

## 🎯 The Mission: Defeating Entropy

Biological youth isn't about predicting the future; it's about holding the line. A living organism exists in a highly specific, high-energy *Stable Basin* of homeostasis. Mechanical noise, DNA methylation drift, and environmental shocks are constantly trying to knock it out of that basin.

We evaluate continuous-time machine learning architectures (like Mamba-2) on their ability to act as the ultimate biological flight computer: processing multi-modal, high-frequency telemetry (wearables, electrophysiology, epigenetics) to detect critical instability *before* a physiological crash occurs.

### Why "Stable Basin"?
It's an impossible balancing act. The datasets in this repository are designed to be explicitly hostile. Sensors will randomly drop offline, hardware will vibrate, and the tissue will undergo variance explosions. If the AI relies on naive temporal memorization instead of deep spatial covariance, the latent geometry shatters, and the model "falls off the wall."

---

## 🏆 The Leaderboard Gauntlet (The "Routes")

Stable Basin is divided into four progressively brutal trial routes. Models are ranked by **Time-in-Basin (TiB)**. How many continuous frames can your physics engine hold the grip before the biological latent state slips over the edge?

### Route 1: The Quake (Hardware Veto)
Can the model hold the biological signal while ignoring a massive mechanical earthquake?
*   **The Test:** Multi-modal data corrupted by a massive 2Hz microfluidic pump artifact.
*   **The Failure State:** Hallucinating a biological crash due to hardware wobble.

### Route 2: The Blind Reach (Fault Tolerance)
Can the model impute missing biology using pure spatial covariance?
*   **The Test:** Mid-sequence, 15% to 50% of the hardware sensors permanently output `NaN`.
*   **The Failure State:** Catastrophic network collapse when the primary input dimensions vanish.

### Route 3: The Wobble (Critical Slowing Down)
Can the model detect the phase transition *before* it happens?
*   **The Test:** Tracking the physical 'wobble' (variance) and sluggishness (lag-1 autocorrelation) of a system approaching a saddle-node bifurcation.
*   **The Leaderboard Stat:** **Detection Latency.** How many milliseconds in advance does the AI radar trigger the alarm?

### Route 4: The Scar (Rejuvenation Hysteresis)
If the system falls out of the basin, can it compute the optimal path back in?
*   **The Test:** A biological tissue is shocked, then rescued with an intervention.
*   **The Leaderboard Stat:** **Hysteresis Area.** The most efficient model leaves the smallest topological "scar" between the aging trajectory and the rejuvenation trajectory.

---

## ⚙️ The Reference Architecture: `MaskAwareMamba`

To provide a baseline for the benchmark, this repository includes the **MaskAwareMamba**, a continuous-time state-space reference architecture powered by **Mamba-2**. 

Unlike standard Transformers that suffer from $\mathcal{O}(N^2)$ context limits and rely on discrete tokens, the `MaskAwareMamba` utilizes Mask-Aware Subspace Routing to dynamically modulate the flow of time and maintain an $\mathcal{O}(1)$ VRAM footprint on edge hardware.

> [!NOTE] 
> **Why no ODE-RNNs?** While continuous-time Ordinary Differential Equation (ODE) RNNs were initially evaluated for this engine, they were excluded from the final benchmark suite. Adaptive ODE solvers (like `dopri5`) exhibit catastrophic computational stiffness when modeling high-frequency biological phase transitions, resulting in inference latencies $>10^4\times$ slower than our Mamba (ZOH discretized) architectures.

### Quickstart: Push-Button Cloud Execution

Stable Basin uses declarative YAML configurations for distributed parallel execution.

**1. Run the CI/CD Smoke Test**
Ensure your hardware and registry are perfectly configured:
`make preflight`

**2. Execute the Clinical Diagnostic Pipeline**
This command spins up parallel workers to evaluate the architectures simultaneously against the pharmacological crash dataset:
`make clinical-diagnostic`
*All inference latencies, KSM traces, and MambaLRP Causal Diagnostic JSONs will automatically sync to your W&B cloud dashboard.*

## 🧬 Thermodynamic Metrics

The core of the Stable Basin is our deterministic physics evaluation suite (src/metrics/), which extracts true macroscopic variables from the model's latent embedding space:

- **Koopman Stability Metric (KSM)**: Dynamic Mode Decomposition (DMD) to calculate the stable eigenvalue bounds of the biological attractor.

- **Critical Slowing Down (CSD)**: Variance and AR1 tracking to detect phase transitions before they occur.

- **Fedichev Macrostates**: Tracking the continuous accumulation of configurational entropy ($Z$) over millions of frames.

- **MambaLRP-Epsilon**: Mathematically exact Layer-wise Relevance Propagation designed explicitly for continuous-time SSMs to trace crashes back to their root biological circuit.

## 🔍 Open Research Problems

The core math of Stable Basin is written, but we are looking for engineers to take ownership of specific infrastructure nodes (e.g. CUDA/Triton Mamba-LRP kernels, Sim2Real dataloaders, Gymnasium environments). 

If you want to solve aging and build out these missing nodes, please check out the [Open Problems Board](OPEN_PROBLEMS.md) in the repository. Pick a constraint, and open a PR!

📄 Citation

If you use Stable Basin to benchmark your infinite-horizon sequence models or biological anomaly detection, please cite:

```
@misc{stable_basin_2026,
  title={Stable Basin: Benchmarking Infinite-Horizon Sequence Models on Thermodynamic Resilience and Rejuvenation Hysteresis},
  author={Continuity Labs},
  year={2026},
  publisher={GitHub},
  howpublished={\url{[https://github.com/continuity-labs-io/stable-basin](https://github.com/continuity-labs-io/stable-basin)}}
}
```
