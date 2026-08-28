### Paper 1: The Core Foundation (Our Current Focus)
**"Latent Stasis: Mask-Aware Subspace Routing for Asynchronous State Space Models"**
* **The Vibe:** Hardcore Machine Learning Architecture.
* **The Problem:** Sensor fusion in continuous-time AI fails because zero-padding asynchronous inputs actively decays the latent state.
* **The Solution:** The Orthogonal Subspace Router dynamically modulates the $\Delta t$ and $B$ parameters to mathematically freeze hidden channels when sensors drop offline.
* **Why it Matters:** This establishes you as the engineer who solved the missing-data problem for modern state-space models. It requires no wet-lab data, making it fast to publish in top ML venues.

### [ICEBOXED] MASR Mamba on Toxic Shock test

*Note: We have pivoted away from deterministic CUDA/Triton development to hierarchical, energy-based state space models.*

**The Problem:** The MASR Mamba architecture instantly trips the Thermodynamic Diagnostic Engine (KSM metric) after the burn-in grace period (frame 501) on simple synthetic sine waves, while linear SSMs remain perfectly stable until the simulated precursor spike at frame 950. Mamba's data-dependent state transition matrices ($B$, $C$, $\Delta t$) naturally inject non-linear chaos into the continuous state trajectory during periodic signals, causing its baseline KSM to hover below the strict `0.95` tripwire.

**Action Items:** Investigate and tune the MASR Mamba to stabilize its baseline KSM so it can accurately detect the toxic shock phase transition. Evaluate one of the following approaches:
- **Option A (Threshold Tuning):** Relax the `ksm_threshold` specifically for the Mamba architecture (e.g., to `0.90`) to accommodate its inherently chaotic, data-dependent state updates during healthy baseline periods.
- **Option B (Architectural Regularization):** Tune the model's hyperparameters (e.g., lower the learning rate) or apply structural regularization (e.g., freeze specific projections or add state constraints) to force it to behave more rigidly like a linear SSM on stable periodic data.

### Paper 3: The "Glass Box" Safety Protocol
**"Thermodynamic Diagnostic: Exact Relevance Propagation for Continuous Biological Trajectories"**
* **The Vibe:** AI Safety / Explainable Biology / DARPA-Grade AI.
* **The Problem:** Even if an AI perfectly predicts a catastrophic biological event (e.g., a Waddington crash), it is useless to clinicians if it acts as a black box.
* **The Solution:** You bring your `mamba_lrp.py` and `diagnostic_engine.py` modules into the spotlight. You introduce MambaLRPEpsilon, proving how to perfectly conserve attribution backward through the continuous $\exp(A \Delta t)$ matrix.
* **Why it Matters:** Explainability is the ultimate bottleneck for FDA-approved biological foundation models. When the model predicts a crash, this engine traces the exact causal chain back to the root event (e.g., "A TP53 RNA flash at T-30 mins caused the collapse").

### 1. [ICEBOXED] The CUDA/Triton Quest: "Exact Mamba-LRP Kernel"

*Note: We have pivoted away from deterministic CUDA/Triton development to hierarchical, energy-based state space models.*

* **Target:** Low-level Systems Engineers, CUDA/Triton Hackers, AI Alignment Researchers.
* **Where it lives in your code:** The docstring at the bottom of `src/demo/raw/6_mamba_lrp_demo.py`.
* **The Problem:** You explicitly requested replacing the First-Order approximation with a mathematically exact Layer-wise Relevance Propagation (LRP-$\epsilon$) ruleset for Mamba-2.
* **The Pitch:** *"Standard interpretability tools fail on continuous-time biology. We need a systems hacker to write a custom PyTorch backward hook directly into Tri Dao’s `mamba_inner_fn` (the fused Triton kernel). It must distribute relevance across the continuous-time discretization parameters ($\Delta, A, B, C$) without breaking $O(N)$ memory complexity."*

### 3. [ICEBOXED] The Mechanistic Interpretability Quest: "Native H-SSM Dictionary Learning"

*Note: We have pivoted away from deterministic CUDA/Triton development to hierarchical, energy-based state space models.*

* **Target:** AI Interpretability Researchers, Sparse Autoencoder (SAE) enthusiasts.
* **Where it lives in your code:** The highly self-aware docstring at the top of `src/metrics/spd_interpreter.py`.
* **The Problem:** You monkey-patched the Goodfire SPD library to force it onto Mamba's 1D Convolutions, noting it is "extremely fragile" and crashes due to SVD NaNs.
* **The Pitch:** *"LLM-centric interpretability tools are too brittle for continuous state-space biology. We need a researcher to build a standalone PyTorch implementation of Stochastic Parameter Decomposition (Sparse Dictionary Learning) designed natively for the `Conv1d` and `Linear` blocks of a Hierarchical-SSM."*
