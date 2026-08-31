# 🛠️ ENGINEERING TICKET: Implement Torx Observer Zero Port

**Role Context:** Principal Software Engineer / AI Architect

**Target File:** `src/demo/torx/observer_zero.py`

**Related Legacy File:** `src/demo/vessel/0_reaction_diffusion_observer.py`

## 📋 1. Background & Objective

In our legacy architecture, "Observer Zero" was implemented as a 2D spatial Reaction-Diffusion (FitzHugh-Nagumo) system via PyTorch. As part of the **Stable Basin 2.0** paradigm shift, we are evaluating Extropic's Torx framework to model this observer probabilistically.

We must abandon the explicit spatial PDE grid. Instead, we want to establish the macroscopic observer using **causal graph topology** via a Torx Directed Factor Graph (DFG).

**The Goal:** Build a standalone Torx script where a macroscopic, slow-moving "Observer" ingests chaotic, high-frequency micro-states (sensory input) and uses affine transformations and stochastic diffusion to extract and encode a stable, long-term memory state (the macroscopic order parameter).

## 📐 2. Technical Specifications

### Dependencies & Substrate

* **Frameworks:** Strictly JAX, Equinox (`eqx`), Optax, and Torx. **No PyTorch (`torch`) imports.**
* **Hardware Routing:** Force CPU execution for this Torx sandbox to prevent XLA memory fragmentation by setting `os.environ["JAX_PLATFORMS"] = "cpu"` at the very top of the script.
* **Unified Device:** Use our existing device provisioner: `from src.utils.device import get_optimal_device` with `backend="jax"`.
* **Logging:** Use Python's standard `logging` module to narrate the biological physics occurring at each step.

### Component 1: The Macro-Observer Factor

Create a custom Torx factor (`MacroObserverFactor`) that inherits from `torx.factor.AbstractReferenceFactor`.

* **Ports:**
* `input_ports`: Expects `micro_state` (fast, noisy observation, e.g., 16-D) and `prev_macro_state` (recurrent memory, e.g., 2-D).
* `output_spec`: The updated `next_macro_state`.
* Define these explicitly using `jax.ShapeDtypeStruct`.


* **Parameters (Equinox):** Embed an Equinox neural module (e.g., `eqx.nn.Linear` or a small `eqx.nn.MLP`) to project the high-dimensional micro-state into the low-dimensional macro-space.
* **The Physics / Math:** In the `.sample()` method, implement a leaky integrator with stochastic diffusion (acting as a thermodynamic low-pass filter):

$$Macro_{t} = (1 - \alpha) \cdot Macro_{t-1} + \alpha \cdot \text{MLP}(Micro_t) + \sigma \omega$$


*(Where $\alpha$ is a small integration rate like 0.05 enforcing massive temporal inertia, and $\sigma \omega$ is `jax.random.normal` thermal noise).*

### Component 2: The Torx DFG & ChainFactor

* Wrap `MacroObserverFactor` inside a `torx.ChainFactor` to handle recurrent sequential unrolling over continuous time (e.g., `n_steps=200`).
* Ensure `feedback_porting_fn="prev_macro_state"` is correctly mapped so the output macro-state loops back into the input port for the next timestep.
* Wrap the `ChainFactor` inside a `torx.DFG` and expose the required `micro_state` and `initial_macro_state` input ports.

### Component 3: Data Generation (The Environment)

Write a synthetic continuous-time data generator:

* **Ground Truth Macro:** A slow, clean, low-frequency sine wave representing stable biological homeostasis.
* **Observed Micro:** A high-frequency signal heavily corrupted by large-variance Gaussian noise. It should act as an entangled, messy 16-D projection of the slow macro-state. Shape: `(batch_size, seq_len, 16)`.

### Component 4: Training Loop (Stochastic Differentiable Programming)

* Define a composite loss function (`@eqx.filter_value_and_grad`):
* **Data Term:** MSE between the Torx DFG's extracted macro-state and the ground-truth clean macro-state.
* **Temporal Slowness Penalty (Optional but recommended):** Penalize the temporal derivative $\vert{}\vert{} Macro_t - Macro_{t-1} \vert{}\vert{}^2$ to mathematically force the network to ignore high-frequency micro-chatter.


* Use `optax.adamw` (lr=0.01) to train the DFG for ~300 epochs.
* Because Torx `sample()` expects unbatched dictionaries by default, use `jax.vmap` to vectorize the sequential DFG sampling across the batch dimension.

### Component 5: Evaluation & Dashboard Rendering

* After training, unroll the trained DFG on a fresh test sequence.
* Generate a dark-themed `matplotlib` dashboard (`matplotlib.use("Agg")`) with two vertically stacked subplots:
1. **Top Panel (The Micro-State Chaos):** A plot showing the chaotic, high-frequency 16-D input data over time.
2. **Bottom Panel (The Macro-State Enslavement):** The DFG's extracted 2-D macro-state superimposed over the ground-truth slow wave. This visually proves the causal graph successfully enslaved the micro-state into a stable order parameter.


* Save the plot to `output/demo/torx_observer_zero.png`. Ensure `os.makedirs` is used to prevent missing directory errors.

## 🛠️ 3. Code Scaffolding

Ensure the file follows the standard Stable Basin 2.0 structure:

```python
"""
Phase 3: Torx Observer Zero Port (Macro-State Extraction)

Demonstrates extracting a slow, stable macroscopic order parameter from chaotic 
micro-states using a Torx probabilistic Directed Factor Graph (DFG).
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import logging
import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torx import DFG, Site, ChainFactor
from torx.factor import AbstractReferenceFactor

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("TorxObserverZero")

# ... (Implementation of MacroObserverFactor, Data Generator, and Training Loop) ...

```

## ✅ 4. Definition of Done

1. **Strict Torx API Adherence:** The code must correctly inherit from `AbstractReferenceFactor` and utilize `ChainFactor` exactly as established in the existing Torx sandbox files (`dfg_basics.py`).
2. **No PyTorch:** Pure JAX/Torx/Equinox implementation.
3. **Execution:** The script must run headlessly without errors and successfully save the visualization artifact.
4. **Visual Proof:** The resulting PNG must clearly show a noisy signal being mathematically filtered/encoded into a smooth, stable macroscopic trajectory, proving the "Observer" paradigm works natively in Torx.

---

*Agent Instructions: Proceed directly to outputting the complete, self-contained, production-grade Python code for `src/demo/torx/observer_zero.py` based on the specifications above. Ensure JAX PRNG keys are split correctly at every stochastic step.*
