# 🛠️ ENGINEERING TICKET: Implement Phase 4 - Torx Hierarchical Enslavement

**Role Context:** Principal Software Engineer / AI Architect

**Target File:** `src/demo/torx/hierarchical_enslavement.py`

**Related Architecture Concept:** `src/echo/architecture/hierarchy.py`

## 📋 1. Background & Objective

In Phase 3, we built the "Observer Zero" to extract a slow macroscopic state from fast microscopic chaos. Now, in Phase 4, we must **close the cybernetic loop**.

According to Hermann Haken’s Synergetics and Karl Friston’s Free Energy Principle, biological systems form hierarchies where slow, macroscopic "order parameters" (the Observer) thermodynamically enslave fast, microscopic variables (the Vessel/Cells) by projecting top-down precision.

**The Goal:** Build a standalone Torx sandbox script featuring a multi-tier Directed Factor Graph (DFG). The macro-level must project a top-down prior (belief and precision) that alters the energy basin of the micro-level, physically restricting its variance and degrees of freedom to pull it into homeostasis.

## 📐 2. Technical Specifications

### Dependencies & Substrate

* **Frameworks:** Strictly JAX, Equinox (`eqx`), and Torx. **No PyTorch (`torch`) imports.**
* **Hardware Routing:** Force CPU execution to prevent XLA memory fragmentation by setting `os.environ["JAX_PLATFORMS"] = "cpu"` at the top.
* **Unified Device:** `from src.utils.device import get_optimal_device` with `backend="jax"`.
* **Logging:** Use Python's standard `logging` module to narrate the thermodynamic A/B test.

### Component 1: The Hierarchical Enslavement Factor

To avoid complex port-mapping with multi-variable temporal feedback loops, implement a joint factor `HierarchicalEnslavementFactor(torx.factor.AbstractReferenceFactor)` that computes a single timestep of the coupled system operating on a flattened state vector.

* **State Ports:**
* `input_ports`: Expects `state` (a 1D array concatenating a 2-D `macro_state` and a 10-D `micro_state`), `dt` (scalar), and `precision` (scalar coupling strength).
* `output_spec`: The updated `state` for the next timestep.


* **Parameters (Equinox Modules):**
* `W_down`: An `eqx.nn.Linear` layer that projects from Macro (2D) to Micro (10D).


* **The Physics (Top-Down Enslavement):**
Inside the `.sample()` method, define a joint energy function for the micro state:

$$E_{joint}(x_{\text{micro}}, x_{\text{macro}}) = E_{\text{local}}(x_{\text{micro}}) + \frac{\Pi}{2} \vert{}\vert{} x_{\text{micro}} - W_{\text{down}}(x_{\text{macro}}) \vert{}\vert{}^2$$



*(Where $E_{local}$ is a simple quadratic basin or multi-well potential, and $\Pi$ is the `precision` input port).*
1. Slice the input `state` into `macro` and `micro`.
2. **Micro Update:** Use `jax.grad` on $E_{joint}$ with respect to `micro` to compute the thermodynamic drift. Apply standard overdamped Langevin diffusion (Euler-Maruyama) using `jax.random.normal` and `dt`.
3. **Macro Update:** $macro_{next} = macro + dt \cdot \alpha \cdot (\text{mean}(micro) - macro) + \sigma \omega$ (A slow moving average/dampened random walk).
4. Concatenate and return the updated `state`.



### Component 2: The Torx DFG & ChainFactor

* Wrap `HierarchicalEnslavementFactor` inside a `torx.ChainFactor` to unroll it over continuous time (e.g., 500 steps).
* Map `feedback_porting_fn="state"`.
* Wrap the `ChainFactor` in a `torx.DFG` exposing `initial_state`, `dt`, and `precision`.

### Component 3: The Cybernetic A/B Test (Simulation)

We want to demonstrate the forward physics without a complex training loop.

* **Regime A (Uncoupled / Dead Tissue):** Set `precision = 0.0`. Run the DFG. The micro-states should exhibit high-variance, unbound chaos since the top-down prior is turned off.
* **Regime B (Enslaved / Living Tissue):** Set `precision = 10.0`. Run the DFG. The macro-state should slowly orbit, and the micro-states should be tightly bound to the macro-state's projection, exhibiting drastically reduced variance.

### Component 4: Evaluation & Dashboard Rendering

* Generate a dark-themed `matplotlib` dashboard (`matplotlib.use("Agg")`) with three subplots:
1. **Top Panel (The Macro Order Parameter):** Plot the 2D macro-state trajectory over time (should be smooth and slow-moving).
2. **Middle Panel (Regime A - Uncoupled Chaos):** Plot the trajectories of the 10D micro-states over time with 0 precision. Note the high variance.
3. **Bottom Panel (Regime B - Thermodynamic Enslavement):** Plot the 10D micro-states under strong top-down precision. They should be tightly clustered, tracking the macro-state's structural contour.


* Log the **Variance Reduction** (Variance of Regime A vs Variance of Regime B) to the console to quantitatively prove the thermodynamic enslavement.
* Save the plot to `output/demo/torx_hierarchical_enslavement.png`. Use `os.makedirs` to ensure the directory exists.

## 🛠️ 3. Code Scaffolding

```python
"""
Phase 4: Torx Hierarchical Enslavement

Demonstrates the cybernetic loop: A slow, macroscopic order parameter projects
top-down precision to thermodynamically enslave fast, microscopic variables,
drastically reducing their entropy/variance via Free Energy minimization.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import logging
import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torx import DFG, Site, ChainFactor
from torx.factor import AbstractReferenceFactor

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("TorxHierarchicalEnslavement")

# ... (Implementation of HierarchicalEnslavementFactor, Simulation Loop, and Plotting) ...

```

## ✅ 4. Definition of Done

1. **Mathematical Accuracy:** The code must explicitly leverage `jax.grad` on the joint Free Energy function to compute the thermodynamic drift, proving we are natively leveraging JAX's differentiable physics capabilities.
2. **Torx API:** Utilize `ChainFactor` to unroll the joint transition over time.
3. **Execution:** The script must run headlessly without errors, logging the calculated variance reduction to the console.
4. **Visual Proof:** The resulting PNG must clearly show three distinct panels, with the bottom panel visually proving that the micro-states have lost their independent degrees of freedom and are strictly enslaved by the macro order parameter.

---

*Agent Instructions: Proceed directly to outputting the complete, self-contained, production-grade Python code for `src/demo/torx/hierarchical_enslavement.py` based on the specifications above. Ensure `jax.random.split` is handled immaculately within the recursive state updates, and Equinox layers are initialized with a key.*
