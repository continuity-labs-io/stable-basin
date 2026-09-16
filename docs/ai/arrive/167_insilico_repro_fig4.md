**Context Files to Load / Create:**
* `src/echo/models/primitives/ebm.py`
* `src/echo/models/observer.py` (Specifically the continuous-time physics step / SDE)
* `src/data/behavior/celegans_gait_dataset.py`
* `src/echo/benchmarks/07_insilico_reprogramming.py` (Create)

**Task: Paper Figure 4 - In-Silico Reprogramming & Thermodynamic Rescue**
Please write the simulation script `src/echo/benchmarks/07_insilico_reprogramming.py`. This script serves as the scientific climax of the project. It demonstrates mathematically that an "old" biological trajectory can be rescued by applying a synthetic control force (Precision Injection) that artificially steepens the Waddington basin, forcing the erratic system back into a youthful limit cycle.

**Core Objectives:**

**1. The Setup (Loading the Youthful Engine):**
* Instantiate the `PredictiveCodingGraph` (utilizing the `PrecisionWeightedEBM`, dissipative $\Gamma$, and solenoidal $Q$) that theoretically represents the fully trained "Young Worm" (Days 1-3) physics engine.
* Extract a single initial state $x_0$ from the "Old Worm" (Day 9+) dataset (or use the high-noise synthetic fallback). This erratic state serves as our pathological starting point.

**2. The Rejuvenation Intervention (Precision Injection SDE):**
* Implement a continuous-time forward simulation function using `jax.lax.scan` to perform Euler-Maruyama integration of the core SDE.
* **The Normal Physics:** $dx = -(Q - \Gamma) \nabla E_\theta(x) dt + \sqrt{2\Gamma T} dW$
* **The Intervention:** Introduce a scalar `precision_injection_gain` ($\lambda$). Modify the energy gradient in the rollout function to evaluate $\nabla (\lambda E_\theta(x))$. 
* *The Physics Meaning:* Multiplying the learned energy landscape by $\lambda > 1$ mathematically mimics a restorative bioelectric injection that artificially restores top-down precision weighting to the aging tissue, steepening the well.
* **Run A (Degraded/Aged Baseline):** Simulate the SDE forward for $N=1000$ steps from $x_0$ with $\lambda = 0.2$ (simulating a flattened, low-precision aged state).
* **Run B (The Rescue):** Simulate the SDE forward for $N=1000$ steps from the *exact same* $x_0$, but apply a therapeutic intervention $\lambda = 5.0$ (artificially re-steepening the youthful basin).

**3. Figure 4 Visual Generation (The Beacon Plot):**
* Generate a cleanly formatted `matplotlib` figure (`outputs/benchmarks/fig4_insilico_rescue.png`) with three distinct panels in a 1x3 layout.
* **Panel A: The Pathology (Degraded Simulation).** Plot the first 3 dimensions of the state vector (3D phase space) for Run A ($\lambda = 0.2$). Visually demonstrate a decaying, erratic orbit that fails to maintain a limit cycle.
* **Panel B: The Phase Space Rescue (3D).** Plot Run B ($\lambda = 5.0$). This must visually demonstrate the trajectory starting at the erratic old state but rapidly snapping tightly back into a healthy, spinning biological limit cycle.
* **Panel C: Thermodynamic Restoration (Hessian Trace).** Compute and plot the trace of the Hessian matrix ($\nabla^2 E$) over the simulated timesteps for both Run A and Run B. This must visually prove that the synthetic control force successfully pushed the state from a "flat" (low trace) region back into a "steep" (high trace) basin.

**Constraints:**
* **JAX Unrolling:** You must strictly use `jax.lax.scan` for the Euler-Maruyama rollout. Do not use Python `for` loops to iterate the SDE, as this will destroy XLA compilation speed.
* **Determinism:** Use strict random seeding (`jax.random.PRNGKey(42)`) for the Euler-Maruyama noise $dW$ to ensure the exact same spiral is generated on every CI run.
* **Offline Plotting:** Ensure `matplotlib` does not block execution (`plt.show()`). Use `plt.savefig()` to save the output, creating the `outputs/benchmarks/` directory if it does not exist, and immediately close the figure.
* **Logging:** Use the standard Python `logging` module. Emit peaceful, precise logs (e.g., `logger.info("Simulating in-silico rescue with precision_injection_gain=5.0.")`). No dramatic, capitalized, or emoji-laden prints.

Note to the human. Add figures to this doc: https://docs.google.com/document/d/1OJrmHQQuP6G5S_p6qT11zmFLuVOweEAl8NCKIXTW8yI/edit?tab=t.0
