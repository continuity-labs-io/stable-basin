We are scrapping the unconstrained reaction-diffusion simulation. It is too chaotic to measure. I need to build a "Controlled Experimental Vessel" to establish a baseline for our Observer. We must act as a 'Creator' and explicitly define the topology of a single, idealized 2D Observer enclosed in a Markov Blanket, resting in a thermal bath.

Please write a pure PyTorch script in `src/models/vessel/vessel_baseline.py`.

**1. The Spatial Initialization (The Anatomy):**

* Create a 2D tensor grid (e.g., 200x200).
* Define a circle in the center (radius ~30 pixels) to be the `Internal State` (The Observer).
* Define a narrow ring immediately outside that circle (thickness ~5 pixels) to be the `Markov Blanket` (The Cell Wall).
* The rest of the grid is the `External World` (The Heat Bath).

**2. The Physics (Continuous-Time RD):**

* Use a standard Reaction-Diffusion update step (e.g., FitzHugh-Nagumo) with a 3x3 Laplacian convolution for spatial coupling.
* **The Crucial Difference (Spatial Heterogeneity):**
* *The Heat Bath:* Inject high Brownian noise (σdW) here at every step.
* *The Cell Wall:* Set the inhibitor variable (v) to a permanently high, rigid threshold here. This wall must act as a dampener, resisting the external noise and protecting the interior. Reduce the diffusion coefficient (D) significantly in this ring to prevent leakage.
* *The Internal State:* Keep the noise injection here near zero. Initialize the internal `u` variables to a stable, low-energy baseline. Let them resonate smoothly based only on the internal Laplacian diffusion.



**3. The Metric (Internal Entropy):**

* At every integration step, calculate the variance of the `u` tensor *strictly within the Internal State circle*.
* Store this scalar value in a list over time.

**4. The Output:**

* Use Matplotlib to create a 2-panel figure using `FuncAnimation`.
* *Left Panel:* The live 2D heatmap animation of the grid, showing the chaotic heat bath crashing against the rigid cell wall, while the interior remains calm.
* *Right Panel:* A live-updating line graph plotting the `Internal Variance (Entropy)` over time. In this baseline "Utopia" run, this line should remain flat and low.

Include a `__main__` block to run the simulation natively. Ensure all math is documented with Unicode characters (Δt, σdW).

---

You will see a glowing, perfectly insulated cell surviving in a chaotic storm, and a flat line graph proving its internal peace.

