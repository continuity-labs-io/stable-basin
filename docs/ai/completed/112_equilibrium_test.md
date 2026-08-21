### Test 3: The Equilibrium Test (The "Wound")

*Goal: Prove the principle of Active Coherence Maintenance. We will strike the Markov Blanket with a massive energy spike (a wound) and mathematically verify that the system is capable of absorbing the shock, repairing the wall, and restoring its internal baseline entropy.*

**File paths to feed into context:**
`src/models/vessel/vessel_baseline.py` (Read-only reference)
`tests/models/vessel/test_equilibrium.py` (Create this file)

**Raw text prompt to execute:**
Please create `tests/models/vessel/test_equilibrium.py` by cloning the core logic from `vessel_baseline.py`. We are running the Active Equilibrium Test to mathematically prove the system's ability to repair its Markov Blanket after a catastrophic external strike.

Modify the baseline code with the following changes:

1. **The Perturbation (The Wound):** Programmatically inject a massive, localized energy spike into the grid at a specific time (e.g., frame 300). The spike must physically intersect the Cell Wall (Markov Blanket) and breach slightly into the Internal State. Set the `u` values in this 10x10 pixel blast radius to an extreme high (e.g., 5.0).
2. **The Physics Response:** The 2D heatmap will show the boundary wall violently rupturing. However, because the baseline parameters are tuned for stability, the Laplacian diffusion and the rigid inhibitor parameters of the Cell Wall must dynamically re-seal the breach over the next 200 frames.
3. **The Output Metric:** The right-hand panel must plot the Internal Variance (Entropy) over time. Mark the exact frame of the "Wound Injection" with a vertical red line. The graph must show the initial baseline, a massive chaotic spike in internal entropy immediately following the wound, and the crucial exponential decay curve as the system actively repairs itself and returns to the stable baseline. Include a `__main__` block to run the simulation natively.
