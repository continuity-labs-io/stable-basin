### Test 1: The Temporal Shatter Test

*Goal: Prove why continuous-time is required. We will incrementally break the physics engine by increasing the integration step size ($\Delta t$). We expect to find the exact critical latency threshold where the differential equations destabilize, causing the Markov Blanket to physically fracture and the internal entropy to spike.*

**File paths to feed into context:**
`src/models/vessel/vessel_baseline.py` (Read-only reference)
`tests/models/vessel/test_temporal_shatter.py` (Create this file)

**Raw text prompt to execute:**
Please create `tests/models/vessel/test_temporal_shatter.py` by cloning the core logic from `vessel_baseline.py`. We are running the Temporal Shatter Test to find the critical latency threshold of the Markov Blanket.

Modify the baseline code with the following changes:

1. **The Perturbation:** Instead of a constant, safe integration step (e.g., Δt = 0.01), implement a dynamic Δt that slowly and linearly increases as the simulation runs (e.g., from 0.01 up to 0.5 over 1000 frames). This simulates the effect of discrete "time-smearing" or high-latency OS scheduling jitter.
2. **The Physics Response:** As Δt increases, the Laplacian convolution will eventually violate its numerical stability condition. The rigid Cell Wall will mathematically fracture, allowing the external Heat Bath noise to flood the Internal State.
3. **The Output Metric:** Modify the Matplotlib animation to have 2 panels. The left panel shows the live 2D heatmap. The right panel must plot two synchronized line graphs over time:
* Top graph: The increasing value of Δt.
* Bottom graph: The Internal Variance (Entropy) of the Observer.
The visual output must clearly demonstrate the exact moment Δt crosses the critical threshold, resulting in a catastrophic spike in Internal Entropy. Include a `__main__` block to run the simulation natively.
