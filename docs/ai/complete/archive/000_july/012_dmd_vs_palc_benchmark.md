Create a new benchmark script: `src/metrics/bifurcation_benchmark.py`. The goal
is to quantitatively benchmark the latency and accuracy tradeoff between our
sliding-window Dynamic Mode Decomposition (DMD/SVD) and the exact Autograd
Jacobian (the computational bottleneck of PALC).

Requirements:

1. **Imports & Setup:**

- Import `torch`, `time`.
- Import `StateSpaceEngine` from `src.models.state_space_engine`.
- Import `ThermodynamicMetrics` from `src.metrics.thermodynamics`.
- Initialize
  `device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`.
- Initialize `model = StateSpaceEngine(d_model=832).to(device).eval()`.

2. **The Accuracy Benchmark (Temporal Lag):**

- Create a dummy sequence `z_seq = torch.randn(100, 832, device=device) * 0.1`.
- At exactly frame 50, inject a catastrophic variance explosion (multiply frames
  50-100 by an exponentially increasing scalar) to simulate a structural
  Waddington crash.
- Run `ThermodynamicMetrics(alpha=500.0).calculate_ksm(z_seq, window_size=4)`.
- Identify the exact frame the KSM metric drops below 0.9.
- Note: DMD will naturally have a slight temporal lag (1-3 frames) because it
  calculates the approximation over a sliding window. PALC operates on
  single-point instantaneous topology, making it frame-perfect. We will log this
  tradeoff.

3. **Algorithm 1: DMD Speed Benchmark**

- Loop over 100 dummy sliding windows of shape `[4, 832]`.
- Implement the exact SVD block from `calculate_ksm` (U, S, Vh, A_tilde,
  eigvals).
- Measure the execution time of ONLY the math block per step using
  `time.perf_counter()`.
- Return the average time per step (ms).

4. **Algorithm 2: Exact Jacobian (The PALC Bottleneck) Speed Benchmark**

- PALC requires finding the roots of the non-linear vector field. To do a
  Newton-Raphson step, we need the exact Jacobian of the neural network.
- Define a wrapper function:
  `def step_fn(z): return model.forward_predictor(model.mamba(z))`
- Initialize `z_t = torch.randn(1, 1, 832, device=device)`.
- Loop over 100 steps.
- For each step, run: a. `J = torch.autograd.functional.jacobian(step_fn, z_t)`.
  Reshape it to a dense `[832, 832]` matrix. b. Simulate the predictor-corrector
  matrix inversion:
  `J_inv = torch.linalg.inv(J + torch.eye(832, device=device) * 1e-4)`.
- Measure the execution time of ONLY the Jacobian + Linear Solve block using
  `time.perf_counter()`.
- Return the average time per step (ms).

5. **Execution & Formatted Output:**

- Write a `__main__` block.
- Run a 5-step warmup loop for both algorithms to compile hardware graphs.
- Run both benchmarks.
- Print a clean ASCII table comparing:
  - Algorithm Name
  - Average Latency per step (ms)
  - Estimated Max FPS
  - Topological Accuracy (Sliding Window Lag vs. Instantaneous Frame-Perfect)
  - Edge-Compute Viability for 100Hz live-streaming (Yes/No)
