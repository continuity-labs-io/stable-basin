Create a self-contained Python script using PyTorch and Matplotlib that
simulates a 2-Layer Hierarchical Continuous-Time State Space Model (H-SSM)
demonstrating a standing wave phase transition.

System Architecture Requirements:

1. Define a PyTorch module `HierarchicalSSM` with two continuous-time state
   layers:
   - Layer 1 (Fast/Local): State dimension d1=16. Parameterize its transition
     matrix A1 as a block skew-symmetric matrix (oscillatory) with small
     dampening.
   - Layer 2 (Slow/Macro): State dimension d2=4. Driven by a pooled projection
     of Layer 1's state.
2. Implement explicit continuous-time integration (e.g., Runge-Kutta 4th order
   or Euler integration) across a time horizon t = 0 to 10 seconds (dt = 0.005).
3. Include a delayed recursive feedback term K \* x1(t - tau) in Layer 1 to
   simulate wave reflection/self-interference.
4. Implement a top-down feedback from Layer 2 that modulates Layer 1's internal
   gain (simulating top-down prediction/expectation).
5. Expose a parameter `feedback_gain` (K). Allow running two simulations:
   - Scenario A (Driven Mode, low K): The network passively responds to an
     external noisy sinusoidal input u(t).
   - Scenario B (Standing Wave Mode, high K): The network transitions into a
     self-sustaining, phase-locked standing wave where internal recurrence
     dominates external noise.

Visualization Output: Generate a clean, publication-ready Matplotlib 2x2 subplot
figure:

- Top-Left: Layer 1 state trajectory over time (Scenario A vs Scenario B).
- Top-Right: Phase space plot (x1[0] vs x1[1]) showing the attractor limit cycle
  forming in Scenario B.
- Bottom-Left: Power Spectral Density (PSD) showing the emergence of a sharp
  resonant peak (standing wave) vs broadband noise.
- Bottom-Right: Top-down error metric over time showing Layer 2 locking onto
  Layer 1's trajectory.

Ensure code is well-commented, modular, and runnable directly from the terminal.
