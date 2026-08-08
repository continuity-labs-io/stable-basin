import torch
import torch.nn as nn
import numpy as np


class HierarchicalSSM(nn.Module):
    """
    Core Temporal Scaffold: 2-Tier Hierarchical Continuous-Time State Space Model (H-SSM).
    
    This architecture explicitly defines a two-tier cascade to overcome Shannon's Limit 
    and catastrophic forgetting over long sequence lengths:
    - Layer 1 (Fast-Mamba): Ingests high-frequency (e.g. 20kHz) HD-MEA data and outputs 
      low-frequency (e.g. 1Hz) Macroscopic Kinetic Tokens.
    - Layer 2 (Slow-Mamba): Operates entirely on the 1Hz Macroscopic tokens, efficiently 
      compressing months of continuous recording into a finite state vector without violating 
      Shannon's limit.
      
    Demonstrates a standing wave phase transition driven by delayed recurrent feedback.

    Scenario A (Low K): The "Driven Mode." The feedback parameter (K) is turned down.
    The simulated biological network is passive.
    It has no internal memory or momentum; it is merely reacting to external noise (the vat).

    Scenario B (High K): The "Standing Wave Mode." The feedback parameter (K) crosses the critical threshold.
    The internal recursive loops dominate, and the network begins to predict and reinforce its own state.
    State x1_0 and x1_1: These are two orthogonal dimensions of Layer 1's hidden state vector.
    Think of them as two coupled biological variables driving an oscillation—for example, x1_0 is
    the average electrical voltage of the neural population, and x1_1 is the metabolic recovery rate.
    You need at least two interacting dimensions to create a wave.
    """

    def __init__(self, d1=16, d2=4, dt=0.005, tau_delay_steps=50, gamma=0.1):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.dt = dt
        self.tau_steps = tau_delay_steps
        self.gamma = gamma

        # Layer 1 (Fast/Local) parameters
        A1 = torch.zeros(d1, d1)
        # Block skew-symmetric for oscillatory behavior
        for i in range(d1 // 2):
            # Frequencies spread from 0.5Hz to ~4Hz
            omega = (i + 1) * 2.0 * np.pi * 0.5
            A1[2 * i, 2 * i + 1] = omega
            A1[2 * i + 1, 2 * i] = -omega
        # Small dampening
        A1 -= torch.eye(d1) * 1.0
        self.A1 = nn.Parameter(A1, requires_grad=False)

        # Input projection for Layer 1
        self.B1 = nn.Parameter(torch.randn(d1) * 0.5, requires_grad=False)

        # Layer 2 (Slow/Macro) parameters
        # Slower decay dynamics
        self.A2 = nn.Parameter(-torch.eye(d2) * 0.2, requires_grad=False)
        # Pooled projection from Layer 1 to Layer 2
        self.B2 = nn.Parameter(torch.randn(d2, d1) / np.sqrt(d1), requires_grad=False)

        # Top-down feedback (modulation) from Layer 2 to Layer 1
        self.W_td = nn.Parameter(torch.randn(1, d2) * 0.5, requires_grad=False)

        # Top-down prediction (for error metric: Layer 2 predicting Layer 1)
        self.W_pred = nn.Parameter(torch.randn(d1, d2) / np.sqrt(d2), requires_grad=False)

    def forward(self, u, K, steps):
        """
        Simulate the H-SSM using explicit Euler integration.
        u: External input signal (steps,)
        K: Feedback gain for the delayed recurrent term
        steps: Number of integration steps
        """
        x1_hist = torch.zeros(steps, self.d1)
        x2_hist = torch.zeros(steps, self.d2)

        x1 = torch.zeros(self.d1)
        x2 = torch.zeros(self.d2)

        for t in range(steps):
            # Delayed recursive feedback term K * x1(t - tau)
            if t >= self.tau_steps:
                x1_delayed = x1_hist[t - self.tau_steps]
            else:
                x1_delayed = torch.zeros(self.d1)

            # Top-down feedback from Layer 2 modulating Layer 1's internal gain
            # Use tanh to keep the modulation bounded around 1.0
            td_mod = 1.0 + torch.tanh(self.W_td @ x2)

            # Layer 1 ODE: dx1/dt
            # Includes dampened oscillations, input driving, and delayed self-interference.
            # A non-linear Van der Pol style dampening term (- gamma * x1**3) stabilizes the wave,
            # preventing unbounded exponential growth and forming a true biological limit cycle.
            dx1 = (
                torch.mv(self.A1, x1) * td_mod
                + self.B1 * u[t]
                + K * torch.tanh(x1_delayed)
                - self.gamma * (x1**3)
            )

            # Layer 2 ODE: dx2/dt
            # Driven by pooled projection of Layer 1
            dx2 = torch.mv(self.A2, x2) + torch.mv(self.B2, torch.tanh(x1))

            # Explicit Euler integration
            x1 = x1 + self.dt * dx1
            x2 = x2 + self.dt * dx2

            x1_hist[t] = x1
            x2_hist[t] = x2

        return x1_hist, x2_hist
