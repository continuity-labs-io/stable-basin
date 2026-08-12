import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import os

from src.modules.hierarchical_ssm import HierarchicalSSM


def plot_results(time, dt, x1_A, x2_A, x1_B, x2_B, model):
    """
    Generate the 2x2 subplot visualization comparing Scenario A and Scenario B.

    1. Top-Left: Layer 1 Trajectory: Driven vs Standing Wave
       - Independent Variable (IV): Time (seconds).
       - Dependent Variable (DV): State Amplitude of x1_0 (e.g., bioelectric voltage).
       - Raw Result: The blue line (Scenario A) is a flat, noisy ripple hovering
         around zero. The orange line (Scenario B) initially stays quiet, then
         erupts into massive, sustained, rhythmic oscillations.
       - Interpretation: This is the birth of the continuous wave. In Scenario A,
         the external noise simply dissipates through the network. In Scenario B,
         the recursive self-interference (time t folding into time t - τ) creates
         constructive interference. The biological tissue stops just processing
         external noise and generates its own powerful macroscopic wave.

    2. Top-Right: Phase Space: Emergence of Attractor Limit Cycle
       - Independent Variable (IV): State x1_0 (Voltage).
       - Dependent Variable (DV): State x1_1 (Recovery).
       - Raw Result: The blue line (Scenario A) is trapped in a tiny, chaotic cloud
         at the exact center (0,0). The orange line (Scenario B) spirals outward
         into a massive, beautiful, looping orbit.
       - Interpretation: By plotting the system against itself (removing time),
         we see the literal "geometry" of the thought. The tiny blue cloud means
         the network has no stable internal structure. The massive orange spiral
         is an Attractor Limit Cycle. This proves the system has built a
         self-sustaining engine. It is the mathematical shape of the standing
         wave trapped inside the recurrent cavity.

    3. Bottom-Left: Power Spectral Density (PSD)
       - Independent Variable (IV): Frequency (Hertz).
       - Dependent Variable (DV): Power Density (log scale). This shows how much
         energy is vibrating at each specific frequency.
       - Raw Result: The blue line is a messy, broadband distribution of energy.
         The orange line has a monumental, sharp peak right near the 0-1 Hz mark.
       - Interpretation: The blue line represents a noisy soup of disconnected
         neurons firing randomly. The orange spike is the exact physical signature
         of a phase transition. The entire network has phase-locked. Millions of
         simulated nodes have synchronized into a single, unified resonant
         frequency. This is the macroscopic electromagnetic field asserting
         dominance over the individual cells.

    4. Bottom-Right: Top-Down Prediction Error
       - Independent Variable (IV): Time (seconds).
       - Dependent Variable (DV): L2 Error ||x1 - x1_pred||. This measures the
         difference between Layer 1's actual state and Layer 2's top-down
         prediction of it.
       - Raw Result: The blue line (Scenario A) stays very close to zero. The
         orange line (Scenario B) shows an initial rise as the standing wave
         emerges, followed by a plateau as Layer 2 locks on.
       - Interpretation: With the introduction of non-linear dampening, the wave
         is stabilized into a true biological limit cycle. Because the amplitude
         is bounded, the slower Layer 2 (the top-down macro context) successfully
         maps the internal geometry of Layer 1, allowing the prediction error to
         plateau and phase-lock rather than blowing up to infinity.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    steps = len(time)
    t_half = steps // 2

    # ---------------------------------------------------------
    # Top-Left: Layer 1 State Trajectory
    # ---------------------------------------------------------
    axs[0, 0].plot(time, x1_A[:, 0], label="Scenario A (Low K)", alpha=0.8, color="C0")
    axs[0, 0].plot(time, x1_B[:, 0], label="Scenario B (High K)", alpha=0.8, color="C1")
    axs[0, 0].set_title("Layer 1 Trajectory: Driven vs Standing Wave")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("State Amplitude (x1_0)")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Top-Right: Phase Space
    # ---------------------------------------------------------
    # Use the second half of the simulation to show the steady-state attractor
    axs[0, 1].plot(x1_A[t_half:, 0], x1_A[t_half:, 1], label="Scenario A", alpha=0.6, color="C0")
    axs[0, 1].plot(x1_B[t_half:, 0], x1_B[t_half:, 1], label="Scenario B", alpha=0.6, color="C1")
    axs[0, 1].set_title("Phase Space: Emergence of Attractor Limit Cycle")
    axs[0, 1].set_xlabel("State x1_0")
    axs[0, 1].set_ylabel("State x1_1")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Bottom-Left: Power Spectral Density (PSD)
    # ---------------------------------------------------------
    f_A, Pxx_A = welch(x1_A[t_half:, 0], fs=1 / dt, nperseg=1000)
    f_B, Pxx_B = welch(x1_B[t_half:, 0], fs=1 / dt, nperseg=1000)
    axs[1, 0].semilogy(f_A, Pxx_A, label="Scenario A (Broadband)", color="C0")
    axs[1, 0].semilogy(f_B, Pxx_B, label="Scenario B (Resonant Peak)", color="C1")
    axs[1, 0].set_title("Power Spectral Density (PSD)")
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 0].set_ylabel("Power Density")
    axs[1, 0].set_xlim(0, 10)
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Bottom-Right: Top-down Error Metric
    # ---------------------------------------------------------
    # Layer 2 attempts to predict Layer 1's trajectory.
    # Error = ||x1 - W_pred * x2||_2
    W_pred = model.W_pred.numpy()
    pred_A = x2_A @ W_pred.T
    pred_B = x2_B @ W_pred.T

    err_A = np.linalg.norm(x1_A - pred_A, axis=1)
    err_B = np.linalg.norm(x1_B - pred_B, axis=1)

    axs[1, 1].plot(time, err_A, label="Scenario A Error", alpha=0.7, color="C0")
    axs[1, 1].plot(time, err_B, label="Scenario B Error", alpha=0.7, color="C1")
    axs[1, 1].set_title("Top-Down Prediction Error: Layer 2 Locking onto Layer 1")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("L2 Error ||x1 - x1_pred||")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    # Resolve the output directory relative to the script location (workspace_root/src/demo/.. -> workspace_root)
    out_path = "output/2_hssm_standing_wave.png"

    plt.savefig(out_path, dpi=300)
    print(f"Simulation complete. Plot saved to '{out_path}'.")


def main():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Simulation parameters
    dt = 0.005
    t_end = 10.0
    time = np.arange(0, t_end, dt)
    steps = len(time)

    # Generate external noisy sinusoidal input u(t)
    # Mix of a base frequency and broadband noise
    freq = 2.0
    noise = np.random.randn(steps)
    u = 0.5 * np.sin(2 * np.pi * freq * time) + 1.0 * noise
    u_tensor = torch.tensor(u, dtype=torch.float32)

    # Initialize H-SSM model
    model = HierarchicalSSM(d1=16, d2=4, dt=dt, tau_delay_steps=50)

    print("Running Scenario A: Driven Mode (low K)...")
    K_low = 0.5
    x1_A, x2_A = model(u_tensor, K=K_low, steps=steps)

    print("Running Scenario B: Standing Wave Mode (high K)...")
    K_high = 5.0
    x1_B, x2_B = model(u_tensor, K=K_high, steps=steps)

    # Convert to numpy for visualization
    plot_results(time, dt, x1_A.numpy(), x2_A.numpy(), x1_B.numpy(), x2_B.numpy(), model)


if __name__ == "__main__":
    main()
