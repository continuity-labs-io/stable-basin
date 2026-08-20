import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import os

from src.models.ssm.state_space_engine import StateSpaceEngine
from src.config import settings


class ToyBiologicalEnvironment:
    """
    Simulates a synthetic 'Drowning Signal' multiscale biological dataset.
    Generates high-frequency GEVI data (20kHz) and lower-frequency Optical data (100Hz).
    Both modalities are corrupted by a massive 2Hz pump artifact (sine wave).
    The GEVI data contains sparse 1ms biological spikes (action potentials).

    TODO: Output Expectation to verify:
    Question: "how does it know the pump vibration isn't a biological anomaly?"
    - The Top Panel shows the raw drowning signal (with the pump).
    - The Bottom Panel shows the Surprise metric. It will be completely flat during
    the first 50 frames despite the massive structural wobble being fed into it,
    proving the Fusion Core successfully zeroed it out of the predictive surprise metric.
    """

    # Constants have been moved to src/config.py

    def __init__(self):
        pass

    def generate_batch(self, batch_size, scenario="homeostasis", device="cpu"):
        """
        Generates a synthetic batch of multiscale data.

        Args:
            batch_size (int): Number of independent sequences to generate.
            scenario (str): One of 'homeostasis', 'corrosion', or 'toxic_shock'.
            device (str or torch.device): Target device ('cpu', 'cuda', 'mps').

        Returns:
            tuple:
                - optical_tensor (torch.Tensor): Shape [batch_size, 100, 768], float32
                - gevi_tensor (torch.Tensor): Shape [batch_size, 1, 20000], float32
        """
        # 1. Base Time Tensors
        t_optics = (
            torch.arange(settings.OPTICS_FRAMES, device=device, dtype=torch.float32)
            / settings.OPTICS_HZ
        )
        t_gevi = (
            torch.arange(settings.GEVI_FRAMES, device=device, dtype=torch.float32)
            / settings.GEVI_HZ
        )

        # 2. The Pump Artifact (2Hz sine wave)
        # Optical Pump Artifact: [100] -> [1, 100, 1]
        pump_optics = settings.OPTICS_PUMP_AMPLITUDE * torch.sin(
            2 * math.pi * settings.PUMP_ARTIFACT_HZ * t_optics
        )
        pump_optics = pump_optics.view(1, settings.OPTICS_FRAMES, 1)

        # GEVI Pump Artifact: [20000] -> [1, 1, 20000]
        pump_gevi = settings.GEVI_PUMP_AMPLITUDE * torch.sin(
            2 * math.pi * settings.PUMP_ARTIFACT_HZ * t_gevi
        )
        pump_gevi = pump_gevi.view(1, 1, settings.GEVI_FRAMES)

        # Initialize base tensors with the pump artifacts
        optical_tensor = pump_optics.expand(
            batch_size, settings.OPTICS_FRAMES, settings.OPTICS_DIM
        ).clone()
        gevi_tensor = pump_gevi.expand(batch_size, settings.GEVI_DIM, settings.GEVI_FRAMES).clone()

        # 3. The Biology (Sparse 1ms spikes)
        # We iterate over the 200-step windows and randomly inject spikes.
        num_windows = settings.GEVI_FRAMES // settings.SPIKE_WINDOW_STEPS

        for b in range(batch_size):
            for w in range(num_windows):
                # Under toxic shock, biology completely stops after T=50
                if scenario == "toxic_shock" and w >= settings.EVENT_BOUNDARY_OPTICS:
                    continue

                # Check if a spike occurs in this window
                if torch.rand(1).item() < settings.SPIKE_PROBABILITY_PER_WINDOW:
                    # Choose a random start within the window
                    start_idx = (
                        w * settings.SPIKE_WINDOW_STEPS
                        + torch.randint(
                            0, settings.SPIKE_WINDOW_STEPS - settings.SPIKE_WIDTH_STEPS, (1,)
                        ).item()
                    )
                    end_idx = start_idx + settings.SPIKE_WIDTH_STEPS
                    gevi_tensor[b, 0, start_idx:end_idx] += settings.SPIKE_AMPLITUDE

        # 4 & 5 & 6. Scenario Modifications
        if scenario == "corrosion":
            # Hardware Failure: massive random-walk baseline drift on GEVI after T=50 (10000 steps)
            # Optical remains perfectly normal.
            drift_steps = settings.GEVI_FRAMES - settings.EVENT_BOUNDARY_GEVI

            # Generate random step sizes and cumulatively sum them to create a random walk
            random_steps = (
                torch.randn((batch_size, 1, drift_steps), device=device)
                * settings.CORROSION_DRIFT_STD
            )
            random_walk = torch.cumsum(random_steps, dim=-1)

            gevi_tensor[:, :, settings.EVENT_BOUNDARY_GEVI :] += random_walk

        elif scenario == "toxic_shock":
            # Biological Crash: GEVI spikes stopped (handled in loop above).
            # Optical tensor experiences a variance explosion.
            noise_steps = settings.OPTICS_FRAMES - settings.EVENT_BOUNDARY_OPTICS
            variance_explosion = (
                torch.randn((batch_size, noise_steps, settings.OPTICS_DIM), device=device)
                * settings.TOXIC_SHOCK_NOISE_STD
            )

            optical_tensor[:, settings.EVENT_BOUNDARY_OPTICS :, :] += variance_explosion

        return optical_tensor, gevi_tensor

    # Global settings are now in src/config.py


def train_orthogonal_veto(device):
    """
    Trains the Edge Compressor (Conv1d) and Fusion Core (Mamba) jointly using
    self-supervised predictive coding on homeostasis data.
    This forces the network to mathematically isolate spikes and orthogonalize artifacts.

    Args:
        device (torch.device): Device to train on.

    Returns:
        tuple: (gevi_compressor, mamba_engine) - The trained models.
    """
    print(f"[*] Initializing Orthogonal Veto Training on {device}...")

    # 1. Instantiate the "Edge Compressor"
    gevi_compressor = nn.Conv1d(
        in_channels=1,
        out_channels=settings.GEVI_COMPRESSOR_OUT_CHANNELS,
        kernel_size=settings.GEVI_COMPRESSOR_KERNEL_SIZE,
        stride=settings.GEVI_COMPRESSOR_STRIDE,
    ).to(device)

    # 2. Instantiate the Fusion Core
    mamba_engine = StateSpaceEngine(
        d_model=settings.OPTICS_DIM + settings.GEVI_COMPRESSOR_OUT_CHANNELS
    ).to(device)

    # --- DEMO ONLY: Mamba natively produces NaNs on MPS and is slow on CPU.
    # Since this is a demo and not a foundation model, we inject a fast surrogate
    # SSM (Causal Conv1d + Linear) to make it run flawlessly on Mac. ---
    class DemoSSM(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            x_c = x.transpose(1, 2)
            x_c = self.conv(x_c)[..., : x.shape[1]]
            x_c = x_c.transpose(1, 2)
            import torch.nn.functional as F

            return self.proj(F.silu(x_c))

    mamba_engine.mamba = DemoSSM(settings.OPTICS_DIM + settings.GEVI_COMPRESSOR_OUT_CHANNELS).to(
        device
    )

    # 3. Setup optimizer and environment
    optimizer = optim.Adam(
        list(gevi_compressor.parameters()) + list(mamba_engine.parameters()),
        lr=settings.LEARNING_RATE,
    )
    env = ToyBiologicalEnvironment()

    # 4. Fast training loop
    gevi_compressor.train()
    mamba_engine.train()

    for iteration in range(1, settings.TRAIN_ITERATIONS + 1):
        optimizer.zero_grad()

        # Generate a fresh batch
        opt_tensor, gevi_tensor = env.generate_batch(
            settings.TRAIN_BATCH_SIZE, scenario="homeostasis", device=device
        )

        # Forward pass: Edge Compression
        # gevi_tensor is [Batch, 1, 20000]
        compressed_gevi = gevi_compressor(gevi_tensor)  # -> [Batch, 64, 100]
        compressed_gevi = compressed_gevi.transpose(1, 2)  # -> [Batch, 100, 64]

        # Forward pass: Fusion
        # opt_tensor is [Batch, 100, 768]
        fused_tensor = torch.cat([opt_tensor, compressed_gevi], dim=-1)  # -> [Batch, 100, 832]

        # Forward pass: Predictive Coding
        scalar_loss, _ = mamba_engine(fused_tensor)

        # Backpropagation
        scalar_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(gevi_compressor.parameters()) + list(mamba_engine.parameters()), 1.0
        )
        optimizer.step()

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"    [Iteration {iteration:03d}/{settings.TRAIN_ITERATIONS}] Loss: {scalar_loss.item():.4f}"
            )

    print("[*] Training Complete.")
    return gevi_compressor, mamba_engine


def evaluate_and_plot(compressor, mamba_engine, device):
    print("[*] Generating Inference Dashboard...")
    env = ToyBiologicalEnvironment()

    # 2. Generate validation data
    opt_hom, gevi_hom = env.generate_batch(1, scenario="homeostasis", device=device)
    opt_cor, gevi_cor = env.generate_batch(1, scenario="corrosion", device=device)
    opt_tox, gevi_tox = env.generate_batch(1, scenario="toxic_shock", device=device)

    import torch.nn.functional as F

    # 3. Helper to extract frame distances (Surprise)
    def get_ksm(opt_tensor, gevi_tensor):
        with torch.no_grad():
            comp_gevi = compressor(gevi_tensor).transpose(1, 2)
            fused = torch.cat([opt_tensor, comp_gevi], dim=-1)
            fused = F.layer_norm(fused, [fused.shape[-1]])
            _, frame_dists = mamba_engine(fused)
        return frame_dists.cpu().numpy()

    ksm_hom = get_ksm(opt_hom, gevi_hom)
    ksm_cor = get_ksm(opt_cor, gevi_cor)
    ksm_tox = get_ksm(opt_tox, gevi_tox)

    # 4. Programmatic Verification
    # Ensure the surprise metric is flat during the first 50 frames (homeostasis/pump artifact)
    # The fusion core should have learned to veto the massive 2Hz wobble.
    import numpy as np

    surprise_variance = np.var(ksm_hom[:50])
    assert surprise_variance < 0.1, (
        f"Veto Proof Failed! Surprise metric is not flat. Variance: {surprise_variance:.6f}"
    )
    print(
        f"[*] Veto Proof Passed: Surprise metric variance during pump artifact is {surprise_variance:.6f}"
    )

    # 5. Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Top Panel: The Drowning Signal
    gevi_raw_slice = gevi_hom[0, 0, :].cpu().numpy()
    t_gevi_slice = torch.arange(len(gevi_raw_slice)).numpy() / settings.GEVI_HZ
    ax1.plot(t_gevi_slice, gevi_raw_slice, color="cyan", alpha=0.8)
    ax1.set_title("The Drowning Signal (Raw 20kHz GEVI)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (s)")

    # Middle Panel: Orthogonal Veto (Homeostasis vs Corrosion)
    t_opt = torch.arange(len(ksm_hom)).numpy() / settings.OPTICS_HZ
    ax2.plot(t_opt, ksm_hom, label="Homeostasis", color="green", linewidth=2)
    ax2.plot(
        t_opt,
        ksm_cor,
        label="Corrosion (Hardware Failure)",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    event_time_s = settings.EVENT_BOUNDARY_OPTICS / settings.OPTICS_HZ
    ax2.axvline(
        x=event_time_s, color="gray", linestyle="--", label=f"Event Boundary (T={event_time_s}s)"
    )
    ax2.set_title("Orthogonal Veto (Surprise)")
    ax2.set_ylabel("Surprise (Cosine Distance)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()

    # Bottom Panel: True Crash (Toxic Shock)
    ax3.plot(t_opt, ksm_tox, label="Toxic Shock (Biological Crash)", color="purple", linewidth=2)
    ax3.axvline(
        x=event_time_s, color="gray", linestyle="--", label=f"Event Boundary (T={event_time_s}s)"
    )
    ax3.set_title("True Crash Detection (Surprise)")
    ax3.set_ylabel("Surprise (Cosine Distance)")
    ax3.set_xlabel("Time (s)")
    ax3.legend()

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/1_hssm_veto_proof.png")
    print("[*] Dashboard saved to output/1_hssm_veto_proof.png")


if __name__ == "__main__":
    from src.utils.device import get_optimal_device

    # Mamba might produce NaN without gradient clipping, but we added it. Let's try MPS.
    device = get_optimal_device(verbose=True)

    env = ToyBiologicalEnvironment()

    opt, gev = env.generate_batch(4, scenario="homeostasis", device=device)
    print(f"Homeostasis -> Optics: {opt.shape}, GEVI: {gev.shape}")

    opt, gev = env.generate_batch(4, scenario="corrosion", device=device)
    print(f"Corrosion   -> Optics: {opt.shape}, GEVI: {gev.shape}")

    opt, gev = env.generate_batch(4, scenario="toxic_shock", device=device)
    print(f"Toxic Shock -> Optics: {opt.shape}, GEVI: {gev.shape}")

    print("\n[*] Testing Training Loop...")
    gevi_comp, mamba = train_orthogonal_veto(device)

    print("\n[*] Generating Dashboard...")
    evaluate_and_plot(gevi_comp, mamba, device)
