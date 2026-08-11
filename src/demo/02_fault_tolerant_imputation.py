"""
Demo 02: Indestructible Edge

This script demonstrates the fault-tolerance and self-healing routing capabilities of the network.
It simulates catastrophic hardware failures (e.g., sensor dropouts) and shows how the architecture
seamlessly maintains representation and inference despite severe input corruption.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("IndestructibleEdge")


from src.models.ssm.meld_engine import MeldEngine


# --- Data Generator ---
class WetLabDisasterSimulator:
    def __init__(self, seq_len=200, input_dim=114):
        self.seq_len = seq_len
        self.input_dim = input_dim
        # Generate random mixing matrix
        self.mixing_matrix = torch.randn(5, input_dim)

    def generate_batch(self, batch_size, scenario="training", device="cpu"):
        t = (
            torch.arange(self.seq_len, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        freqs = torch.tensor([0.5, 1.2, 1.8, 2.5, 3.0], device=device).view(1, 1, 5)
        # Base biology [batch, time, 5]
        biology_signals = torch.sin(2 * np.pi * freqs * (t / 100.0))
        # Mix to 114 dimensions [batch, time, 114]
        clean_tensor = torch.matmul(biology_signals, self.mixing_matrix.to(device))
        clean_tensor = clean_tensor.expand(batch_size, -1, -1).clone()
        clean_tensor += torch.randn(batch_size, self.seq_len, self.input_dim, device=device) * 0.1

        corrupt_tensor = clean_tensor.clone()
        # The Pump Artifact: massive 2Hz sine wave, amplitude=5.0
        pump_artifact = 5.0 * torch.sin(2 * np.pi * 2.0 * (t / 100.0))
        corrupt_tensor += pump_artifact

        if scenario == "training":
            # Randomly drop 15% of sensors for the whole sequence
            drop_mask = torch.rand(self.input_dim, device=device) < 0.15
            corrupt_tensor[:, :, drop_mask] = float("nan")
        elif scenario == "disaster":
            # At DROP_FRAME = 100, permanently kill sensors 90-110
            corrupt_tensor[:, 100:, 90:110] = float("nan")

        return corrupt_tensor, clean_tensor


# --- Dashboard ---
def plot_indestructible_dashboard(corrupt_seq, true_seq, pred_seq, drop_frame, output_dir):
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Panel 1: Input
    c_seq_np = corrupt_seq.cpu().numpy()
    im1 = ax1.imshow(c_seq_np.T, aspect="auto", cmap="viridis", origin="lower")
    ax1.axvline(
        x=drop_frame,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Catastrophic Sensor Dropout (NaN)",
    )
    ax1.set_title(
        "Analog Biological Layer (Massive Pump Artifact + Catastrophic Sensor Dropout)",
        color="white",
        fontweight="bold",
    )
    ax1.set_ylabel("Features")
    ax1.legend()
    fig.colorbar(im1, ax=ax1)

    # Panel 2: Imputation & Veto (focus on sensor 100)
    target_sensor = 100
    t_full = np.arange(len(true_seq))

    true_signal = true_seq[:, target_sensor].cpu().numpy()
    pred_signal = pred_seq[:, target_sensor].cpu().numpy()

    ax2.plot(t_full, true_signal, color="cyan", label="Clean Ground Truth Biology", linewidth=2)
    ax2.plot(
        t_full,
        pred_signal,
        color="orange",
        linestyle="--",
        label="Model Imputed Prediction",
        linewidth=2,
    )
    ax2.axvline(x=drop_frame, color="red", linestyle="--", linewidth=2)
    ax2.set_title(
        "State-Space Recovery: Artifact Vetoed & Missing Sensor Imputed",
        color="white",
        fontweight="bold",
    )
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")

    # Panel 3: Error
    abs_error = np.abs(pred_signal - true_signal)
    ax3.plot(t_full, abs_error, color="magenta", linewidth=2, label="Absolute Error")
    ax3.axvline(x=drop_frame, color="red", linestyle="--", linewidth=2)
    ax3.axhline(
        y=0.1, color="lime", linestyle=":", linewidth=2, label="Theoretical Biological Noise Floor"
    )
    ax3.set_title("Imputation Error (Absolute Difference)", color="white", fontweight="bold")
    ax3.set_ylabel("Error")
    ax3.set_xlabel("Time Step")
    ax3.legend(loc="upper right")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "02_indestructible_edge.png")
    plt.savefig(output_path, dpi=300)
    logger.info(f"[*] Dashboard saved to {output_path}")


def main():
    device = get_optimal_device(allow_mps=False, verbose=True)
    # MPS can sometimes have issues with NaNs/backward in these specific ops. Safer on CPU if forced.
    device = torch.device("cpu") if device.type == "mps" else device

    print("\n[*] BOOTING DEMO 2: THE INDESTRUCTIBLE EDGE")

    engine = MeldEngine(input_dim=114, d_model=256, mask_aware=True).to(device)
    simulator = WetLabDisasterSimulator(seq_len=200, input_dim=114)
    optimizer = optim.AdamW(engine.parameters(), lr=1e-3)

    logger.info("[*] Starting Burn-In (Training) to learn spatial covariance and veto noise...")
    engine.train()
    for i in range(30):
        corrupt_batch, clean_batch = simulator.generate_batch(8, scenario="training", device=device)
        optimizer.zero_grad()

        preds, _ = engine(corrupt_batch[:, :-1, :])
        targets = clean_batch[:, 1:, :]

        loss = F.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            logger.info(f"    Iteration {i + 1}/30 - Loss: {loss.item():.4f}")

    logger.info("[*] Simulating catastrophic sensor failure (NaN)...")
    engine.eval()
    with torch.no_grad():
        logger.info("[*] Injecting 2Hz microfluidic pump vibration...")
        corrupt_test, clean_test = simulator.generate_batch(1, scenario="disaster", device=device)

        preds, _ = engine(corrupt_test)

        output_dir = "output/demo"
    os.makedirs(output_dir, exist_ok=True)

    plot_indestructible_dashboard(corrupt_test[0], clean_test[0], preds[0], 100, output_dir)
    logger.info("[+] Demo 2 Complete.")


if __name__ == "__main__":
    main()
