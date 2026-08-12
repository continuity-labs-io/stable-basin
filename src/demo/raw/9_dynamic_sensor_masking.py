"""
BDC-RFC-004: Dynamic Sensor Masking Demo

This script demonstrates the training of a Continuous-Time State Space Engine
(using a Mamba-2 backbone) to handle real-time catastrophic sensor failures.

KEY STUDY FINDING:
Dynamic sensor imputation relies entirely on the model learning deep spatial
covariance. Our study showed that training on simple "salt-and-pepper" noise
is insufficient; the network cannot zero-shot a catastrophic hardware failure.
To successfully impute a completely severed sensor, the training curriculum
MUST utilize "block masking" (dropping entire sensor channels for the duration
of the sequence). This forces the network to rely on spatial correlation rather
than temporal memorization.
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
from tqdm import tqdm

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

from src.data.sim2real.neocortical_assembloid_dataloader import NeocorticalAssembloidDataset
from src.utils.device import get_optimal_device


class DynamicMaskingEngine(nn.Module):
    """
    A Continuous-Time State Space Engine that dynamically routes around
    hardware sensor failures (NaNs) in real-time.
    """

    def __init__(self, input_dim=114, d_model=256, d_state=64):
        super().__init__()
        self.input_dim = input_dim

        # The input dimension is doubled (114 data + 114 mask)
        # This explicit bottleneck forces the surviving sensors to compensate.
        self.mask_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Mamba-2 Backbone
        if Mamba2 is not None:
            self.mamba_block1 = Mamba2(d_model=d_model, d_state=d_state)
            self.mamba_block2 = Mamba2(d_model=d_model, d_state=d_state)
        else:
            self.mamba_block1 = nn.Identity()
            self.mamba_block2 = nn.Identity()

        # Forward Predictor
        self.predictor = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        x: [Batch, Time, Features] containing NaNs where hardware dropped.
        """
        # 1. Detect missing sensors (1.0 if dead/NaN, 0.0 if healthy)
        mask = torch.isnan(x).float()

        # 2. Sanitize the input (Zero-fill NaNs so PyTorch math doesn't explode)
        x_safe = torch.nan_to_num(x, nan=0.0)

        # 3. The Crucial Step: Concatenate Data + Mask
        # Shape becomes [Batch, Time, input_dim * 2]
        x_combined = torch.cat([x_safe, mask], dim=-1)

        # 4. Project into the continuous latent space
        h = self.mask_encoder(x_combined)

        # 5. Continuous-time sequence modeling
        h = self.mamba_block1(h)
        h = self.mamba_block2(h)

        # 6. Predict the true biological state (even the missing parts)
        preds = self.predictor(h)

        return preds, mask


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    device = get_optimal_device(allow_mps=False)
    print(f"\n[*] Device: {device.type.upper()}")

    INPUT_DIM = 114
    SEQUENCE_LENGTH = 200
    TRAIN_EPOCHS = 1000

    dataset = NeocorticalAssembloidDataset(time_steps=SEQUENCE_LENGTH, latent_dim=INPUT_DIM)
    dataset_iter = iter(dataset)

    engine = DynamicMaskingEngine(input_dim=INPUT_DIM, d_model=256).to(device)
    optimizer = optim.AdamW(engine.parameters(), lr=1e-3)

    # --- 1. BURN-IN TRAINING (Learning the Spatial Covariance) ---
    print("[*] Training network to learn multi-modal covariance (Burn-In)...")
    engine.train()

    # Early Stopping State
    best_loss = float("inf")
    patience_counter = 0
    PATIENCE_LIMIT = 50

    pbar = tqdm(range(TRAIN_EPOCHS), desc="Burn-In Training", unit="iter")
    for iteration in pbar:
        seq, _ = next(dataset_iter)
        seq = seq.unsqueeze(0).to(device)  # Clean ground truth [1, SEQUENCE_LENGTH, INPUT_DIM]

        # CURRICULUM UPGRADE: Dynamic Mid-Sequence Sensor Failure
        # We simulate sensors suddenly dying partway through the recording (matching the disaster scenario)
        seq_corrupt = seq.clone()

        # Pick a random time for the failure to occur (e.g. between step 10 and 190)
        fail_t = torch.randint(10, SEQUENCE_LENGTH - 10, (1,)).item()

        # Pick a random 15% of sensors to suffer catastrophic failure
        failed_sensors = torch.rand(INPUT_DIM, device=device) < 0.15

        # Inject the NaNs from the failure point to the end of the sequence
        seq_corrupt[0, fail_t:, failed_sensors] = float("nan")

        optimizer.zero_grad()

        # Forward pass predicting T+1
        pred_t_plus_1, _ = engine(seq_corrupt[:, :-1, :])
        target_t_plus_1 = seq[:, 1:, :]  # Predict against the UNCORRUPTED ground truth

        loss = F.mse_loss(pred_t_plus_1, target_t_plus_1)
        loss.backward()

        # Calculate the total gradient norm before the optimizer steps
        # We set max_norm to infinity so it just measures the norm without clipping it
        grad_norm = torch.nn.utils.clip_grad_norm_(engine.parameters(), max_norm=float("inf"))

        optimizer.step()

        current_loss = loss.item()
        current_grad_norm = grad_norm.item()

        # Update progress bar metrics
        pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Grad Norm": f"{current_grad_norm:.4f}"})

        # 1. Early Stopping on Gradient Collapse
        if current_grad_norm < 1e-4:
            pbar.write(
                f"[*] Early Stopping Triggered: Gradients vanished at iteration {iteration + 1}"
            )
            break

        # 2. Early Stopping on Loss Plateau
        if current_loss < (best_loss - 0.0005):
            best_loss = current_loss
            patience_counter = 0  # Reset patience if we improved
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE_LIMIT:
            pbar.write(
                f"[*] Early Stopping Triggered: Loss plateaued for {PATIENCE_LIMIT} iterations (Stopped at {iteration + 1})"
            )
            break

    pbar.close()

    # --- 2. THE WET-LAB DISASTER SIMULATION ---
    print("\n[*] Simulating catastrophic sensor failure (Voltage Array disconnected)...")
    engine.eval()

    true_seq, _ = next(dataset_iter)
    true_seq = true_seq.unsqueeze(0).to(device)

    test_seq = true_seq.clone()

    # Inject NaNs into the last two features halfway through the sequence (Omega_VoltRed, Omega_VoltGrn)
    DROP_FRAME = SEQUENCE_LENGTH // 2
    # The last two features are Omega_VoltRed (112) and Omega_VoltGrn (113)
    FAILED_INDEX = 112
    test_seq[:, DROP_FRAME:, FAILED_INDEX:] = float("nan")

    with torch.no_grad():
        # Mask encoder intercepts the NaNs and infers the voltage based on the optical shape (Features 0-111)
        pred_seq, _ = engine(test_seq[:, :-1, :])

        # Calculate the error between the mask encoder's guess and the ground truth we hid
        imputed_voltage = pred_seq[0, DROP_FRAME - 1 :, FAILED_INDEX:].cpu().numpy()
        true_voltage = true_seq[0, DROP_FRAME:, FAILED_INDEX:].cpu().numpy()

        # Compute and log the precise delta (MSE and MAE)
        mse = np.mean((imputed_voltage - true_voltage) ** 2)
        mae = np.mean(np.abs(imputed_voltage - true_voltage))
        print(f"[*] Imputation Performance - MSE: {mse:.4f} | MAE: {mae:.4f}")

    print("[*] Generating Dashboard...")
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Panel 1: What the Mamba Engine Received
    im1 = ax1.imshow(test_seq[0].T.cpu().numpy(), aspect="auto", cmap="viridis", origin="lower")
    ax1.axvline(
        x=DROP_FRAME, color="red", linestyle="--", linewidth=2, label="Sensors Dropped (NaN)"
    )
    ax1.set_title("114-D Tensor with Catastrophic NaN Dropout", color="white", fontweight="bold")
    ax1.set_ylabel("Features")
    ax1.legend()
    fig.colorbar(im1, ax=ax1)

    # Panel 2: The Imputation Accuracy
    t_full = np.arange(SEQUENCE_LENGTH)
    t_drop = np.arange(DROP_FRAME, SEQUENCE_LENGTH)

    full_true_voltage = true_seq[0, :, FAILED_INDEX].cpu().numpy()

    ax2.plot(t_full, full_true_voltage, color="cyan", label="Ground Truth Voltage", linewidth=2)
    ax2.plot(
        t_drop,
        imputed_voltage[:, 0],
        color="orange",
        linestyle="--",
        label="Masker Imputation",
        linewidth=2,
    )
    ax2.axvline(x=DROP_FRAME, color="red", linestyle="--", linewidth=2)
    ax2.set_title(
        "Continuous-Time Spatial Imputation (Omega Voltage Track)", color="white", fontweight="bold"
    )
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    # Panel 3: Error Time Series
    abs_error = np.abs(imputed_voltage[:, 0] - true_voltage[:, 0])
    ax3.plot(t_drop, abs_error, color="magenta", linewidth=2, label="Absolute Error")
    ax3.axvline(x=DROP_FRAME, color="red", linestyle="--", linewidth=2)

    # Theoretical noise floor for Absolute Error
    # Given noise ~ N(0, 0.1), Mean Absolute Error should optimally be ~0.08
    ax3.axhline(y=0.08, color="lime", linestyle=":", linewidth=2, label="Optimal (Noise Floor)")
    ax3.set_title("Imputation Error (Absolute Difference)", color="white", fontweight="bold")
    ax3.set_ylabel("Error")
    ax3.set_xlabel("Time Step")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "9_sensor_dropout_demo.png")
    plt.savefig(output_path, dpi=300)
    print(f"[+] Demo Complete. Dashboard saved to: {output_path}")


if __name__ == "__main__":
    main()
