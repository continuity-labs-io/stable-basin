import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from src.models.ssm.masr_mamba import MaskAwareMamba
from src.metrics.spectral import SpectralMetrics


def main():
    torch.manual_seed(42)

    # 1. Generate a simple 2D temporal dataset
    batch_size = 1
    seq_len = 100
    features = 2

    # Channel 0: Slow sine wave
    # Channel 1: Fast sine wave
    t = torch.linspace(0, 10, seq_len)

    # Note: Shifted up by 2.0 because MaskAwareMamba uses F.softplus on outputs
    # and therefore can only predict positive values.
    ch0 = torch.sin(t) + 2.0
    ch1 = torch.sin(3 * t) + 2.0

    x = torch.stack([ch0, ch1], dim=-1).unsqueeze(0)  # [1, 100, 2]

    # 2. At T=70, abruptly flatline both waves to 0.0 to simulate a crash
    x[:, 70:, :] = 0.0

    # For next-frame forecasting
    X_input = x[:, :-1, :]
    Y_target = x[:, 1:, :]

    # 3. Instantiate MaskAwareMamba with small latent space
    model = MaskAwareMamba(input_dim=features, d_model=16, d_state=8, mask_aware=False)

    # 4. Minimal training loop (100 epochs) using AdamW and MSE
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    print("Training trivial toy overfit model for 100 epochs...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()

        # model returns (pred_t_plus_1, reconstructed_t)
        preds, _ = model(X_input)

        loss = criterion(preds, Y_target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d}/100 | Loss: {loss.item():.4f}")

    # 5. Evaluate and plot
    model.eval()
    with torch.no_grad():
        preds, _ = model(X_input)

    preds = preds.squeeze(0).numpy()
    targets = Y_target.squeeze(0).numpy()

    # Extract predicted macro (Ch 0) and micro (Ch 1) channels
    macro_pred = torch.tensor(preds[:, 0])
    micro_pred = torch.tensor(preds[:, 1])

    # Instantiate SpectralMetrics and calculate global PAC
    spectral = SpectralMetrics()
    global_pac = spectral.calculate_cfc_pac(macro_pred, micro_pred).item()
    print(f"Global Phase-Amplitude Coupling (MVL) during prediction: {global_pac:.4f}")

    # Calculate rolling window PAC
    window_size = 20
    rolling_pac = []

    for i in range(len(macro_pred) - window_size + 1):
        window_macro = macro_pred[i : i + window_size]
        window_micro = micro_pred[i : i + window_size]
        pac_val = spectral.calculate_cfc_pac(window_macro, window_micro).item()
        rolling_pac.append(pac_val)

    # Pad to align with time steps
    import numpy as np

    rolling_pac = [np.nan] * (window_size - 1) + rolling_pac

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Channel 0 Plot
    axes[0].plot(targets[:, 0], label="True (Ch 0 - Slow)", color="#00ffcc", linewidth=2)
    axes[0].plot(preds[:, 0], label="Predicted", color="#ff00ff", linestyle="--", linewidth=2)
    axes[0].axvline(x=70, color="red", linestyle=":", linewidth=2, label="Biological Crash (T=70)")
    axes[0].set_title("Channel 0: Slow Sine Wave", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Channel 1 Plot
    axes[1].plot(targets[:, 1], label="True (Ch 1 - Fast)", color="#00ffcc", linewidth=2)
    axes[1].plot(preds[:, 1], label="Predicted", color="#ff00ff", linestyle="--", linewidth=2)
    axes[1].axvline(x=70, color="red", linestyle=":", linewidth=2, label="Biological Crash (T=70)")
    axes[1].set_title("Channel 1: Fast Sine Wave", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    # PAC Plot
    axes[2].plot(rolling_pac, label="Rolling PAC (Macro->Micro)", color="orange", linewidth=2)
    axes[2].axvline(x=70, color="red", linestyle=":", linewidth=2, label="Biological Crash (T=70)")
    axes[2].set_title("Spectral Decoherence: Phase-Amplitude Decoupling", fontweight="bold")
    axes[2].set_xlabel("Time Step (Frames)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs("output/demo", exist_ok=True)
    out_path = "output/demo/00_trivial_overfit.png"
    plt.savefig(out_path, dpi=300)
    print(f"Evaluation complete. Plot saved to {out_path}")


if __name__ == "__main__":
    main()
