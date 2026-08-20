import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.sim2real.neocortical_assembloid_dataloader import NeocorticalAssembloidDataset
from src.models.ssm.neocortical_engine import NeocorticalEngine
from src.models.losses.meld_loss import MeldLoss
from src.utils.device import get_optimal_device
from src.metrics.diagnostic_engine import ThermodynamicDiagnosticEngine


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    device = get_optimal_device(allow_mps=False)
    print(f"[*] Booting Neocortical Benchmark Demo on: {device.type.upper()}")

    print("[*] Initializing Neocortical Assembloid Mock DataLoader...")
    dataset = NeocorticalAssembloidDataset(time_steps=200, latent_dim=114)
    dataset_iter = iter(dataset)

    engine = NeocorticalEngine(input_dim=114, d_model=768, d_state=64).to(device)
    criterion = MeldLoss(alpha=1.0, beta=0.1, gamma=0.5, L=1.5).to(device)
    optimizer = optim.AdamW(engine.parameters(), lr=1e-3)

    iterations = 20
    print(f"[*] Running rapid burn-in optimization ({iterations} iterations)...")
    engine.train()

    for i in range(iterations):
        seq, _ = next(dataset_iter)
        seq = seq.unsqueeze(0).to(device)  # Add batch dimension [1, 200, 114]

        optimizer.zero_grad()

        input_seq = seq[:, :-1, :]
        target_seq = seq[:, 1:, :]

        pred_t_plus_1, reconstructed_t = engine(input_seq)

        # Simple scalar delta_x
        delta_x = torch.ones(seq.size(0), 1).to(device) * 0.1

        loss, metrics = criterion(input_seq, target_seq, pred_t_plus_1, reconstructed_t, delta_x)
        loss.backward()
        optimizer.step()

        print(f"    Iteration {i + 1}/{iterations} | Loss: {loss.item():.4f}")

    print("[*] Simulating Waddington Crash & running MambaLRP attribution...")
    engine.eval()

    test_seq, _ = next(dataset_iter)
    test_seq = test_seq.unsqueeze(0).to(device)

    # Inject severe metabolic drop mid-stream
    EVENT_FRAME = 140
    test_seq[:, EVENT_FRAME:, :] *= 0.05

    # Extract attribution map
    from src.metrics.attribution_engine import AttributionEngine
    attribution_map = AttributionEngine.get_instance().compute_attribution(engine, test_seq, target_time_step=EVENT_FRAME)

    print("[*] Generating Thermodynamic Diagnostic Report...")
    diagnostic_engine = ThermodynamicDiagnosticEngine(engine)
    diagnostic_report = diagnostic_engine.generate_diagnostic(test_seq, EVENT_FRAME)
    import json

    print(json.dumps(diagnostic_report, indent=2))

    print("[*] Generating Publication-Ready Dashboard...")
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: Multi-Modal Biological Input Heatmap
    im1 = ax1.imshow(
        test_seq[0].detach().cpu().numpy().T, aspect="auto", cmap="viridis", origin="lower"
    )
    ax1.axvline(
        x=EVENT_FRAME, color="yellow", linestyle="--", linewidth=2, label="EVENT_FRAME (Crash)"
    )
    ax1.set_title("Multi-Modal Biological Input Heatmap", color="white", fontweight="bold")
    ax1.set_ylabel("Features (114-D)", color="white")
    ax1.set_xlabel("Time", color="white")
    ax1.legend()
    fig.colorbar(im1, ax=ax1)

    # Panel 2: MambaLRP Feature Attribution Heatmap
    attr_data = attribution_map[0].detach().cpu().numpy().T
    # Clamp the color scale to the 95th percentile so early-warning signals (like at T=125) pop
    vmax_val = np.percentile(attr_data, 95)
    im2 = ax2.imshow(attr_data, aspect="auto", cmap="inferno", origin="lower", vmax=vmax_val)
    ax2.axvline(
        x=EVENT_FRAME, color="yellow", linestyle="--", linewidth=2, label="EVENT_FRAME (Crash)"
    )
    ax2.set_title("MambaLRP Feature Attribution", color="white", fontweight="bold")
    ax2.set_ylabel("Features (114-D)", color="white")
    ax2.set_xlabel("Time", color="white")
    ax2.legend()
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "7_neocortical_benchmark_dashboard.png")
    plt.savefig(output_path)
    print(f"[*] Dashboard saved to {output_path}")


if __name__ == "__main__":
    main()
