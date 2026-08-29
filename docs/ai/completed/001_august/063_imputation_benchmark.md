# Role Instruction

You are an expert PyTorch ML Engineer preparing empirical baselines for a ML
submission. We need to implement two standard engineering baselines for missing
data (Forward-Fill and Mask-Concatenation) and execute a 5-way
out-of-distribution extrapolation benchmark.

# Task 1: Update the Predictor Module

Overwrite `src/models/simulators/waddington_predictor.py`.

```python
import torch
import torch.nn as nn
from src.models.encoders.fusion import BiologicalCartridgeFusion
from src.models.ssm.baseline_ssm import BaselineSSM
from src.models.ssm.mask_aware_ssm import MaskAwareSSM
from src.models.attention.baseline_transformer import BaselineTransformer

class WaddingtonPredictor(nn.Module):
    def __init__(
        self, ssm_type: str, d_cartridge: int = 30, n_modalities: int = 2, d_model: int = 64
    ):
        super().__init__()
        self.ssm_type = ssm_type

        # For mask_concat, we concatenate the 2D mask to the 30D raw data
        if ssm_type == "mask_concat":
            self.fusion = BiologicalCartridgeFusion(d_cartridge + n_modalities, n_modalities, d_model)
        else:
            self.fusion = BiologicalCartridgeFusion(d_cartridge, n_modalities, d_model)

        if ssm_type in ["baseline", "forward_fill", "mask_concat"]:
            self.ssm = BaselineSSM(d_model)
        elif ssm_type == "mask_aware":
            self.ssm = MaskAwareSSM(d_model)
        elif ssm_type == "transformer":
            self.ssm = BaselineTransformer(d_model)
        else:
            raise ValueError(f"Unknown ssm_type: {ssm_type}")

        self.readout = nn.Linear(d_model, 1)

    def forward(self, x_raw: torch.Tensor, mask: torch.Tensor):
        if self.ssm_type == "forward_fill":
            # Apply Forward-Fill (Hold-Last-Value) to the sparse modality (indices 20-29)
            x_ff = x_raw.clone()
            batch_size, seq_len, _ = x_raw.shape
            last_known = torch.zeros_like(x_raw[:, 0, 20:])
            for t in range(seq_len):
                m_t = mask[:, t, 1:2]
                current_x = x_raw[:, t, 20:]
                last_known = torch.where(m_t == 1.0, current_x, last_known)
                x_ff[:, t, 20:] = last_known

            latent_x, latent_gate = self.fusion(x_ff, mask)
            h = self.ssm(latent_x)

        elif self.ssm_type == "mask_concat":
            # Concatenate mask directly to the features
            x_concat = torch.cat([x_raw, mask], dim=-1)
            latent_x, latent_gate = self.fusion(x_concat, mask)
            h = self.ssm(latent_x)

        else:
            latent_x, latent_gate = self.fusion(x_raw, mask)
            if self.ssm_type == "baseline":
                h = self.ssm(latent_x)
            elif self.ssm_type == "mask_aware":
                h = self.ssm(latent_x, latent_gate)
            elif self.ssm_type == "transformer":
                h = self.ssm(latent_x)

        return self.readout(h)
```

Task 2: Create the 5-Way Imputation Benchmark

Create src/experiments/03_imputation_benchmark.py.

```
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.models.simulators.waddington_predictor import WaddingtonPredictor
from src.utils.device import get_optimal_device
from torch.utils.data import DataLoader, Dataset

class DatasetWrapper(Dataset):
    def __init__(self, size: int, seq_len: int = 500):
        self.dataset = SyntheticWaddingtonDataset(size=size, seq_len=seq_len)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]

def main():
    device = get_optimal_device(verbose=True)

    # Data Preparation
    train_dataset = DatasetWrapper(size=100, seq_len=500)
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Initialize all 5 models
    model_names = ["baseline", "forward_fill", "mask_concat", "transformer", "mask_aware"]
    models = {name: WaddingtonPredictor(name).to(device) for name in model_names}

    optimizers = {name: optim.AdamW(models[name].parameters(), lr=0.005) for name in model_names}
    criterion = nn.MSELoss()

    loss_history = {name: [] for name in model_names}

    # Training Loop
    epochs = 40
    print(f"Training {len(models)} architectures on seq_len=500...")
    for epoch in range(1, epochs + 1):
        for m in models.values():
            m.train()

        running_losses = {name: 0.0 for name in model_names}

        for batch in dataloader:
            x_raw = batch["x_raw"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["y_true"].to(device)

            for name, model in models.items():
                optimizers[name].zero_grad()
                preds = model(x_raw, mask)
                loss = criterion(preds, y_true)
                loss.backward()
                optimizers[name].step()
                running_losses[name] += loss.item()

        log_str = f"Epoch {epoch:02d}/{epochs} | "
        for name in model_names:
            avg_loss = running_losses[name] / len(dataloader)
            loss_history[name].append(avg_loss)
            log_str += f"{name}: {avg_loss:.3f} | "

        if epoch % 5 == 0 or epoch == 1:
            print(log_str)

    print("\nRunning Length Extrapolation Stress Test (seq_len=2000)...")
    for m in models.values():
        m.eval()

    ood_dataset = SyntheticWaddingtonDataset(size=1, seq_len=2000)
    test_batch = ood_dataset[0]

    test_x_raw = test_batch["x_raw"].unsqueeze(0).to(device)
    test_mask = test_batch["mask"].unsqueeze(0).to(device)
    test_y_true = test_batch["y_true"].cpu().numpy()

    preds_dict = {}
    with torch.no_grad():
        for name, model in models.items():
            preds_dict[name] = model(test_x_raw, test_mask)[0].cpu().numpy()

            # Calculate OOD MSE (T > 500)
            ood_mse = ((preds_dict[name][500:] - test_y_true[500:])**2).mean()
            print(f"OOD-MSE [{name}]: {ood_mse:.4f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Top Subplot: Training Loss
    colors = {
        "baseline": "r--",
        "forward_fill": "m-.",
        "mask_concat": "y-",
        "transformer": "g:",
        "mask_aware": "b-"
    }

    labels = {
        "baseline": "Zero-Padded SSM",
        "forward_fill": "Forward-Fill SSM",
        "mask_concat": "Mask-Concat SSM",
        "transformer": "Causal Transformer",
        "mask_aware": "MASR (Ours)"
    }

    for name in model_names:
        ax1.plot(loss_history[name], colors[name], linewidth=2, label=labels[name])

    ax1.set_title("MSE Loss Convergence (Training on seq_len=500)")
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Bottom Subplot: Extrapolation
    ax2.plot(test_y_true, "k-", linewidth=4, label="True Phase")
    for name in model_names:
        linewidth = 2.5 if name == "mask_aware" else 1.5
        alpha = 1.0 if name == "mask_aware" else 0.8
        ax2.plot(preds_dict[name], colors[name], linewidth=linewidth, alpha=alpha, label=labels[name])

    ax2.axvline(x=500, color="grey", linestyle="--", linewidth=2)
    ax2.text(510, ax2.get_ylim()[1] * 0.9, "Training Horizon\n(Length Extrapolation)", color="grey", fontsize=10)

    ax2.set_title("Waddington Phase Tracking: Out-Of-Distribution Stress Test (seq_len=2000)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Phase State")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs("output/data", exist_ok=True)
    plt.savefig("output/data/04_arxiv_money_chart.png", dpi=300)
    print("Saved plot to output/data/04_arxiv_money_chart.png")

    results = {"True Phase": test_y_true.flatten()}
    for k, v in preds_dict.items():
        results[k] = v.flatten()

    df = pd.DataFrame(results)
    df.to_csv("output/data/04_arxiv_results.csv", index=False)
    print("Saved CSV to output/data/04_arxiv_results.csv")

if __name__ == "__main__":
    main()
```
