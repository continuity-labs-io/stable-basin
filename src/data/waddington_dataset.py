import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import math


class SyntheticWaddingtonDataset(Dataset):
    """
    Synthetic biological dataset representing a cell moving through a phase transition
    (the Waddington landscape).

    Generates synthetic sequences comprising:
    - y_true: The 1D target tracking continuous phase transitions (FitzHugh-Nagumo fast variable).
    - x_raw: A 30-dimensional tensor composed of two modalities:
      - Modality 0 (20D): Continuous projection of the slow recovery variable w.
      - Modality 1 (10D): Sparse projection of the fast spiking variable v (masked 95% of the time).
    - mask: A 2-dimensional tensor representing the observability of the two modalities.
    """

    def __init__(self, size: int = 100, seq_len: int = 500, sparsity: float = 0.05):
        self.size = size
        self.seq_len = seq_len
        self.sparsity = sparsity
        # Ensure consistent biological mapping across datasets
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.W_0 = torch.randn(1, 20)
        self.W_1 = torch.randn(1, 10)
        torch.set_rng_state(rng_state)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # FHN Parameters
        tau = 10000.0
        a = 0.7
        b = 0.8
        I_ext = 0.5
        dt = 0.1
        sub_steps = 200

        v = torch.zeros(self.seq_len, 1)
        w = torch.zeros(self.seq_len, 1)

        # Randomize initial conditions slightly to vary sequences
        v_curr = torch.randn(1).item() * 0.5
        w_curr = torch.randn(1).item() * 0.1

        # Euler Integration Loop
        for i in range(self.seq_len):
            for _ in range(sub_steps):
                dv = v_curr - (v_curr**3) / 3.0 - w_curr + I_ext
                dw = (v_curr + a - b * w_curr) / tau
                v_curr += dv * dt
                w_curr += dw * dt
            v[i, 0] = v_curr
            w[i, 0] = w_curr

        # Target is the fast variable
        y_true = v

        # Modality 0 (Continuous slow variable, 20D)
        modality_0 = w @ self.W_0 + torch.randn(self.seq_len, 20) * 0.05

        # Modality 1 (Sparse fast variable, 10D)
        modality_1 = v @ self.W_1 + torch.randn(self.seq_len, 10) * 0.05

        # The Mask
        mask_0 = torch.ones(self.seq_len, 1)
        # sparsity is the fraction of active sensors
        mask_1 = (torch.rand(self.seq_len, 1) < self.sparsity).float()

        # CRITICAL ZERO-PADDING
        modality_1 = modality_1 * mask_1

        # Combine masks
        mask = torch.cat([mask_0, mask_1], dim=1)

        # Output
        x_raw = torch.cat([modality_0, modality_1], dim=1)
        return {"x_raw": x_raw, "mask": mask, "y_true": y_true}

if __name__ == "__main__":
    dataset = SyntheticWaddingtonDataset(size=1)
    batch = dataset[0]

    y_true = batch["y_true"]
    x_raw = batch["x_raw"]
    mask = batch["mask"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(y_true.numpy(), color="black", linewidth=2)
    axes[0].set_title("y_true Trajectory (Fast Variable v)")

    im1 = axes[1].imshow(x_raw.numpy().T, aspect="auto", cmap="viridis", interpolation="none")
    axes[1].set_title("x_raw Heatmap (Top 20=Continuous w, Bottom 10=Sparse v)")

    im2 = axes[2].imshow(mask.numpy().T, aspect="auto", cmap="binary", interpolation="none")
    axes[2].set_title("mask Heatmap")

    plt.tight_layout()
    os.makedirs("output/data", exist_ok=True)
    plt.savefig("output/data/00_synthetic_data_preview.png")
    print("Saved diagnostic preview to output/data/00_synthetic_data_preview.png")
