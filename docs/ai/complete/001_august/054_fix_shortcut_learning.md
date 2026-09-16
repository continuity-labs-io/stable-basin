# Role Instruction

You are an expert PyTorch ML Engineer. We need to fix a "Shortcut Learning" data
leakage issue in our synthetic benchmark. The neural networks cheated by
ignoring the sparse data entirely because the continuous data contained all the
answers.

# Task 1: Rewrite the Dataset Simulator

Overwrite `waddington_dataset.py` at the root of the repository. Update
`SyntheticWaddingtonDataset` with this strict logic:

1. **Imports:** `import torch`, `from torch.utils.data import Dataset`,
   `import matplotlib.pyplot as plt`, `import os`

2. **The Dataset Class (`SyntheticWaddingtonDataset`):**

   - `__init__(self, size=100, seq_len=500)`: Store size and seq_len. Generate a
     static projection matrix for consistency across the dataset:
     `self.W_1 = torch.randn(1, 10)`.
   - `__len__(self)`: Return `self.size`.
   - `__getitem__(self, idx)`:
     - **The Target (`y_true`):** The cell moves through discrete Waddington
       attractor basins. Initialize a zeros tensor of shape `[seq_len, 1]`. Pick
       a random integer `jump1` between 100 and 200. Pick a random integer
       `jump2` between 300 and 400. Set `y_true[jump1:jump2] = 1.0`, and
       `y_true[jump2:] = 2.0`. Add slight Gaussian noise (std=0.02).
     - **Modality 0 (Continuous Voltage, 20D) -> Pure Background Noise:** This
       modality must **NOT** track `y_true`. Generate pure Gaussian noise and
       random sine waves (Shape `[seq_len, 20]`). It must contain zero
       information about the phase transition.
     - **Modality 1 (Sparse Epigenetics, 10D) -> The Causal Driver:** This
       modality **MUST** track `y_true`.
       `modality_1 = y_true * self.W_1 + torch.randn(seq_len, 10) * 0.05`.
     - **The Mask:**
       - Modality 0 mask is a column of ones: `torch.ones(seq_len, 1)`.
       - Modality 1 mask is strictly sparse:
         `mask_1 = (torch.rand(seq_len, 1) > 0.95).float()`. (Only ~5% active).
       - **Hack to ensure observability:** Guarantee the mask is `1.0` exactly
         at `jump1 + 5` and `jump2 + 5` so the model definitively sees the state
         transition shortly after it occurs.
       - **CRITICAL ZERO-PADDING:** `modality_1 = modality_1 * mask_1`.
       - Combine masks:
         `mask = torch.cat([torch.ones(seq_len, 1), mask_1], dim=1)`. (Shape
         `[seq_len, 2]`).
     - **Output:** `x_raw = torch.cat([modality_0, modality_1], dim=1)`. Return
       `{'x_raw': x_raw, 'mask': mask, 'y_true': y_true}`.

3. **Diagnostic Plotting (Bottom of file):**
   - Retain the `if __name__ == '__main__':` block.
   - Instantiate `dataset = SyntheticWaddingtonDataset(size=1)`
   - Fetch `batch = dataset[0]`.
   - Plot a 3-panel matplotlib figure vertically stacked: Top is `y_true`,
     Middle is `x_raw` heatmap (transposed), Bottom is `mask` heatmap
     (transposed). Save to `outputs/01_synthetic_data_preview.png`.

# Task 2: Update the Training Script Epochs

In `src/experiments/01_train_synthetic_benchmark.py`:

- Change the number of epochs to `30`.
- Do not change anything else in the training script.
