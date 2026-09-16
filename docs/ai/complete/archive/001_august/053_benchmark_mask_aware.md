# Role Instruction

You are an expert PyTorch ML Engineer. We have successfully verified the core
architectural components. Now, we are building the final empirical benchmarking
suite. Ensure strict separation of concerns, clean logging, and professional
matplotlib plotting.

# Task 1: The Predictor Module

Create `src/models/simulators/waddington_predictor.py`. Implement a `nn.Module`
named `WaddingtonPredictor`.

- **Imports:**
  - `import torch`
  - `import torch.nn as nn`
  - Import `BiologicalCartridgeFusion` from `src.models.encoders.fusion`
  - Import `BaselineSSM` from `src.models.ssm.baseline_ssm`
  - Import `MaskAwareSSM` from `src.models.ssm.mask_aware_ssm`
- **Init Args:** `ssm_type` (str: either `'baseline'` or `'mask_aware'`),
  `d_cartridge=30`, `n_modalities=2`, `d_model=64`.
- **Layers:**
  - `self.ssm_type = ssm_type`
  - `self.fusion = BiologicalCartridgeFusion(d_cartridge, n_modalities, d_model)`
  - If `ssm_type == 'baseline'`, `self.ssm = BaselineSSM(d_model)`
  - If `ssm_type == 'mask_aware'`, `self.ssm = MaskAwareSSM(d_model)`
  - `self.readout = nn.Linear(d_model, 1)` (To predict the 1D biological phase
    trajectory).
- **Forward Pass:** `forward(self, x_raw, mask)`
  - `latent_x, latent_gate = self.fusion(x_raw, mask)`
  - If `self.ssm_type == 'baseline'`, `h = self.ssm(latent_x)`
  - If `self.ssm_type == 'mask_aware'`, `h = self.ssm(latent_x, latent_gate)`
  - Return `self.readout(h)` (Shape: `[batch, seq_len, 1]`).

# Task 2: The Training Loop and Benchmarking Script

Create `src/experiments/01_train_synthetic_benchmark.py`.

- **Setup & Imports:**
  - Use
    `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))`
    to ensure absolute imports from `src` work when running from the root.
  - Import `torch`, `torch.nn as nn`, `torch.optim as optim`,
    `matplotlib.pyplot as plt`.
  - Import `SyntheticWaddingtonDataset` from `waddington_dataset` (Note: the
    dataset is at the ROOT of the repository, not in src/data).
  - Import `WaddingtonPredictor` from
    `src.models.simulators.waddington_predictor`.
- **Data Preparation:**
  - Since `SyntheticWaddingtonDataset` currently might only have a `__len__` of
    1 or doesn't support a size parameter natively, dynamically patch or wrap it
    in this script so the dataloader can draw 100 samples (e.g., dynamically set
    `dataset.__len__ = lambda: 100` or create a small wrapper class).
  - Use `torch.utils.data.DataLoader` with `batch_size=8`, `shuffle=True`.
- **Model Initialization:**
  - Instantiate `model_baseline = WaddingtonPredictor('baseline')`
  - Instantiate `model_mask_aware = WaddingtonPredictor('mask_aware')`
  - Create two optimizers: `AdamW` for both with `lr=0.005`.
  - Use `nn.MSELoss()`.
- **Training Loop:**
  - Train for exactly 20 epochs.
  - Move inputs (`x_raw`, `mask`, `y_true`) to the appropriate device (CPU or
    MPS/CUDA).
  - Track `baseline_loss_history` and `mask_aware_loss_history` (average loss
    per epoch).
  - Print progress for each epoch (e.g., "Epoch 1/20 | Baseline Loss: 0.850 |
    Mask-Aware Loss: 0.650").
- **Evaluation & Plotting:**
  - Fetch a single batch from the dataset to use as a test sequence. Take the
    first item in the batch (index `0`).
  - Set models to `eval()` and use `with torch.no_grad():`. Get predictions for
    the test sequence.
  - Generate a `matplotlib` figure with 2 vertically stacked subplots (figsize
    10x8).
    - **Top Subplot (Loss Convergence):** Plot `baseline_loss_history` (Red
      dashed) and `mask_aware_loss_history` (Blue solid). Add a legend, title
      "MSE Loss Convergence", and Y-axis label "MSE".
    - **Bottom Subplot (Trajectory Prediction):** Plot `test_y_true` (Black,
      thick, label "True Phase"). Plot Baseline prediction (Red dashed, label
      "Zero-Padded Baseline"). Plot Mask-Aware prediction (Blue solid, label
      "Mask-Aware Routing"). Add a legend, and title "Waddington Phase
      Transition Tracking".
  - Save to `output/data/02_benchmark_results.png`. Ensure the `output/data`
    folder is created.

# Execution Requirement

Add the standard `if __name__ == '__main__':` block to execute the training run.
Ensure robust error handling for device placement.
