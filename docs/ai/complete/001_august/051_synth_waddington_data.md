# Role Instruction

You are an expert PyTorch ML Engineer and Computational Biologist. We are
building a modular PyTorch benchmarking suite for a novel AI architecture.
Create a file at `src/data/waddington_dataset.py`. Do not output monolithic
scripts; prioritize clean object-oriented design, strictly typed methods,
docstrings, and PyTorch best practices.

# Task

Create a PyTorch `Dataset` class called `SyntheticWaddingtonDataset`. It must
generate a batch of synthetic biological sequences representing a cell moving
through a phase transition (the Waddington landscape).

1. **The True State (The Target):**

   - For a sequence of length `seq_len` (default 500), generate a continuous
     hidden trajectory `y_true` (Shape: `[seq_len, 1]`).
   - This should smoothly transition from 0.0 to 1.0 over time using a scaled
     sigmoid function centered randomly between `t=200` and `t=300`. Add a tiny
     bit of Gaussian noise (random walk) so it isn't perfectly smooth.

2. **The Modalities (The Raw Data):**

   - **Modality 0 (Continuous, 20-Dimensional):** A continuous high-frequency
     sine wave whose baseline tracks `y_true`. Add distinct frequencies and
     noise per dimension.
   - **Modality 1 (Sparse, 10-Dimensional):** A slow-moving signal tracking
     `y_true`, plus noise.

3. **The Masking (Hardware Limits):**

   - Create a float `mask` tensor (Shape: `[seq_len, 2]`).
   - Modality 0 is fully observed. Its mask channel is always `1.0`.
   - Modality 1 is highly sparse. It should only be observed in rare, isolated
     bursts (e.g., randomly choose 5% of the timesteps to have mask = `1.0`,
     rest are `0.0`).
   - **CRITICAL:** Apply the mask to Modality 1. Whenever the mask is `0.0`, the
     10-D data for Modality 1 must be explicitly forced to exactly `0.0`
     (zero-padded).
   - Concatenate the 20-D Modality 0 and 10-D Modality 1 into a single `x_raw`
     tensor (Shape: `[seq_len, 30]`).

4. **Output Format:**

   - `__getitem__` must return a dictionary:
     - `x_raw`: Tensor of shape `(seq_len, 30)`
     - `mask`: Tensor of shape `(seq_len, 2)`
     - `y_true`: Tensor of shape `(seq_len, 1)`
   - Also implement `__len__`.

5. **Diagnostic Block:**
   - At the bottom of the file, include an `if __name__ == '__main__':` block.
   - Instantiate the dataset (size=1), fetch the sequence, and use `matplotlib`
     to plot three vertically stacked subplots:
     1. The 1D `y_true` trajectory over time.
     2. A heatmap (using `imshow`) of the 30-D `x_raw` (showing the continuous
        20D signal and the heavily zero-padded 10D signal). Transpose the tensor
        so time is on the x-axis.
     3. The 2-D Mask over time (heatmap or line plot).
   - Save this plot to `outputs/data/01_synthetic_data_preview.png`. Ensure the
     `outputs/data` directory is created safely if it doesn't exist.
