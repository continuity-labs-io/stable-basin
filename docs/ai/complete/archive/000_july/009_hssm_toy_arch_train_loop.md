In `src/demo/hssm_toy_model.py`, add the modeling and training logic to prove
the Learned High-Pass Filter and Orthogonal Veto.

1. Import `StateSpaceEngine` from `src.models.state_space_engine`.
2. Import `torch.nn` as `nn` and `torch.optim` as `optim`.
3. Write a `train_orthogonal_veto(device)` function:
   - Instantiate the "Edge Compressor":
     `gevi_compressor = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=200, stride=200).to(device)`
   - Instantiate the Fusion Core:
     `mamba_engine = StateSpaceEngine(d_model=768 + 64).to(device)`
   - Set up an Adam optimizer (lr=1e-3).
   - Initialize `env = ToyBiologicalEnvironment()`.
   - Run a fast training loop (150 iterations). Every iteration, generate a
     fresh batch (size 16) of `scenario="homeostasis"`.
   - Forward pass: a) Pass GEVI through `gevi_compressor` and transpose to
     [Batch, Time, 64]. b) Concatenate with Optical [Batch, Time, 768] ->
     [Batch, Time, 832]. c) Pass through `mamba_engine` to get `scalar_loss`
     (predictive coding).
   - Backpropagate.
4. By training the network to predict T\_{t+1} on data where artifacts are
   deterministic continuous noise, this loop mathematically forces the `Conv1d`
   to isolate the spikes and teaches Mamba the expected orthogonal coupling.
   Return the trained `gevi_compressor` and `mamba_engine`.

Requirements:

- Include basic support for apple MPS if possible.
- Ensure adequate documentation.
- Avoid magic numbers by using constants with meaningful names.
