Create a new PyTorch IterableDataset class named `ContinuousLFPDataset` in
`src/pipeline/uhd_lfp_dataloader.py`. Context: We are bypassing discrete spike
sorting to map the continuous macroscopic electromagnetic field of a 2D UHD-CMOS
microelectrode array.

Requirements:

1. Initialize with `time_steps` (e.g., 500 for a 500ms window at 1kHz) and
   `grid_size` (e.g., 64x64 electrodes).
2. For the mock data generation in `__iter__`:
   - Generate a continuous 2D traveling wave (representing the Local Field
     Potential, V_e) moving across the 64x64 grid over the time steps. Add
     biological 1/f noise.
   - The raw voltage shape must be `[time_steps, 1, 64, 64]`.
3. Implement the physics transformation: Calculate the electric field gradients
   E = -∇V_e.
   - Use `torch.gradient` to compute the spatial derivative of the voltage grid
     in both the X and Y directions.
   - Stack these gradients to form a 2-channel continuous tensor representing
     the directional flow of the standing wave.
4. The final yielded tensor for the LFP must be shape `[time_steps, 2, 64, 64]`.
5. Simultaneously yield a mock visual stimulus embedding (a randomly generated
   768-D vector representing the visual stimuli presented to the tissue during
   this window).
