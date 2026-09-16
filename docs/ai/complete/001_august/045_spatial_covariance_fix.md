Context: We are diagnosing the `9_dynamic_sensor_masking.py` imputation failure.
The neural network is failing to impute the dropped sensor because
`NeocorticalAssembloidDataset` generates 114 mathematically independent
frequencies, providing zero spatial covariance for the network to learn from.

Task: Update `src/pipeline/neocortical_assembloid_dataloader.py` to simulate
biological spatial covariance.

1. In `__iter__`, instead of creating 114 independent frequencies, create 5
   "master" biological rhythms (e.g., `num_master_rhythms = 5`). Generate `t`
   and create a `master_signals` tensor of shape `[time_steps, 5]` using 5
   distinct frequencies.
2. Outside the `while True:` loop (so it remains constant across batches),
   generate a random mixing matrix:
   `self.mixing_matrix = torch.randn(5, self.latent_dim)`.
3. Inside the loop, generate the 114-D sequence by multiplying the master
   signals by the mixing matrix:
   `signal = torch.matmul(master_signals, self.mixing_matrix)`.
4. Add the standard biological `noise` tensor on top as before.
5. Yield the `sequence_tensor`.
