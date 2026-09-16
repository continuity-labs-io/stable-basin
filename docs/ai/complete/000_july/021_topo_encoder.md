Create a new PyTorch module `TopoEncoder` in `src/models/topo_encoder.py`.
Context: This model ingests the continuous E-field flow and uses Mamba-2 to
extract the macroscopic geometric shape (the Dynamic Attractor Basin) into a
fixed latent vector.

Requirements:

1. Inherit from `nn.Module`. Import `Mamba` from `mamba_ssm`.
2. Architecture:
   - A 2D Convolutional frontend (`nn.Conv2d`) to compress the spatial
     dimensions of the E-field `[Batch, Time, 2, 64, 64]`. Reshape the spatial
     frames so the output is `[Batch, Time, d_model]` where d_model=768.
   - Pass the flattened sequence through a `Mamba` block (d_model=768,
     d_state=16, d_conv=4, expand=2) to track the continuous thermodynamic loops
     over time.
   - Extract the hidden state of the final time step (representing the fully
     formed standing wave).
   - Pass this final state through an MLP projection head with LayerNorm to
     ensure it is mapped smoothly to the 768-D space.
3. Return the 768-D projected LFP latent vector, and optionally the full
   sequence of hidden states for KSM metric extraction (if requested via a
   `return_hidden=True` flag).
