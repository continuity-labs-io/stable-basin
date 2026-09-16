Context: We need to upgrade the MambaLRP interpretability module in
`src/demo/6_mamba_lrp_demo.py`. Currently, `MambaLRP_FirstOrder` uses a naive
Input \* Gradient approximation (First-Order Taylor Decomposition). This
violates the relevance conservation axiom, causing the attribution signal to
shatter when backpropagating through the continuous-time discretization
parameters of the Mamba-2 backbone.

Task: Implement a mathematically exact Layer-wise Relevance Propagation ruleset
specifically designed for the `mamba_ssm.Mamba2` architecture.

1. Create a new class `MambaLRPEpsilon` to replace `MambaLRP_FirstOrder`.
2. Do not use standard PyTorch `backward()`. You must implement a custom
   backward hook (or gradient override) that distributes relevance using the
   LRP-epsilon rule:
   `R_i = sum_j ( (a_i * w_ij) / (sum_k (a_k * w_kj) + epsilon) ) * R_j`.
3. To handle Mamba-2 specifically, unroll the continuous-time discretization
   matrices (the B and C gating parameters) and assign exact
   conservation-of-relevance scores to the state memory retention and input
   gating matrices.
4. The output must be a spatiotemporal relevance tensor
   `[Batch, Time, Channels]` that perfectly conserves the relevance of the
   output prediction back to the input, handling zero-division safely using the
   `epsilon` stabilizer.
5. Ensure memory complexity remains O(N) by operating chunk-wise, avoiding
   materializing the full attention-equivalent matrix in VRAM.
6. Include unit tests that verify the relevance conservation axiom (total output
   relevance equals total input relevance) within a single backward pass.
7. Put the new class in a dedicated module, perhaps in the metrics folder or an
   interpretability folder.
