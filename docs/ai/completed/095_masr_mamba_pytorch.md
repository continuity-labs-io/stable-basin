See: src/models/ssm/masr_mamba.py

Let's strip away all complexity and fix the MASR + Mamba integration using a pure PyTorch reference implementation. We are not doing hierarchical SSMs right now. We are strictly focusing on the Latent Stasis sensor fusion paper baseline.

Since we cannot use the Triton hardware-accelerated scan on this Mac, please write a mamba_masr_reference_scan function in pure PyTorch, wrapped in a clean nn.Module named PyTorchMambaMASR.

Requirements for the Reference Model:

Accept standard inputs: x (sequence data), dt (dynamic time deltas), and mask (boolean sparsity mask indicating missing sensor data).

Use a standard PyTorch for loop over the sequence length to simulate the recurrent Mamba scan sequentially.

Implement Latent Stasis by integrating the sparsity mask directly into the data-dependent Δt parameter. Before calculating the discrete Ā and B̄ matrices at each time step, multiply the continuous Δt by the boolean mask.

Ensure that when the mask is 0, Δt goes to 0, driving Ā to the Identity matrix and B̄ to 0, thereby perfectly freezing the hidden state (h_t = h_{t-1}) for that subspace.

Create a simple execution block at the bottom that generates a synthetic dataset mimicking asynchronous multi-rate biology (e.g., a fast signal and a slow signal with random dropouts). Pass it through PyTorchMambaMASR, and output a matplotlib graph comparing the MASR reconstruction against a standard zero-padded approach.

Ensure all mathematical variables in the Python comments use standard Unicode (e.g., Δt, Ā, B̄) rather than LaTeX formatting.

