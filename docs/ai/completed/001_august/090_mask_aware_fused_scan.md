Context Files to Load:

- src/models/ssm/mask_aware_mamba.py 
- (Create Target) src/models/ssm/triton_fused_scan.py

Raw Prompt to Execute:

We are writing a Mask-Aware Fused Scan Triton Kernel. If we run our packing and dynamic integration in standard PyTorch, the GPU will choke reading and writing the massive hidden state back to VRAM for every irregular time step[cite: 1]. 

Write a .py file containing a Triton kernel (@triton.jit) wrapped in a standard torch.autograd.Function[cite: 1]. The kernel must load the biological hidden state (h_t) into the GPU's ultra-fast SRAM and read the packed Event Tensor[cite: 1]. 

If an event belongs to a specific sensor (e.g., the MEA), it must compute the Dynamic Δt and update only that subspace of the hidden state, explicitly ignoring the masked subspace (e.g., leaving the optical subspace in stasis)[cite: 1]. It must process the entire sequence in one blistering fast sweep inside the SRAM, and only write the final output to slow memory[cite: 1].

