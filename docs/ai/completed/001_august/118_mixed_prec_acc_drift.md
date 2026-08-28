Please fix a severe mixed-precision accumulation drift issue in the `mamba_masr_reference_scan` function. 

Currently, the hidden state `h` is initialized using `dtype=x.dtype`. If `x` is passed in as FP16 or BF16 on edge devices, accumulating the recurrent state over thousands of continuous time steps will suffer from floating-point rounding drift, permanently shifting the biological attractor centroid. 

Please make the following changes:
1. Force the initialization of `h` to strictly use `dtype=torch.float32`, regardless of `x.dtype`. Keep `device=x.device`.
2. Inside the sequence loop, ensure the recurrence `h = A_bar * h + B_bar * x_t.unsqueeze(-1)` operates entirely in FP32 (cast `A_bar`, `B_bar`, and `x_t` to `torch.float32` during this step if necessary).
3. Cast the resulting output `y_t` back to `x.dtype` right before assigning it to the output tensor `y` to preserve the network's expected forward-pass types.
