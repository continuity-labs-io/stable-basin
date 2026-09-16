Role Instruction You are an expert PyTorch ML Engineer. We must implement a
"Dual-Lock Stasis" fix to prevent our Mask-Aware SSM from integrating ghost
noise, and then execute the true Length Extrapolation benchmark.

Task 1: Update the Fusion Encoder (Stasis Prior) Modify
src/models/encoders/fusion.py. Add a custom initialization to
BiologicalCartridgeFusion. Update the **init** method so it ends with these
exact two lines:

```python
# THE STASIS PRIOR: Default closed (-3), opens when active (+6)
        torch.nn.init.constant_(self.W_gate.bias, -3.0)
        torch.nn.init.constant_(self.W_gate.weight, 6.0 / n_modalities)
```

Task 2: Update the Mask-Aware SSM (Explicit Payload Gating) Modify
src/models/ssm/mask_aware_ssm.py. In the forward loop, we must explicitly gate
the input payload so offline sensors inject absolutely zero noise into the
hidden state. Change the inner loop to look exactly like this:

```python
dt_base = torch.nn.functional.softplus(self.dt_proj(x_t))
dt_gated = dt_base * g_t + 1e-8

# EXPLICIT INPUT GATING: Block offline sensors from adding ghost noise to the state
B = self.B_proj(x_t) * g_t

A_bar = torch.exp(A * dt_gated)
B_bar = (A_bar - 1.0) / (A - 1e-8) * B

h_prev = A_bar * h_prev + B_bar
hidden_states.append(h_prev)
```

Task 3: Create the Extrapolation Script Overwrite
src/experiments/02_extrapolation_benchmark.py with the following complete
script. (This updates DatasetWrapper to properly accept seq_len so the
extrapolation actually runs on 2000 steps!)

Modify `src/experiments/02_extrapolation_benchmark.py` to surgically apply the
following changes (do NOT blindly overwrite the file, as that would delete
recent CSV logic and device fixes):

1. **DatasetWrapper Update**: Modify `DatasetWrapper.__init__` to accept
   `seq_len: int = 500` and pass it to `SyntheticWaddingtonDataset`.
2. **Training Setup**: Update the `train_dataset` instantiation to pass
   `seq_len=500`.
3. **Epochs**: Increase the training loop to run for `40` epochs.
