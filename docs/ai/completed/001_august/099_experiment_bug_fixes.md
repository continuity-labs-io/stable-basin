apply these exact fixes to sanitize the environment:

1. Fix the MASR Mamba Explosion (src/models/ssm/masr_mamba.py)Replace the Euler
   approximation with the exact Zero-Order Hold (ZOH) discretization. Because
   $A$ is strictly negative, ZOH mathematically bounds the state injection even
   if $\Delta t$ spikes

```Python
# Replace B_bar = (dt_masked_t_exp * B_t.unsqueeze(1)) with:
A_bar = torch.exp(dt_masked_t_exp * A)
B_bar = (A_bar - 1.0) / (A - 1e-8) * B_t.unsqueeze(1)
```

2. Restore Subspace Routing (src/models/ssm/masr_mamba.py)

Remove the amax aggregation so the exact continuous mask acts upon the distinct
dimensional subspaces:

```Python
if self.mask_aware and mask is not None:
    # Pass the per-channel mask directly!
    hidden_states = self.mamba(h, mask)
```

3. Fix the A-Matrix Initialization log(0) Risk
   (src/models/ssm/physics/matrix_a_factory.py)In RandomAInit, torch.rand can
   technically generate 0.0, and torch.log(0.0) = -inf, which will silently
   brick your $A$ matrix with NaNs on initialization. Clamp it:

```Python
self.A_log = nn.Parameter(torch.log(torch.clamp(torch.rand(shape), min=1e-4) * a_scale + a_shift))
```

4. Fix the Transformer Gradient Stability
   (src/models/attention/baseline_transformer.py)Add norm_first=True to prevent
   the initial gradient explosions during burn-in training:

```Python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model, nhead=nhead, batch_first=True,
    dim_feedforward=d_model * ff_expansion_factor,
    norm_first=True  # <--- ADD THIS
)
```

ensure the changes are tested.
