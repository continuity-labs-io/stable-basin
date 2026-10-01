# Plan: Optimize Sequential Unrolling in BaselineSSM

## Objective
Optimize the sequential unrolling of the `BaselineSSM` model in `src/models/ssm/baseline_ssm.py` by replacing the element-wise computation inside the native Python for-loop with a parallelized/vectorized precomputation phase. 

## Current Problem
The `BaselineSSM` (and similarly `MaskAwareSSM`) computes linear projections (`dt_proj` and `B_proj`) and their activations sequentially inside a for-loop over `seq_len`. This significantly increases the overhead by extending the computation graph with $O(L)$ PyTorch operations.

## Proposed Fix
1. **Vectorized Precomputation**: We will compute `dt_base` and `B_base` for all timesteps at once by passing the entire sequence through the linear layers:
   ```python
   dt = torch.nn.functional.softplus(self.dt_proj(latent_x))
   B = self.B_proj(latent_x)
   ```
2. **Parallelized Graph**: We will also precompute the discretization coefficients (`A_bar` and `B_bar`) for the entire sequence in a vectorized manner:
   ```python
   A_bar = torch.exp(A * dt)
   B_bar = (A_bar - 1.0) / (A - 1e-8) * B
   ```
3. **Simpler Recurrence Loop**: Only the core hidden state recurrence ($h_t = A\_bar_t \times h_{t-1} + B\_bar_t$) will remain inside the Python for-loop, reducing the per-timestep operations to a single scalar-vector multiplication and addition.
4. **Validation**: Run the existing test suite (`tests/models/ssm/test_baseline_ssm.py`) to verify that the temporal state transitions match the previous behavior perfectly down to floating point precision.

*(Note: We can also apply this identical optimization to `src/models/ssm/masr_ssm.py` if desired, as it suffers from the exact same unoptimized unrolling bottleneck.)*
