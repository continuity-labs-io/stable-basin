Context: We are building the Large Biological Model (LBM) for Project CHRONOS
using the PyTorch ecosystem. I need to implement a custom loss function class
`MeldLoss(nn.Module)` that calculates a composite score for our state-space
model training loop.

The state space model processes continuous biological telemetry. Our forward
pass generates a predicted next state, and a reverse pass attempts to
reconstruct the previous state.

Please generate the PyTorch code for `MeldLoss` incorporating the following
three metrics:

1. Next-Frame Forecasting (L_forecast): Standard Mean Squared Error (MSE)
   between the predicted state at t+1 and the actual ground-truth state at t+1.
2. Lipschitz Penalty (L_lipschitz): A physics enforcer that penalizes the model
   if the rate of change exceeds biologically plausible limits. Calculate the L2
   norm of the predicted state change (Δy = pred_t_plus_1 - state_t). Penalize
   this norm if it exceeds L × Δx, where L is a configurable Lipschitz constant
   parameter and Δx is the magnitude of the perturbation or time step. Use a
   ReLU activation to only penalize positive violations: max(0, ||Δy|| - L ×
   Δx).
3. Time-Reversal Error (L_reverse): MSE between the reconstructed state at time
   t (derived by running the model backward from the predicted t+1 state) and
   the actual ground-truth state at time t.

The total loss should be a weighted sum: L_total = α × L_forecast + β ×
L_lipschitz + γ × L_reverse.

Requirements:

- Inherit from `torch.nn.Module`.
- The `__init__` method should accept the weights α, β, γ, and the Lipschitz
  constant L, and provide sensible defaults (e.g.,
  `alpha=1.0, beta=0.1, gamma=1.0, L=1.0`).
- The `forward` method signature should be:
  `def forward(self, state_t, target_t_plus_1, pred_t_plus_1, reconstructed_t, delta_x)`
- Use clean, vectorized, and optimized PyTorch tensor operations to ensure fast
  batch processing on edge GPUs.
- Assume inputs are batched tensors of shape `(batch_size, ...)`. Compute the L2
  norm for the Lipschitz penalty per-sample across all non-batch dimensions, and
  return the mean penalty across the batch.
- Assume `delta_x` is a batched tensor of shape `(batch_size, 1)`.
- The `forward` method should return a tuple containing the `L_total` scalar,
  and a dictionary of the individual detached loss components for telemetry
  logging.
