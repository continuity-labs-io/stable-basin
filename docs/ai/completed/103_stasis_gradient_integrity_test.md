# Stasis Gradient Integrity Test

### Prompt 3: Stasis Gradient Integrity (The Rests)

**Target File:** `tests/models/ssm/test_masr_mamba_rigor.py`
**Context Files:** `src/models/ssm/masr_mamba.py`

**Instructions:**
Create a test named `test_masr_mamba_stasis_gradient_isolation`.
1. Initialize `PyTorchMambaMASR`. Forcefully set `model.D.data.fill_(0.0)` so the skip connection does not leak gradients directly.
2. Pass an input `x` with `requires_grad=True`.
3. Create a `mask` where a specific time step (e.g., `t=2`) is completely `0.0` (missing data), but `1.0` elsewhere.
4. Perform a forward pass, sum the output, and call `.backward()`.
5. Assert strictly that the gradient of `x` (`x.grad`) at the exact masked time step `t=2` is *exactly* `0.0`. 
If a sensor is masked, it must be mathematically severed from the backward pass. No error signal can leak across the void!
