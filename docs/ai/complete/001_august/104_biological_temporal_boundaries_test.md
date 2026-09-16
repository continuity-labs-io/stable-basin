# Biological Temporal Boundaries Test

### Prompt 4: Biological Temporal Boundaries (The Tempo Constraint)

**Target File:** `tests/models/ssm/test_masr_mamba_rigor.py` **Context Files:**
`src/models/ssm/masr_mamba.py`

**Instructions:** Create a test named `test_masr_mamba_log_space_dt_bounds`.

1. Initialize `PyTorchMambaMASR`.
2. Create a completely zeroed input tensor
   `x = torch.zeros(2, 10, model.d_model)`. This isolates the bias of the linear
   layer by ensuring no data interference.
3. Intercept the values of the continuous time step `dt` exactly as the model
   computes it: `dt = F.softplus(model.dt_proj(x))`.
4. Assert that the minimum and maximum values of `dt` fall strictly between
   `0.0009` and `0.11` (allowing slight float epsilon margins for 0.001 and
   0.1). The network must prove it operates exclusively within biologically
   relevant temporal frequencies, proving the log-space initialization of the
   bias holds true under activation.
