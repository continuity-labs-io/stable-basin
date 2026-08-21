# Analytical Exactness Scales Test

### Prompt 1: Analytical Exactness (The Scales)

**Target File:** `tests/models/ssm/test_masr_mamba_rigor.py`
**Context Files:** `src/models/ssm/masr_mamba.py`

**Instructions:**
Create a strict, mathematically deterministic unit test named `test_mamba_masr_analytical_verification`. 
1. Do not use random tensors! Initialize `x`, `dt`, `mask`, `A`, `B`, `C`, and `D` with static, hardcoded scalar float values for a trivial shape (batch_size=1, seq_len=2, d_model=1, d_state=1). (e.g., A = -0.5, dt = 0.1).
2. Manually calculate the exact expected mathematical output of the Zero-Order Hold discretization for the 2 time steps using basic Python float arithmetic on a scratchpad inside the test.
3. Assert that `mamba_masr_reference_scan` matches your manual float calculations using `torch.testing.assert_close` with a stringent `atol=1e-6`. 
Prove that the PyTorch loop perfectly mirrors the discrete math. No sloppy approximations!
