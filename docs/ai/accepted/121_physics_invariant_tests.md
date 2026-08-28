Create a new pytest file `tests/test_physics_invariants.py` to write synthetic unit tests that mathematically prove our core physics operations. Add two tests:

1. **Test Solenoidal Flow Antisymmetry**: 
Import `SolenoidalFlow` from `src.echo.physics.solenoidal`. Initialize it with a random JAX PRNGKey and `d_state=16`. Extract its `Q` matrix. Mathematically assert that Q is perfectly skew-symmetric (`jnp.allclose(Q, -Q.T)`) and that for a random state vector `v`, the quadratic form `v.T @ Q @ v` is exactly 0 within a tight float tolerance (e.g., `atol=1e-6`).

2. **Test MambaLRP Relevance Conservation**: 
Import `MambaLRPEpsilon` from `src.icebox.metrics.mamba_lrp`. Mock a dummy PyTorch `nn.Module` that has `.fusion.W_proj`, `.readout`, and a `.get_hidden_states()` method returning a dummy tensor, so `MambaLRPEpsilon` can initialize. Generate a random input tensor `x` (shape: [1, 10, 5]). Run `.attribute(x, target_time_step=9)`. Assert that the total sum of the input attributions `R_x.sum()` is equal to the total sum of the target prediction `preds[:, 9, :].sum()` within a 1% relative tolerance margin. Ensure the test runs entirely on CPU without requiring external data.
