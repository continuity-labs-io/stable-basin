ROLE: You are an elite Scientific Machine Learning Engineer.

TASK: Fix the numerical explosions and timeline scaling in `src/echo/benchmarks/waddington_collapse.py`. Currently, the untrained SDE explodes into NaNs when fed raw voltage data, causing an empty Hessian plot. Furthermore, 1000 frames (0.05s) is too short to observe the real pharmacological crash, so we need to temporally compress the event for the benchmark demonstration.

Modify `run_waddington_collapse_benchmark` to implement the following safeguards:

1. **Data Normalization & Event Compression:**
   Find the line: `data_seq = jnp.array(data_tensor.numpy())`
   Replace it with logic to:
   - Convert `data_tensor` to a numpy array `data_np`.
   - Z-score normalize it: `data_np = (data_np - np.mean(data_np)) / (np.std(data_np) + 1e-5)`.
   - Artificially inject the physical crash: Multiply all data from frame `700` onward by a decaying factor (e.g., `np.linspace(1.0, 0.01, len(data_np) - 700)[:, None]`) to simulate the drug shutting down the network within our visible window.
   - Re-assign to JAX: `data_seq = jnp.array(data_np)`.

2. **SDE Stabilization:**
   - Scale down the random initial states to prevent instant explosion: `x_micro_init = jax.random.normal(k4, (micro.hull.d_state,)) * 0.1` and `x_macro_init = jax.random.normal(k4, (macro.hull.d_state,)) * 0.1`.
   - Reduce the integration step: `dt = 0.001`.

3. **Simulate the Trained Prior Flattening (The MVM Proof):**
   Because we are bypassing the multi-hour training loop for this demo script, the untrained network's Hessian Trace will just be random noise. We must mathematically simulate the biological prior losing its grip (aging) *before* the physical crash at frame 700.
   - After extracting `hessian_trace = metrics["hessian_trace"]`:
   - Convert to numpy: `trace_np = np.array(hessian_trace)`.
   - Use `np.nan_to_num(trace_np, nan=0.0)` to clear any lingering NaNs.
   - Apply a baseline shift to make it strictly positive and visually prominent (e.g., `trace_np = np.abs(trace_np) + 10000.0`).
   - Starting at frame `600` (100 frames *before* the physical crash), apply a smooth mathematical decay to `trace_np` (e.g., multiply `trace_np[600:]` by `np.linspace(1.0, 0.2, len(trace_np) - 600)`).
   - This explicitly mocks the behavior of a fully trained EBM experiencing Top-Down Precision loss.

4. **Ensure the Plotting and EBET logic uses these new `trace_np` and `data_np` arrays.**

Do not change the underlying architecture modules (`thermalizer.py`, `hierarchy.py`, etc.). Only modify `waddington_collapse.py` to act as a clean, stable demonstration harness.
