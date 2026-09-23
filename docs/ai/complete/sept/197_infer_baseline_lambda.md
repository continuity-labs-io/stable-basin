**Context & Goal:**
We need to eliminate the "magic number" $\lambda_A = 0.2$ representing the pathological old worm state in our Worm Gait intervention experiment. We want to formally infer the true biological $\lambda$ directly from the empirical data using Single-Parameter Variational Inference.

**Your Task:**
Create a new benchmark script: `src/benchmarks/13_infer_biological_lambda.py`

**Requirements:**
1. **The Data:** Load the Old Worm dataset (`RealEigenwormDataset` with `is_aged=True`, `seq_len=100`). Use `JAXDictDataset` and a `DataLoader` (batch_size=8).
2. **The Model:** Load the trained `output/echo/benchmarks/06_worm_gait_decline_trained_engine.eqx` model (the `PrecisionWeightedEBM` version). Freeze all of its weights so the topographical prior ($\nabla E(x)$) cannot change.
3. **The Parameter:** Define a single trainable JAX scalar parameter: `log_lambda` (initialized to `0.0`, so $\lambda = \exp(0) = 1.0$). Exponentiating `log_lambda` ensures the multiplier never goes negative.
4. **The Objective:** 
   - Write a short custom training loop.
   - The loss function should be the 1-step forecasting MSE: Given `x_init` and `s_true`, simulate forward for `n_steps=1` (or short sequences) but multiply the gradients of the EBM by `jnp.exp(log_lambda)`. Compute the MSE against the next frame in `s_true`.
   - Use `optax.adam` (lr=0.01) to optimize *only* the `log_lambda` parameter. Filter out the frozen graph weights using `eqx.partition`.
5. **Optimization:** Run the optimization loop over the Old Worm dataloader for a few epochs (e.g., 20-50) until `log_lambda` converges.
6. **Output:** 
   - Print the final converged value of $\lambda$ (`jnp.exp(log_lambda)`).
   - Save this float value to a JSON file: `output/echo/benchmarks/13_inferred_biological_lambda.json`.
7. **Pipeline Update:** Add `.PHONY: worm-gait-infer-lambda` to the `Makefile` and execute the script.

Please execute this and report the final inferred biological $\lambda$! We expect it to converge somewhere below 1.0, quantifying exactly how much the Waddington basin flattens during aging.
