**Context & Goal:**
We are executing Part 1 of our paper finalization strategy: The Clinical Dose-Response Sweep. 
To prove our therapeutic intervention ($\lambda = 5.0$) isn't just an arbitrary "magic number," we need to sweep the precision-injection parameter ($\lambda$) to find the algorithmic $EC_{50}$ (the exact threshold where the Waddington basin steepens and the limit cycle is rescued).

Preliminary testing showed rescue happens around $\lambda = 0.3$. We need to formalize this into a benchmark.

**Your Task:**
Create a new benchmark script: `src/benchmarks/09_worm_gait_lambda_sweep.py`

**Requirements:**
1. **The Physics Engine:** Reuse the SDE integration and EBM setup logic from `src/benchmarks/07_worm_gait_intervention.py`. Load the trained `output/echo/benchmarks/06_worm_gait_decline_trained_engine.eqx` weights and the Old worm dataset (`EigenWorms_TEST.ts`, `is_aged=True`).
2. **The Sweep:** Iterate the SDE rollout (Run B) over the following `lambda` values: `[0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0]`. (Use `lambda_A = 0.2` for the baseline control).
3. **Metrics Extraction:** For each $\lambda$, compute the Effective Hessian Trace using `batch_calculate_curvature` on the macro EBM energy function. 
4. **Statistical Output:** Calculate the Cohen's $d$ effect size between the Baseline trace ($\lambda=0.2$) and the Rescued trace for *each* lambda step.
5. **Serialization:** Save the results to `output/echo/benchmarks/09_lambda_sweep_metrics.json`. The JSON should map each $\lambda$ value to its resulting mean trace and Cohen's $d$.
6. **Visualization:** Generate a classic pharmacological Dose-Response plot (`output/echo/benchmarks/09_lambda_dose_response.png`). 
    * X-axis: $\lambda$ value (Log scale is often preferred, but linear is fine if the curve is clear).
    * Y-axis: Mean Effective Hessian Trace.
    * Add a horizontal dashed red line representing the Baseline ($\lambda=0.2$) trace.
    * The resulting plot should look like a classic sigmoidal pharmacological receptor saturation curve.
7. **Pipeline Update:** Add a `.PHONY: worm-gait-sweep` target to the `Makefile` (executing this new script) and append it to the `worm-gait-experiments` chain.

**Constraints:**
- Do not modify the core `src/echo/` physics engine.
- You can import `setup_experiment` and `simulate_sde` directly from `src.benchmarks.07_worm_gait_intervention` if possible to keep the codebase DRY, or duplicate the minimal necessary boilerplate.

Please execute this and let me know when the JSON and PNG are successfully generated!
