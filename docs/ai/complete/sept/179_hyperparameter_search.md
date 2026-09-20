**Context & Goal:**
We have successfully cleared the math audits and our physics engine is now thermodynamically sound. However, in our Wormgate aging benchmark (`src/echo/benchmarks/06_worm_gait_aging_ebm.py`), the visual separation between the Young and Old histograms in Panel C is practically non-existent. While the KS test proves they are distinct, the effect size is tiny (Cohen's d = 0.0019). We strongly suspect the `PrecisionWeightedEBM` is underfitting the biological complexity of the manifold.

We need to "sell" this visually for the paper. The "Young" worm histogram should have a noticeably sharper, concentrated Effective Hessian Trace (representing a tight, steep limit cycle), while the "Old" worm histogram should be wider and flattened (representing a decaying Waddington basin). 

**Your Task:**
Please write a Python script using the `optuna` library to run an intelligent, time-bounded hyperparameter search (1-hour time budget) over `configs/worm_gait_ebm.yaml`. Your objective is to find the configuration that maximizes the absolute value of `cohens_d` (and `wasserstein_distance`) between the Young and Old trace distributions for the Precision-Weighted EBM, yielding visually distinct histograms.

**Parameters to Explore (Optuna Search Space):**
1. `observer.macro.ebm_hidden_size`: Categorical [16, 32, 64, 128].
2. `observer.micro.ebm_hidden_size`: Categorical [128, 256].
3. `observer.macro.ebm_depth` and `observer.micro.ebm_depth`: Categorical [2, 3].
4. `optimization.max_epochs`: Categorical [2000, 4000]. (Avoid excessively high epochs like 6000+ for this small dataset).
5. `optimization.learning_rate`: Log-uniform or categorical [1e-5, 5e-5, 1e-4].

**Budget:**
- **Time Limit:** 1 hour (`timeout=3600` in Optuna). The script should automatically terminate the search after 1 hour and report the best configuration found within that timeframe.

**Deliverables:**
1. Find the best configuration that provides striking visual separation.
2. Update `configs/worm_gait_ebm.yaml` permanently with these winning parameters.
3. Run the final benchmark to generate the updated `06_worm_gait_decline_ablation.png` and `06_worm_gait_metrics.json`.
4. Report the new Cohen's d and confirm the histograms look scientifically compelling!

