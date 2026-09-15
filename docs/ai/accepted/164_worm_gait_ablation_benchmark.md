**Context Files to Load / Create:**
* `src/echo/models/primitives/ebm.py`
* `src/echo/models/observer.py` (Specifically the `PredictiveCodingGraph`)
* `src/data/behavior/celegans_gait_dataset.py`
* `src/echo/harness/echo_runner.py`
* `src/echo/benchmarks/06_worm_gait_decline.py` (Create)

**Task: Phase 3 Organism - The Worm Gait Integration Benchmark**
Please write the integration benchmark script `src/echo/benchmarks/06_worm_gait_decline.py`. This script acts as our ultimate integration test, proving the training loop converges on a biological limit cycle and validating that our EBMs can measure the thermodynamic flattening of a Waddington basin due to aging.

**Core Objectives:**

**1. The Ablation Orchestrator:**
* Build a `main()` execution script that instantiates two separate training runs via the `EchoRunner`.
* **Run A:** Train the `PredictiveCodingGraph` utilizing the `GaussianEBM` (The Laplace Baseline).
* **Run B:** Train a separate instance of the `PredictiveCodingGraph` utilizing the `PrecisionWeightedEBM` (Multimodal MLP).
* Ensure both runs utilize the `use_blanket_topology=True` flag for the partitioned biological state and train strictly on the "Young Worm" (Days 1-3) data split (or the synthetic 6D limit cycle fallback).

**2. The Thermodynamic Generalization Test (Aging):**
* After both models are fully trained on young worms, freeze their weights.
* Evaluate them both on the unseen "Old Worm" (Day 9+) dataset split (or a noisy synthetic fallback representing aging).
* Retrieve the `hessian_trace` metrics from the `HessianCurvatureTracker` for both the young (training) and old (evaluation) datasets.

**3. The 3-Panel Visual Proof:**
* Generate a cleanly formatted `matplotlib` figure (`worm_gait_ablation.png`) with three distinct panels to visually summarize the ablation:
    * **Panel A: The Limit Cycle (Data):** Plot a 2D or 3D phase-space projection of the raw 6D eigenworm trajectories, comparing the tight orbit of a Young worm vs. the erratic orbit of an Old worm (or predicted vs true trajectory). This establishes the behavioral decay we are modeling.
    * **Panel B: The Laplace Flatline (The Mathematical Proof):** Plot the Hessian Trace over time/states for the `GaussianEBM` evaluated on both young and old data. This *must* visually demonstrate a perfectly flat horizontal line to prove the calculus, curvature tracker, and positive-definite constraints are mathematically sound.
    * **Panel C: The Waddington Basin Flattening (The Biological Decline):** Plot the Hessian Trace distribution or timeline for the `PrecisionWeightedEBM` evaluated on the Young vs. Old data splits. This visualizes whether the learned neural network naturally captures the biological reality that the Waddington attractor basin becomes shallower (lower trace) and less precise in older organisms without ever being trained on old data.

**Constraints:**
* Use strict random seeding (e.g., `jax.random.PRNGKey(42)`, `torch.manual_seed(42)`) to ensure deterministic output for CI testing.
* Handle all matplotlib plotting cleanly without blocking execution (e.g., `plt.savefig()` instead of `plt.show()`, followed by `plt.close()`). Save the plot to an `outputs/benchmarks/` directory (create it if it doesn't exist).
* Ensure the script can run fully offline using the synthetic dataset generator from `CElegansGaitDataset` if local biological data files are not found, ensuring the benchmark never fails in CI.
* Use the standard Python `logging` module. Keep all log messages peaceful, precise, and practical (e.g., `logger.info("Evaluating frozen EBM models on Day 9+ biological population.")`). Avoid dramatic, capitalized, or emoji-laden print statements.
