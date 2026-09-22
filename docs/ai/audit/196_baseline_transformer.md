**Context & Goal:**
We are executing the final part of our "Reviewer 2 Baselines". We just built the continuous-time `BaselineSSM` test, and now we must test the absolute industry standard: a discrete-time Transformer. 

By showing that a standard causal Transformer also fails to meaningfully separate the young and old worms via Mean Squared Error (MSE), we conclusively prove that standard deep learning sequence predictors are fundamentally blind to continuous thermodynamic phase transitions. The biological decay can only be seen geometrically through our continuous energy-based Waddington basin.

**Your Task:**
Create a new benchmark script: `src/benchmarks/12_worm_gait_aging_transformer.py`

**Requirements:**
1. **The Architecture:** 
   - Instantiate the model using `SensorFusionPredictor(ssm_type="causal_transformer", modality_dims=[6], d_model=64, out_dim=6)` from `src.harness.sensor_fusion_predictor`. 
   - This automatically wraps our PyTorch `BaselineTransformer` with the proper input/output projections for the 6D worm data.
2. **The Data:**
   - Load the `RealEigenwormDataset` with `seq_len=100` from `src.data.behavior.celegans_gait_dataset`.
   - *Train set:* Young worms (`data/worm/EigenWorms_TRAIN.ts`, `is_aged=False`).
   - *Young Eval set:* Young worms (`data/worm/EigenWorms_TEST.ts`, `is_aged=False`).
   - *Old Eval set:* Old worms (`data/worm/EigenWorms_TEST.ts`, `is_aged=True`).
   - Use standard PyTorch `DataLoader`s (batch_size=8).
3. **The Objective (Next-Step Forecasting):** 
   - Train the model on the Train set to predict the next sensory frame.
   - For a sequence `x_raw` (shape `[batch, seq_len, 6]`), the target is `y_true = x_raw[:, 1:, :]` and the prediction is `pred = preds[:, :-1, :]`.
   - Use `mask = torch.ones_like(x_raw[:, :, :1])` to pass to the model.
   - Train with `torch.optim.AdamW` (lr=1e-3) for 50 epochs.
4. **Evaluation (The Shift Test):**
   - Run the trained model in `eval()` mode over both the Young Eval set and the Old Eval set.
   - Calculate the sequence-wise MSE (one scalar MSE value for each sequence/trajectory in the dataset).
5. **Statistical Output:**
   - Calculate the KS-statistic, p-value (`scipy.stats.ks_2samp`), and Cohen's $d$ (`pingouin.compute_effsize`) between the distribution of Young MSEs and the distribution of Old MSEs.
6. **Serialization & Visualization:** 
   - Save the metrics to `output/echo/benchmarks/12_baseline_transformer_metrics.json`.
   - Generate a plot `output/echo/benchmarks/12_baseline_transformer_mse.png` showing two overlapping histograms (or KDE plots): the MSE distribution for Young worms and the MSE distribution for Old worms.
7. **Pipeline Update:** 
   - Add a `.PHONY: worm-gait-transformer` target to the `Makefile` executing this script, and append it to the `worm-gait-experiments` chain.

**Constraints:**
- This script is strictly PyTorch-based. Do not import JAX/Equinox engines.
- Ensure the model and data are placed on the optimal device using `get_optimal_device()`.
- To keep the codebase DRY, if you abstracted the PyTorch train/eval loop in the SSM script, reuse it here.

Please execute this and report the final Cohen's $d$ and p-value between the Young and Old MSE distributions!
