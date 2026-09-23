**Context & Goal:** To prove that biological aging is a thermodynamic geometry
problem and NOT just a standard sequence prediction problem, we need to show
that standard continuous-time deep learning models fail to detect the
age-related shift.

We will start by testing our existing PyTorch `BaselineSSM`.

**Your Task:** Create a new benchmark script:
`src/benchmarks/11_worm_gait_aging_ssm.py`

**Requirements:**
1. **The Architecture:** 
   - Instantiate the model using
     `SensorFusionPredictor(ssm_type="zero_padded_ssm", modality_dims=[6],
     d_model=64, out_dim=6)` from `src.harness.sensor_fusion_predictor`. 
   - This automatically wraps our PyTorch `BaselineSSM` with the proper
     input/output projections for the 6D worm data.
2. **The Data:**
   - Load the `RealEigenwormDataset` with `seq_len=100` from
     `src.data.behavior.celegans_gait_dataset`.
   - *Train set:* Young worms (`data/worm/EigenWorms_TRAIN.ts`,
     `is_aged=False`).
   - *Young Eval set:* Young worms (`data/worm/EigenWorms_TEST.ts`,
     `is_aged=False`).
   - *Old Eval set:* Old worms (`data/worm/EigenWorms_TEST.ts`, `is_aged=True`).
   - Use standard PyTorch `DataLoader`s (batch_size=8).
3. **The Objective (Next-Step Forecasting):** 
   - Standard SSMs don't have energy landscapes; they use Mean Squared Error
     (MSE). 
   - Train the model on the Train set to predict the next sensory frame.
   - For a sequence `x_raw` (shape `[batch, seq_len, 6]`), the target is `y_true
     = x_raw[:, 1:, :]` and the prediction is `pred = preds[:, :-1, :]`.
   - Use `mask = torch.ones_like(x_raw[:, :, :1])` to pass to the model.
   - Train with `torch.optim.AdamW` (lr=1e-3) for 50 epochs.
4. **Evaluation (The Shift Test):**
   - Run the trained model in `eval()` mode over both the Young Eval set and the
     Old Eval set.
   - Calculate the sequence-wise MSE (one scalar MSE value for each
     sequence/trajectory in the dataset). This will give you an
     array/distribution of MSE scores for Young and Old.
5. **Statistical Output:**
   - Calculate the KS-statistic, p-value (`scipy.stats.ks_2samp`), and Cohen's
     $d$ (`pingouin.compute_effsize`) between the distribution of Young MSEs and
     the distribution of Old MSEs.
6. **Serialization & Visualization:** 
   - Save the metrics to `output/echo/benchmarks/11_baseline_ssm_metrics.json`.
   - Generate a plot `output/echo/benchmarks/11_baseline_ssm_mse.png` showing
     two overlapping histograms (or KDE plots): the MSE distribution for Young
     worms and the MSE distribution for Old worms.
7. **Pipeline Update:** 
   - Add a `.PHONY: worm-gait-ssm` target to the `Makefile` executing this
     script, and append it to the `worm-gait-experiments` chain.

**Constraints:**
- This script is strictly PyTorch-based (no JAX/Equinox needed here since
  `BaselineSSM` is a PyTorch `nn.Module`). Do not import the EchoTrainer or JAX
  engines.

Please execute this and report the final Cohen's $d$ and p-value between the
Young and Old MSE distributions! We expect the effect size to be negligible
compared to our EBM.

Include this experiment in the overall makefile target.
