**Antigravity IDE: MeldLoss Ablation Sweep Directives**

**Context Files:**
`configs/sensor_fusion.yaml`

**Prompt 1:**
Please add a new task block named `loss_ablation` to `configs/sensor_fusion.yaml`. Set `csv_name` to `04_loss_ablation.csv` and `png_name` to `04_loss_ablation.png`. Configure `epochs` to 30, `train_seq_len` to 200, and `test_seq_len` to 500 to enforce the extrapolation evaluation. Restrict `models` to `["masr_mamba"]`, `densities` to `[0.1]`, and `seeds` to `[42, 43, 44]`. Add a new field `loss_fns` containing `["mse", "meld"]`.

---

**Context Files:**
`src/harness/sensor_fusion_predictor.py`

**Prompt 2:**
Modify the `SensorFusionPredictor` class forward pass to return a three-element tuple: `preds, h, reconstructed_t`. Before the routing logic, initialize `reconstructed_t = None`. For the condition where `ssm_type == "masr_mamba"`, extract `preds, reconstructed_t, h` directly from the underlying model, and apply `self.readout()` to `h` to generate the final `preds`. For all other model conditions, leave `reconstructed_t` as `None` and proceed with the existing forward pass logic. Ensure all variable names and logic remain calm and standard.

---

**Context Files:**
`src/harness/sensor_fusion_sweep.py`

**Prompt 3:**
Implement the loss function sweep in `sensor_fusion_sweep.py` using a calm, standard approach. First, import `MeldLoss` from `src.models.losses.meld_loss`. Inside `evaluate_model`, extract `loss_type` from `trial_config` (defaulting to "mse") and add a subdued informational log (e.g., `logger.info(f"Initialized training with loss type: {loss_type}")`). Initialize both `nn.MSELoss()` and `MeldLoss(alpha=1.0, beta=0.1, gamma=0.5, L=1.5)`. 

Update the training loop's forward pass unpacking to accept the new three-element tuple: `preds, _, reconstructed_t = model(x_raw, mask)`. If `loss_type == "meld"` and `reconstructed_t` is not None, route the optimization through `MeldLoss`. Calculate `state_t`, `target_t_plus_1`, `pred_t_plus_1`, and `recon_t` by slicing off the terminal frames, and approximate `delta_x` as a tensor of ones with shape `(batch_size, 1)`. If the conditions are not met, fall back to the standard MSE calculation. Finally, in `main()`, add `"loss_type": tune.grid_search(task_config.get("loss_fns", ["mse"]))` to the `search_space` dictionary.

---

**Dependencies:**
No novel software dependencies are required for these updates.
