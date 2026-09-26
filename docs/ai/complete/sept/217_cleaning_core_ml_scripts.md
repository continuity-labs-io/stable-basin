# Prompt 3: Cleaning the Core ML Scripts (02 through 09)

Refactor the core ML scripts in `src/benchmarks/worm_gait/`: `02_worm_gait_aging_ssm.py` through `09_pharmacological_translation.py`.

For EACH script:
1. Remove all direct imports of `RealEigenwormDataset` and `SyntheticWormMockDataset`.
2. Load the task: `task = get_benchmark_task(config)` (If a script doesn't natively load a config, create a dummy dict: `config = {"dataset": {"name": "worm_gait"}}`).
3. Replace manual dataset and dataloader creation with `train_loader, young_eval_loader, old_eval_loader = task.get_dataloaders(config, d_state, batch_size)`.
4. Replace hardcoded dimensions (like `modality_dims=[6]`, `out_dim=6`) with `[task.d_sensory]` and `task.d_sensory`.
5. Update `wandb.init` to use `project="stable_basin_aging", group=config.get("dataset", {}).get("name", "worm_gait")`.
