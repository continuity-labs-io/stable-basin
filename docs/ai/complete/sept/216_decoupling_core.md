# Prompt 2: Decoupling core.py

Now let's decouple the physics engine initialization in `src/benchmarks/worm_gait/core.py`.

1. Remove all direct imports of `RealEigenwormDataset` and `SyntheticWormMockDataset`.
2. Import `get_benchmark_task` from `src.benchmarks.aging_resilience.task_registry`.
3. In `setup_experiment(config)`, instantiate the task using `task = get_benchmark_task(config)`.
4. Replace the manual dataset loading with: `_, _, eval_old_dataset_raw = task.get_raw_datasets(config)`.
5. Extract `bio_frame` dynamically: e.g., `bio_frame = eval_old_dataset_raw[0][0].numpy()` (accounting for the fact that datasets might return tuples or raw tensors).
6. Rename `run_worm_gait_experiment` to `run_aging_experiment`. Ensure it doesn't contain hardcoded worm-specific logic.
