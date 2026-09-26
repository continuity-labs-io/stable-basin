Now let's add the new Killifish dataset and its Task adapter.

0. In `task_registry.py`, update factory function `get_benchmark_task(config: dict) -> AgingBenchmarkTask`. 
   - Add condition: If "killifish", dynamically import and return `KillifishTask()`.
1. Save the `KillifishContinuousDataset` code I provided into `src/data/behavior/killifish_dataset.py`.
2. Modify `KillifishContinuousDataset.__getitem__` to return ONLY `sample['trajectory_chunk']`. This ensures it perfectly matches `JAXDictDataset`'s API expectations (which expects a single tensor).
3. Create `src/benchmarks/aging_resilience/tasks/killifish_task.py` implementing `KillifishTask(AgingBenchmarkTask)`.
4. Implement `get_raw_datasets(self, config)`: 
   - Instantiate a master `KillifishContinuousDataset`.
   - Iterate over `dataset.samples` and split them based on lifespan metadata:
     - `young_samples`: where `chronological_age / ultimate_lifespan <= 0.3`
     - `old_samples`: where `chronological_age / ultimate_lifespan >= 0.7`
     - `train_samples`: a random 80% split of ALL samples.
   - Create three new empty `KillifishContinuousDataset` instances (bypassing `__init__` logic by using `__new__` or just overwriting). Overwrite their `.samples` attributes with these three lists. Return `(train, young, old)`.
5. Implement `d_sensory`: dynamically return the feature dimension by checking `self.train_dataset.samples[0]['trajectory_chunk'].shape[-1]`.
6. Implement `get_dataloaders` identically to `WormGaitTask` (wrapping in `JAXDictDataset` and `DataLoader`).
7. For `apply_dataset_change`, `compute_domain_metrics`, and `render_animation`, provide safe, generic stubs for now (e.g., computing basic variance for metrics, returning `change_fn(trajectory, **kwargs)` if provided else the trajectory, and skipping animation with a `logger.warning`).
8. **Config & Makefile**:
   - Create `configs/killifish_experiments.yaml` with `dataset.name: "killifish"` and add the paths to your Killifish CSV/HDF5 data.

Testing
- ensure `make aging-resilience-ebm DATASET=killifish_experiments` works end to end
