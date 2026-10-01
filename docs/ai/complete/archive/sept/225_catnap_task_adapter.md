We are implementing Phase 2 of "Project Catnap": building the `AgingBenchmarkTask` adapter.

1. **Create the Task Adapter:**
   - Create a new file `src/benchmarks/aging_resilience/tasks/catnap_task.py`.
   - Import `AgingBenchmarkTask` from `src.benchmarks.aging_resilience.tasks.task_registry`.
   - Import `CatnapContinuousDataset` from `src.data.behavior.catnap_dataset` (wrap in a try/except ImportError for safety).
   - Import `JAXDictDataset` from `src.data.datasets` and `DataLoader` from `torch.utils.data`.
   - Import `logging` and setup `logger = logging.getLogger(__name__)`.
   - Import `numpy as np`.
   - Create `class CatnapTask(AgingBenchmarkTask):`
   - Implement `get_raw_datasets(self, config) -> tuple`:
     - Extract `h5_path = config.get("dataset", {}).get("h5_path", "data/catnap/trace_features.h5")`.
     - Extract `seq_len = config.get("dataset", {}).get("ebm_seq_len", 10)`.
     - Instantiate `train_data = CatnapContinuousDataset(h5_path=h5_path, sequence_length=seq_len, cohort="train")`.
     - Instantiate `young_data = CatnapContinuousDataset(h5_path=h5_path, sequence_length=seq_len, cohort="young")`.
     - Instantiate `old_data = CatnapContinuousDataset(h5_path=h5_path, sequence_length=seq_len, cohort="old")`.
     - Return `(train_data, young_data, old_data)`.
   - Implement `@property def d_sensory(self) -> int`:
     - Return `309` (the known number of features in Catnap).
   - Implement `get_dataloaders(self, config, d_state: int, batch_size: int) -> tuple`:
     - Call `self.get_raw_datasets(config)` to get the raw datasets.
     - Wrap each in `JAXDictDataset(raw_dataset, d_state)`.
     - Return three `DataLoader` instances (train loader `shuffle=True`, eval loaders `shuffle=False`).
   - Implement `apply_dataset_change(self, trajectory, change_fn=None, **kwargs)`:
     - Just return `trajectory`. No synthetic degradation is needed because we have true longitudinal aging data.
   - Implement `compute_domain_metrics(self, trajectory) -> dict`:
     - Return a basic dictionary: `{"variance_norm": float(np.var(trajectory))}`.
   - Implement `render_animation(self, young_data, old_data, output_path: str, **kwargs)`:
     - Just log a warning: `logger.info("Catnap animation not yet implemented. Skipping.")`

2. **Update the Factory Registry:**
   - Open `src/benchmarks/aging_resilience/tasks/task_registry.py`.
   - In `get_benchmark_task(config)`, add an `elif dataset_name == "catnap":` branch.
   - Inside the branch, import and return `CatnapTask()`.
