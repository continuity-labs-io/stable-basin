# Prompt 1: The Strategy Interface & Worm Implementation

I am refactoring my machine learning benchmark pipeline to use the Strategy Pattern so it can support multiple datasets (like Worms and Killifish). 

1. Create a new directory `src/benchmarks/aging_resilience/tasks/`.
2. Create `src/benchmarks/aging_resilience/task_registry.py` with an abstract base class `AgingBenchmarkTask(abc.ABC)`. Define the following abstract properties/methods:
   - `@property def d_sensory(self) -> int`
   - `@abc.abstractmethod def get_dataloaders(self, config, d_state: int = None, batch_size: int = 2)`
   - `@abc.abstractmethod def get_raw_datasets(self, config)`
   - `@abc.abstractmethod def apply_dataset_change(self, trajectory, change_fn: callable, **kwargs)`
   - `@abc.abstractmethod def compute_domain_metrics(self, trajectory) -> dict`
   - `@abc.abstractmethod def render_animation(self, young_data, old_data, output_path: str, **kwargs)`
   - Create a factory function `get_benchmark_task(config: dict) -> AgingBenchmarkTask` that returns `WormGaitTask()` when `config.get("dataset", {}).get("name", "worm_gait") == "worm_gait"`.
3. Create `src/benchmarks/aging_resilience/tasks/worm_task.py` implementing `WormGaitTask(AgingBenchmarkTask)`. 
   - `d_sensory` returns 6.
   - `get_raw_datasets` encapsulates the initialization of `RealEigenwormDataset` and `SyntheticWormMockDataset` (with fallback logic).
   - `get_dataloaders` uses `get_raw_datasets` to initialize `JAXDictDataset`s and PyTorch `DataLoader`s.
   - `apply_dataset_change` should execute and return `change_fn(trajectory, **kwargs)` if `change_fn` is not None, else return `trajectory`.
   - `compute_domain_metrics` encapsulates `amplitude_residual_stats` and the `SpectralMetrics` peak frequency logic currently inside `01_worm_gait_baseline_metrics.py`.
   - `render_animation` uses `create_worm_gait_animation` from `10_animate_worm_gait.py`.
