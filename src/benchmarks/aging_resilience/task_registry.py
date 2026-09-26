import abc
from typing import Dict, Any, Tuple, Optional, Callable

class AgingBenchmarkTask(abc.ABC):
    """Abstract base class for biological aging resilience benchmark tasks."""

    @property
    @abc.abstractmethod
    def d_sensory(self) -> int:
        """Return the dimensionality of the sensory input for this dataset."""
        pass

    @abc.abstractmethod
    def get_dataloaders(self, config: Dict[str, Any], d_state: int = None, batch_size: int = 2):
        """Return PyTorch DataLoaders for train (young), eval (young), and eval (old)."""
        pass

    @abc.abstractmethod
    def get_raw_datasets(self, config: Dict[str, Any]):
        """Return the underlying raw PyTorch Datasets for train (young), eval (young), and eval (old)."""
        pass

    @abc.abstractmethod
    def apply_dataset_change(self, trajectory, change_fn: Optional[Callable], **kwargs):
        """Apply an optional transformation (like an intervention or lambda change) to a trajectory."""
        pass

    @abc.abstractmethod
    def compute_domain_metrics(self, trajectory) -> dict:
        """Compute task-specific biological/domain metrics for a given trajectory."""
        pass

    @abc.abstractmethod
    def render_animation(self, young_data, old_data, output_path: str, **kwargs):
        """Render a task-specific side-by-side animation of young vs old/intervention behavior."""
        pass

def get_benchmark_task(config: Dict[str, Any]) -> AgingBenchmarkTask:
    """Factory function to instantiate the correct task based on the config."""
    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name", "worm_gait")
    
    if dataset_name == "worm_gait":
        # Import inside here to prevent circular imports if the task imports from registry
        from src.benchmarks.aging_resilience.tasks.worm_task import WormGaitTask
        return WormGaitTask()
    elif dataset_name == "killifish":
        from src.benchmarks.aging_resilience.tasks.killifish_task import KillifishTask
        return KillifishTask()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
