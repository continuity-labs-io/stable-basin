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
        """Return PyTorch DataLoaders for train, val, eval (young), and eval (old)."""
        pass

    @abc.abstractmethod
    def get_raw_datasets(self, config: Dict[str, Any]):
        """Return the underlying raw PyTorch Datasets for train, val, eval (young), and eval (old)."""
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

    @property
    @abc.abstractmethod
    def cohort_labels(self) -> Tuple[str, str]:
        """Return the labels for the young/old cohorts."""
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
    elif dataset_name == "catnap":
        from src.benchmarks.aging_resilience.tasks.catnap_task import CatnapTask
        return CatnapTask()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def build_cohorts(samples, dataset_name, seed, young_fn, old_fn):
    import random
    import os
    import json
    
    random.seed(seed)
    individuals = {}
    for s in samples:
        ind_id = s.get('individual_id', None)
        if ind_id is None:
            continue
        if ind_id not in individuals:
            individuals[ind_id] = []
        individuals[ind_id].append(s)
        
    all_inds = list(individuals.keys())
    random.shuffle(all_inds)
    
    n_train_inds = int(len(all_inds) * 0.8)
    train_inds = all_inds[:n_train_inds]
    held_out_inds = all_inds[n_train_inds:]
    
    n_val_inds = int(len(train_inds) * 0.2)
    val_inds = train_inds[:n_val_inds]
    pure_train_inds = train_inds[n_val_inds:]
    
    train_samples = []
    for ind in pure_train_inds:
        train_samples.extend([s for s in individuals[ind] if young_fn(s)])
        
    val_samples = []
    for ind in val_inds:
        val_samples.extend([s for s in individuals[ind] if young_fn(s)])
        
    eval_young_samples = []
    eval_old_samples = []
    for ind in held_out_inds:
        eval_young_samples.extend([s for s in individuals[ind] if young_fn(s)])
        eval_old_samples.extend([s for s in individuals[ind] if old_fn(s)])
        
    if not eval_young_samples or not eval_old_samples:
        raise ValueError(f"Empty cohorts! Young: {len(eval_young_samples)}, Old: {len(eval_old_samples)}")
        
    dataset_name = config.get("dataset", {}).get("name", "worm_gait")
    output_dir = f"output/benchmarks/aging_resilience/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    cohorts_dict = {
        "train": {"individuals": len(pure_train_inds), "chunks": len(train_samples), "ids": pure_train_inds},
        "val": {"individuals": len(val_inds), "chunks": len(val_samples), "ids": val_inds},
        "eval_young": {"individuals": len(held_out_inds), "chunks": len(eval_young_samples), "ids": held_out_inds},
        "eval_old": {"individuals": len(held_out_inds), "chunks": len(eval_old_samples), "ids": held_out_inds},
    }
    with open(f"{output_dir}/{dataset_name}_cohorts.json", "w") as f:
        json.dump(cohorts_dict, f, indent=2)
        
    assert set(pure_train_inds).isdisjoint(set(held_out_inds))
    assert set(val_inds).isdisjoint(set(held_out_inds))
    assert set(pure_train_inds).isdisjoint(set(val_inds))
    
    return train_samples, val_samples, eval_young_samples, eval_old_samples
