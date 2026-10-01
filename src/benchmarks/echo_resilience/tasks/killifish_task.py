import random
import logging
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, Callable
from torch.utils.data import DataLoader

from src.benchmarks.echo_resilience.task_registry import AgingBenchmarkTask
from src.data.behavior.killifish_dataset import KillifishContinuousDataset
from src.data.datasets import JAXDictDataset

logger = logging.getLogger(__name__)


class KillifishTask(AgingBenchmarkTask):
    def __init__(self):
        super().__init__()
        self.train_dataset = None
        
    @property
    def d_sensory(self) -> int:
        if self.train_dataset is None or len(self.train_dataset.samples) == 0:
            raise ValueError("Datasets not loaded. Call get_raw_datasets first.")
        # trajectory_chunk is [seq_len, features]
        return self.train_dataset.samples[0]['trajectory_chunk'].shape[-1]

    def get_raw_datasets(self, config: Dict[str, Any]):
        dataset_config = config.get("dataset", {})
        metadata_csv = dataset_config.get("metadata_csv", "data/killifish/data/a1_20241119/26441580/df_reformat_10_20241119.csv")
        kinematics_dir = dataset_config.get("kinematics_dir", "data/killifish/data/p3_20230526/test/standardization/")
        seq_len = dataset_config.get("sequence_length", 100)
        max_samples = dataset_config.get("max_samples", None)

        master_dataset = KillifishContinuousDataset(
            metadata_csv=metadata_csv,
            kinematics_dir=kinematics_dir,
            sequence_length=seq_len,
            max_samples=max_samples
        )

        from src.benchmarks.echo_resilience.task_registry import build_cohorts
        
        def young_fn(s):
            return (s['chronological_age'].item() / s['ultimate_lifespan'].item()) <= 0.5
            
        def old_fn(s):
            return (s['chronological_age'].item() / s['ultimate_lifespan'].item()) > 0.5
            
        seed = config.get("experiment", {}).get("seed", 42)
        train_s, val_s, young_s, old_s = build_cohorts(master_dataset.samples, "killifish", seed, young_fn, old_fn)

        self.train_dataset = KillifishContinuousDataset.__new__(KillifishContinuousDataset)
        self.train_dataset.samples = train_s
        
        val_dataset = KillifishContinuousDataset.__new__(KillifishContinuousDataset)
        val_dataset.samples = val_s

        young_eval_dataset = KillifishContinuousDataset.__new__(KillifishContinuousDataset)
        young_eval_dataset.samples = young_s

        old_eval_dataset = KillifishContinuousDataset.__new__(KillifishContinuousDataset)
        old_eval_dataset.samples = old_s

        return self.train_dataset, val_dataset, young_eval_dataset, old_eval_dataset

    def get_dataloaders(self, config: Dict[str, Any], d_state: int = None, batch_size: int = 2):
        train_raw, val_raw, young_raw, old_raw = self.get_raw_datasets(config)

        train_ds = JAXDictDataset(train_raw, d_state)
        val_ds = JAXDictDataset(val_raw, d_state)
        young_ds = JAXDictDataset(young_raw, d_state)
        old_ds = JAXDictDataset(old_raw, d_state)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=True
        )
        young_loader = DataLoader(
            young_ds, batch_size=batch_size, shuffle=False, drop_last=True
        )
        old_loader = DataLoader(
            old_ds, batch_size=batch_size, shuffle=False, drop_last=True
        )

        return train_loader, val_loader, young_loader, old_loader

    def apply_dataset_change(self, trajectory, change_fn: Optional[Callable], **kwargs):
        if change_fn:
            return change_fn(trajectory, **kwargs)
        return trajectory

    def compute_domain_metrics(self, trajectory) -> dict:
        if isinstance(trajectory, torch.Tensor):
            variance = torch.var(trajectory).item()
        else:
            variance = float(np.var(trajectory))
        return {"variance": variance}

    def render_animation(self, young_data, old_data, output_path: str, **kwargs):
        logger.warning(f"Skipping animation for Killifish dataset. Output path: {output_path}")

    @property
    def cohort_labels(self):
        return ("<= 50% of lifespan", "> 50% of lifespan")
