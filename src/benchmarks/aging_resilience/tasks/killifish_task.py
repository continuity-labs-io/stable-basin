import random
import logging
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, Callable
from torch.utils.data import DataLoader

from src.benchmarks.aging_resilience.task_registry import AgingBenchmarkTask
from src.data.behavior.killifish_dataset import KillifishContinuousDataset
from src.data.datasets import JAXDictDataset

logger = logging.getLogger(__name__)

def numpy_collate(batch):
    """Collate function to return numpy arrays for JAX."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch).numpy()
    else:
        return np.array(batch)

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

        master_dataset = KillifishContinuousDataset(
            metadata_csv=metadata_csv,
            kinematics_dir=kinematics_dir,
            sequence_length=seq_len
        )

        young_samples = []
        old_samples = []

        for sample in master_dataset.samples:
            chrono_age = sample['chronological_age'].item()
            ult_life = sample['ultimate_lifespan'].item()
            
            if ult_life > 0:
                ratio = chrono_age / ult_life
                if ratio <= 0.3:
                    young_samples.append(sample)
                if ratio >= 0.7:
                    old_samples.append(sample)
                    
        all_samples = master_dataset.samples.copy()
        random.shuffle(all_samples)
        train_samples = all_samples[:int(len(all_samples)*0.8)]

        self.train_dataset = KillifishContinuousDataset.__new__(KillifishContinuousDataset)
        self.train_dataset.samples = train_samples

        young_eval_dataset = KillifishContinuousDataset.__new__(KillifishContinuousDataset)
        young_eval_dataset.samples = young_samples

        old_eval_dataset = KillifishContinuousDataset.__new__(KillifishContinuousDataset)
        old_eval_dataset.samples = old_samples

        return self.train_dataset, young_eval_dataset, old_eval_dataset

    def get_dataloaders(self, config: Dict[str, Any], d_state: int = None, batch_size: int = 2):
        train_raw, young_raw, old_raw = self.get_raw_datasets(config)

        train_ds = JAXDictDataset(train_raw, d_state)
        young_ds = JAXDictDataset(young_raw, d_state)
        old_ds = JAXDictDataset(old_raw, d_state)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=numpy_collate
        )
        young_loader = DataLoader(
            young_ds, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=numpy_collate
        )
        old_loader = DataLoader(
            old_ds, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=numpy_collate
        )

        return train_loader, young_loader, old_loader

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
        pass
