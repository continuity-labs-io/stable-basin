import logging
import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
from torch.utils.data import DataLoader

from src.benchmarks.aging_resilience.task_registry import AgingBenchmarkTask
from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset, SyntheticWormMockDataset
from src.data.datasets import JAXDictDataset
from src.metrics.spectral import SpectralMetrics
from src.data.behavior.synthetic_aging import amplitude_residual_stats
from src.utils.animation import create_worm_gait_animation

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
    else:
        return np.array(batch)


class WormGaitTask(AgingBenchmarkTask):
    """Implementation of the aging resilience benchmark task for C. elegans gait."""
    
    def __init__(self):
        super().__init__()
        self.spec_metrics = SpectralMetrics()

    @property
    def d_sensory(self) -> int:
        return 6

    def get_raw_datasets(self, config: Dict[str, Any]):
        dataset_config = config.get("dataset", {})
        data_path = dataset_config.get("path", "data/worm/EigenWorms_TEST.ts")
        seq_len = dataset_config.get("seq_len", 500)
        
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Biological data not found at {data_path}.")
            
            logger.info("Loading biological RealEigenwormDataset...")
            train_young = RealEigenwormDataset(data_path, seq_len=seq_len, inject_synthetic_degradation=False)
            eval_young = RealEigenwormDataset(data_path, seq_len=seq_len, inject_synthetic_degradation=False)
            eval_old = RealEigenwormDataset(data_path, seq_len=seq_len, inject_synthetic_degradation=True)
            
        except FileNotFoundError:
            logger.warning("Local biological data not found. Falling back to SyntheticWormMockDataset.")
            train_young = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
            eval_young = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
            eval_old = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
            
        return train_young, eval_young, eval_old

    def get_dataloaders(self, config: Dict[str, Any], d_state: int = None, batch_size: int = 2):
        train_young_raw, eval_young_raw, eval_old_raw = self.get_raw_datasets(config)
        
        train_young_dataset = JAXDictDataset(train_young_raw, d_state)
        eval_young_dataset = JAXDictDataset(eval_young_raw, d_state)
        eval_old_dataset = JAXDictDataset(eval_old_raw, d_state)
        
        train_loader = DataLoader(
            train_young_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=numpy_collate,
            drop_last=True
        )
        eval_young_loader = DataLoader(
            eval_young_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=numpy_collate,
            drop_last=True
        )
        eval_old_loader = DataLoader(
            eval_old_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=numpy_collate,
            drop_last=True
        )
        
        return train_loader, eval_young_loader, eval_old_loader

    def apply_dataset_change(self, trajectory, change_fn: Optional[Callable], **kwargs):
        if change_fn is not None:
            return change_fn(trajectory, **kwargs)
        return trajectory

    def compute_domain_metrics(self, trajectory) -> dict:
        # Expect trajectory as a torch Tensor or convert to numpy for amplitude residual stats
        traj_np = trajectory.numpy() if hasattr(trajectory, "numpy") else trajectory
        traj_torch = trajectory if isinstance(trajectory, torch.Tensor) else torch.from_numpy(trajectory)
        
        stats = amplitude_residual_stats(traj_np, pair=(0, 1))
        
        # Calculate Peak Frequency (25.0 Hz biological framerate)
        freq, power = self.spec_metrics.calculate_psd(traj_torch, sampling_rate=25.0)
        peak_idx = torch.argmax(power.mean(dim=1)) if len(power.shape) > 1 else 0
        peak_freq = float(freq[peak_idx])
        
        return {
            "amp_ar1": stats["amp_ar1"],
            "amp_var": stats["amp_var"],
            "peak_freq": peak_freq
        }

    def render_animation(self, young_data, old_data, output_path: str, **kwargs):
        frames = kwargs.get("frames", young_data.shape[0])
        fps = kwargs.get("fps", 30)
        create_worm_gait_animation(
            young_data=young_data,
            old_data=old_data,
            output_path=output_path,
            frames=frames,
            fps=fps
        )
