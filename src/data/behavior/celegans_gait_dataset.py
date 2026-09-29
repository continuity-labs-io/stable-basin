import os
import logging
import math
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional
from src.data.behavior.synthetic_aging import slow_amplitude_relaxation

logger = logging.getLogger(__name__)


class RealEigenwormDataset(Dataset):
    """
    Dataset for C. elegans gait trajectories using only biological data.
    Extracts random, contiguous, fixed-length crops.
    """

    def __init__(self, data_path: str, seq_len: int = 500, inject_synthetic_degradation: bool = False):
        """
                Initializes the dataset.

                Args:
                    data_path: Path to the biological eigenworm data (.npy, .csv, .ts).
        seq_len: The fixed length of each extracted sequence crop. Biological recordings are
                    often
        too long to process all at once due to memory and BPTT constraints. This
                             parameter
        defines the number of consecutive frames in the bite-sized chunks that the
                             dataset
        will return when sampled, ensuring uniform and computationally manageable
                             inputs.
        inject_synthetic_degradation: Slows gait-amplitude relaxation 3x via synthetic_aging.slow_amplitude_relaxation. Synthetic positive control, not aged worms.
        """
        self.seq_len = seq_len
        self.data = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Biological data not found at {data_path}")

        logger.info(f"Loaded Real Biological Data from {data_path}")
        if data_path.endswith(".npy"):
            raw_data = np.load(data_path)
            if raw_data.ndim == 2:
                self.data = [torch.tensor(raw_data, dtype=torch.float32)]
            elif raw_data.ndim == 3:
                self.data = [torch.tensor(traj, dtype=torch.float32) for traj in raw_data]
            else:
                raise ValueError("Unexpected shape for biological data.")
        elif data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
            self.data = [torch.tensor(df.values, dtype=torch.float32)]
        elif data_path.endswith(".ts"):
            with open(data_path, "r") as f:
                in_data = False
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("@data"):
                        in_data = True
                        continue
                    if in_data:
                        parts = line.split(":")
                        dims = parts[:6]
                        tensor_dims = []
                        for d in dims:
                            vals = [float(x) for x in d.split(",") if x]
                            tensor_dims.append(vals)
                        traj = torch.tensor(tensor_dims, dtype=torch.float32).t()
                        self.data.append(traj)

        # Global Z-score Normalization
        all_trajectories = torch.cat(self.data, dim=0)
        global_mean = all_trajectories.mean(dim=0)
        global_std = all_trajectories.std(dim=0)

        normalized_data = []
        for traj in self.data:
            traj = (traj - global_mean) / (global_std + 1e-8)

            if inject_synthetic_degradation:
                # Slows gait-amplitude relaxation 3x via synthetic_aging.slow_amplitude_relaxation. Synthetic positive control, not aged worms.
                traj = torch.tensor(slow_amplitude_relaxation(traj.numpy(), slowdown=3.0, pair=(0, 1)), dtype=torch.float32)

            normalized_data.append(traj)

        self.data = normalized_data

        # Filter out trajectories that are shorter than seq_len
        valid_data = [t for t in self.data if t.shape[0] >= self.seq_len]
        if len(valid_data) < len(self.data):
            logger.info(
                f"Filtered out {len(self.data) - len(valid_data)} trajectories shorter than "
                f"seq_len={self.seq_len}."
            )
        self.data = valid_data

        if len(self.data) == 0:
            raise ValueError(
                f"No trajectories in the dataset are long enough for seq_len={seq_len}."
            )

    def __len__(self) -> int:
        """
        Returns the number of trajectories in the dataset.

        Returns:
            int: The number of valid trajectories.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Extracts a random, contiguous, fixed-length crop from the trajectory.

        Args:
            idx: Index of the trajectory.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, 6).
        """
        trajectory = self.data[idx]
        max_start_idx = trajectory.shape[0] - self.seq_len

        start_idx = 0
        if max_start_idx > 0:
            start_idx = random.randint(0, max_start_idx)

        return trajectory[start_idx : start_idx + self.seq_len]


from src.data.behavior.synthetic_aging import slow_amplitude_relaxation, _stuart_landau

class SyntheticWormMockDataset(Dataset):
    """
    Noisy Stuart-Landau oscillator in channels 0-1, Gaussian noise in channels 2-5.
    """

    def __init__(self, seq_len: int = 500, num_samples: int = 100, degraded: bool = False):
        """
        Initializes the synthetic dataset.

        Args:
            seq_len: The length of each synthetic sequence.
            num_samples: The number of sequences in the dataset.
            degraded: Whether to apply thermodynamic degradation.
        """
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.degraded = degraded
        self.data = [
            self._generate_synthetic_data(seq_len=seq_len, seed=42 + i, degraded=degraded) for i in range(num_samples)
        ]

    def __len__(self) -> int:
        """
        Returns the number of trajectories in the synthetic dataset.

        Returns:
            int: The number of valid trajectories.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Extracts a synthetic trajectory.

        Args:
            idx: Index of the trajectory.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, 6).
        """
        return self.data[idx]

    @staticmethod
    def _generate_synthetic_data(seq_len: int = 500, seed: int = 42, degraded: bool = False) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        dt = 1.0 / 25.0
        
        osc = _stuart_landau(seq_len, dt, freq=0.5, mu=1.0, noise=0.1, rng=rng)
        
        traj = np.zeros((seq_len, 6), dtype=np.float32)
        traj[:, :2] = osc
        traj[:, 2:] = rng.standard_normal((seq_len, 4)) * 0.1
        
        if degraded:
            traj = slow_amplitude_relaxation(traj, slowdown=3.0, pair=(0, 1))
            
        return torch.tensor(traj, dtype=torch.float32)
