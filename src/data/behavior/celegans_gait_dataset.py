import os
import logging
import math
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional

logger = logging.getLogger(__name__)

class RealEigenwormDataset(Dataset):
    """
    Dataset for C. elegans gait trajectories using only biological data.
    Extracts random, contiguous, fixed-length crops.
    """

    def __init__(self, data_path: str, seq_len: int = 500, is_aged: bool = False):
        """
        Initializes the dataset.

        Args:
            data_path: Path to the biological eigenworm data (.npy, .csv, .ts).
            seq_len: The fixed length of each extracted sequence crop.
            is_aged: If True, applies an OU/Gaussian noise process to simulate thermodynamic degradation.
        """
        self.seq_len = seq_len
        self.data = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Biological data not found at {data_path}")

        logger.info(f"Loaded Real Biological Data from {data_path}")
        if data_path.endswith('.npy'):
            raw_data = np.load(data_path)
            if raw_data.ndim == 2:
                self.data = [torch.tensor(raw_data, dtype=torch.float32)]
            elif raw_data.ndim == 3:
                self.data = [torch.tensor(traj, dtype=torch.float32) for traj in raw_data]
            else:
                raise ValueError("Unexpected shape for biological data.")
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            self.data = [torch.tensor(df.values, dtype=torch.float32)]
        elif data_path.endswith('.ts'):
            with open(data_path, 'r') as f:
                in_data = False
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line.startswith('@data'):
                        in_data = True
                        continue
                    if in_data:
                        parts = line.split(':')
                        dims = parts[:6]
                        tensor_dims = []
                        for d in dims:
                            vals = [float(x) for x in d.split(',') if x]
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
            
            if is_aged:
                # Apply thermodynamic noise degradation
                traj = traj * 0.5 + torch.randn_like(traj) * 0.2
                
            normalized_data.append(traj)
            
        self.data = normalized_data
            
        # Filter out trajectories that are shorter than seq_len
        valid_data = [t for t in self.data if t.shape[0] >= self.seq_len]
        if len(valid_data) < len(self.data):
            logger.info(f"Filtered out {len(self.data) - len(valid_data)} trajectories shorter than seq_len={self.seq_len}.")
        self.data = valid_data
        
        if len(self.data) == 0:
            raise ValueError(f"No trajectories in the dataset are long enough for seq_len={seq_len}.")

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


class SyntheticWormMockDataset(Dataset):
    """
    Purely for CI smoke tests. Generates simple, deterministic 6D sine waves.
    """
    
    def __init__(self, seq_len: int = 500, num_samples: int = 100):
        """
        Initializes the synthetic dataset.
        
        Args:
            seq_len: The length of each synthetic sequence.
            num_samples: The number of sequences in the dataset.
        """
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = [self._generate_synthetic_data(seq_len=seq_len, seed=42 + i) for i in range(num_samples)]
        
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
    def _generate_synthetic_data(seq_len: int = 500, seed: int = 42) -> torch.Tensor:
        rng = torch.Generator().manual_seed(seed)
        time_steps = torch.arange(seq_len, dtype=torch.float32)
        frequency = 0.05
        
        phases = torch.rand(6, generator=rng) * 2 * math.pi
        amplitudes = torch.rand(6, generator=rng) * 0.5 + 0.5
        
        cycle = torch.zeros((seq_len, 6), dtype=torch.float32)
        for i in range(6):
            cycle[:, i] = amplitudes[i] * torch.sin(2 * math.pi * frequency * time_steps + phases[i])
            
        return cycle
