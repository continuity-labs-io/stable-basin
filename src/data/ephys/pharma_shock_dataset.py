import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", "..", ".."))


class PharmacologicalShockDataset(Dataset):
    """
    Pharmacological Shock Dataset.

    Data source: Functional neuronal circuitry and oscillatory dynamics in human brain organoids (Nature Communications, 2022)
    Format: Continuous extracellular neural activity recorded via high-density CMOS microelectrode spatial arrays.
    Array shape: (seq_len, num_channels)
    """

    def __init__(self, condition: str = "control", base_path: str = None, seq_len: int = 1024):
        if base_path is None:
            # Resolve relative to the module's original absolute location
            # so it survives Ray Tune changing the worker's CWD
            base_path = os.path.join(PROJECT_ROOT, "data", "ephys", "pharmacological_shock")

        self.data_path = os.path.abspath(os.path.join(base_path, f"Drug_2953_{condition}.raw.h5"))
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Could not find dataset for condition '{condition}' at {self.data_path}"
            )

        self.seq_len = seq_len

        with h5py.File(self.data_path, "r") as f:
            # sig shape is (channels, time) e.g., (1028, 3597600)
            self.total_time_steps = f["sig"].shape[1]
            self.num_channels = min(1024, f["sig"].shape[0])  # Use up to 1024 neural channels
            self.length = self.total_time_steps // self.seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open inside getitem for multiprocessing safety
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        with h5py.File(self.data_path, "r") as f:
            # Slicing time (axis 1) and channels (axis 0)
            chunk = f["sig"][: self.num_channels, start_idx:end_idx]

        # chunk is (channels, time). We want (time, channels) for sequence modeling
        chunk = chunk.T

        # Convert uint16 to float32
        tensor_data = torch.from_numpy(chunk.astype(np.float32))
        return tensor_data
