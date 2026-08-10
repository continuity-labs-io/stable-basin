import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class PharmacologicalShockDataset(Dataset):
    """
    Pharmacological Shock Dataset.
    
    Origin: Functional neuronal circuitry and oscillatory dynamics in human brain organoids (Nature Communications, 2022)
    Sampling: Continuous extracellular neural activity recorded via high-density CMOS microelectrode spatial arrays.
    
    The MVM Proof (Phase Transition): 
    Serves as the ultimate ground-truth test for the continuous-time solver. Instead of just counting 
    dropped spikes, this dataset proves the architecture can detect the exact millisecond the pharmacological 
    agent collapses the network's Kinetic Stability Metric (KSM) from 1.0 down to 0.0.
    """
    def __init__(self, condition: str = "control", base_path: str = "data/ephys/pharmacological_shock", seq_len: int = 1024):
        self.data_path = os.path.join(base_path, f"Drug_2953_{condition}.raw.h5")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Could not find dataset for condition '{condition}' at {self.data_path}")
            
        self.seq_len = seq_len
        
        with h5py.File(self.data_path, 'r') as f:
            # sig shape is (channels, time) e.g., (1028, 3597600)
            self.total_time_steps = f['sig'].shape[1]
            self.num_channels = min(1024, f['sig'].shape[0])  # Use up to 1024 neural channels
            self.length = self.total_time_steps // self.seq_len

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Open inside getitem for multiprocessing safety
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        with h5py.File(self.data_path, 'r') as f:
            # Slicing time (axis 1) and channels (axis 0)
            chunk = f['sig'][:self.num_channels, start_idx:end_idx]
            
        # chunk is (channels, time). We want (time, channels) for sequence modeling
        chunk = chunk.T
        
        # Convert uint16 to float32
        tensor_data = torch.from_numpy(chunk.astype(np.float32))
        return tensor_data
