import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class HDMEADataset(Dataset):
    """
    3Brain HD-MEA 4,096-Channel Stress Test Dataset.
    
    Origin: HD-MEA NEUROPulse
    Sampling: 20kHz continuous telemetry stored in raw BrainWave format (hdmea_neuropulse.brw).
    
    The MVM Proof (Hardware Scale): 
    The ultimate engineering benchmark. By piping this massive spatial grid into the state-space engine, 
    you generate the visual proof that the architecture maintains a linear, O(1) VRAM footprint without 
    triggering Out-Of-Memory crashes, defeating standard Transformer models side-by-side.
    """
    def __init__(self, data_path: str = "data/ephys/hdmea_neuropulse.brw", seq_len: int = 1024):
        self.data_path = data_path
        self.seq_len = seq_len
        self.num_channels = 4096
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
        with h5py.File(self.data_path, 'r') as f:
            # BRW files have a flat 1D array for data. 
            # We must calculate total frames based on the flat array size / 4096.
            total_elements = f['3BData/Raw'].shape[0]
            self.total_frames = total_elements // self.num_channels
            self.length = self.total_frames // self.seq_len

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Open inside getitem for multiprocessing safety
        start_frame = idx * self.seq_len
        end_frame = start_frame + self.seq_len
        
        start_idx = start_frame * self.num_channels
        end_idx = end_frame * self.num_channels
        
        with h5py.File(self.data_path, 'r') as f:
            # Read flat chunk
            flat_chunk = f['3BData/Raw'][start_idx:end_idx]
            
        # Reshape to (seq_len, channels)
        chunk = flat_chunk.reshape((self.seq_len, self.num_channels))
        
        tensor_data = torch.from_numpy(chunk.astype(np.float32))
        return tensor_data
