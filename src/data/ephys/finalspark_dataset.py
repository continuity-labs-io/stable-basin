import os
import h5py
import torch
from torch.utils.data import Dataset

class FinalSparkDataset(Dataset):
    """
    FinalSpark Whole-Life Telemetry Dataset.
    
    Origin: Open and remotely accessible Neuroplatform for research in wetware computing
    Sampling: 30kHz-resolution raw activity samples. Data is provided in HDF5 format.
    
    The MVM Proof (Multi-Modal Trajectory):
    Validates that the Invariant Core can fuse microsecond electrical telemetry with macroscopic 
    environmental shifts (like incubator door openings). It demonstrates how environmental perturbations 
    alter the thermodynamic vector and predict the ultimate lifespan of the organoid.
    """
    def __init__(self, data_path: str = "data/ephys/finalspark/fs437_export/fs437_raw.hdf5", seq_len: int = 1024):
        self.data_path = data_path
        self.seq_len = seq_len
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
        # Determine total length by checking file once.
        with h5py.File(self.data_path, 'r') as f:
            total_samples = f['fs437_wholelife_raw']['table'].shape[0]
            self.length = total_samples // self.seq_len

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Open inside getitem for multiprocessing safety (avoids HDF5 segfaults when num_workers > 0)
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        with h5py.File(self.data_path, 'r') as f:
            chunk = f['fs437_wholelife_raw']['table'][start_idx:end_idx]
            # Extract the actual numerical values from the structured array
            values = chunk['values_block_0']
        
        # The values shape is (seq_len, 1), convert to float tensor
        tensor_data = torch.from_numpy(values).float()
        return tensor_data
