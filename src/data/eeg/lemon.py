import os
import torch
from torch.utils.data import Dataset
import numpy as np
import mne
import logging
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Suppress MNE INFO logs globally
mne.set_log_level("WARNING")

class LemonEEGDataset(Dataset):
    """
    LEMON EEG Dataloader.
    Uses MNE to load preprocessed .set (EEGLAB) files, performs PCA reduction
    to manage dimensionality, and yields fixed-length crops. Features a synthetic
    fallback mechanism for CI testing.
    """
    def __init__(self, data_path: str = "data/lemon/", size: int = 100, seq_len: int = 3000, n_components: int = 5):
        super().__init__()
        self.data_path = data_path
        self.size = size
        self.seq_len = seq_len
        self.n_components = n_components
        
        self.use_synthetic = True
        self.data = None
        
        if os.path.exists(data_path):
            set_files = [f for f in os.listdir(data_path) if f.endswith('.set')]
            if len(set_files) > 0:
                self.use_synthetic = False
                logger.info(f"Loading real LEMON data from {data_path}")
                # Load the first file for demonstration
                raw_file = os.path.join(data_path, set_files[0])
                try:
                    # Load and get data
                    raw = mne.io.read_raw_eeglab(raw_file, preload=True, verbose="WARNING")
                    data = raw.get_data().T  # Shape: [time, channels]
                    
                    # PCA Reduction
                    pca = PCA(n_components=self.n_components)
                    self.data = torch.tensor(pca.fit_transform(data), dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"Failed to load real LEMON data: {e}.")
                    logger.info("Local biological data not found. Falling back to deterministic synthetic generation for CI.")
                    self.use_synthetic = True
            else:
                logger.info("Local biological data not found. Falling back to deterministic synthetic generation for CI.")
                self.use_synthetic = True
        else:
            logger.info("Local biological data not found. Falling back to deterministic synthetic generation for CI.")
            self.use_synthetic = True
            
        if self.use_synthetic:
            self.synthetic_data = [
                self.generate_synthetic_data(self.seq_len, self.n_components, seed=42 + i)
                for i in range(self.size)
            ]

    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        if self.use_synthetic:
            x_raw = self.synthetic_data[idx]
        else:
            max_start = max(0, len(self.data) - self.seq_len)
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
            x_raw = self.data[start_idx : start_idx + self.seq_len]
            # Pad if the data is shorter than seq_len
            if len(x_raw) < self.seq_len:
                pad = torch.zeros(self.seq_len - len(x_raw), self.n_components)
                x_raw = torch.cat([x_raw, pad], dim=0)
                
        mask = torch.ones_like(x_raw)
        y_true = x_raw[:, 0:1] # Dummy target for autoregression compatibility
        
        return {"x_raw": x_raw, "mask": mask, "y_true": y_true}

    @staticmethod
    def generate_synthetic_data(seq_len: int, n_components: int, seed: int = 42) -> torch.Tensor:
        """
        Generates deterministic multivariate Gaussian AR(1) process noise mimicking PCA.
        """
        rng = torch.Generator().manual_seed(seed)
        noise = torch.randn(seq_len, n_components, generator=rng)
        x_raw = torch.zeros(seq_len, n_components)
        x_raw[0] = noise[0]
        alpha = 0.95
        for t in range(1, seq_len):
            x_raw[t] = alpha * x_raw[t-1] + (1 - alpha) * noise[t]
        return x_raw
