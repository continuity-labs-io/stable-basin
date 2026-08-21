import os
import pytest
import numpy as np
import tifffile
from torch.utils.data import DataLoader
from src.data.optical.aollsm_dataloader import MeldTemporalDataset, AOLLSMDataset

def test_aollsm_dataloader(tmp_path):
    raw_tiffs_dir = tmp_path / "raw_tiffs"
    raw_tiffs_dir.mkdir()
    
    sample_dir = raw_tiffs_dir / "sample_0"
    sample_dir.mkdir()
    for i in range(10):
        dummy_data = np.zeros((10, 32, 32), dtype=np.uint16)
        tifffile.imwrite(str(sample_dir / f"dummy_stack{i:04d}_ch1_.tif"), dummy_data)
        tifffile.imwrite(str(sample_dir / f"dummy_stack{i:04d}_ch2_.tif"), dummy_data)
        tifffile.imwrite(str(raw_tiffs_dir / f"frame_{i:04d}.tif"), dummy_data)
        
    dataset_original = MeldTemporalDataset(data_dir=str(raw_tiffs_dir), sequence_length=5)
    dataloader_original = DataLoader(dataset_original, batch_size=1, shuffle=False)
    batch_orig = next(iter(dataloader_original))
    # Shape [Batch, Channel, Time, Z, Y, X]
    assert batch_orig.shape == (1, 1, 5, 10, 32, 32)
    
    dataset_aollsm = AOLLSMDataset(
        data_dir=str(raw_tiffs_dir), num_frames=10, crop_size=(8, 8, 8)
    )
    dataloader_aollsm = DataLoader(dataset_aollsm, batch_size=1, shuffle=False)
    batch_aollsm = next(iter(dataloader_aollsm))
    # Shape [Batch, Time, Channels, Depth, Height, Width]
    assert batch_aollsm.shape == (1, 10, 2, 8, 8, 8)
