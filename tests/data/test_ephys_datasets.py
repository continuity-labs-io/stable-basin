import os
import json
import h5py
import torch
import numpy as np
import pytest

from src.data.ephys.finalspark_dataset import FinalSparkDataset
from src.data.ephys.pharma_shock_dataset import PharmacologicalShockDataset
from src.data.ephys.hdmea_dataset import HDMEADataset
from src.data.ephys.maxwell_dataset import MaxWellHDMEADataset
from src.data.ephys.spike_dataset import SpikeProphecyDataset
from src.data.ephys.uhd_lfp_dataset import ContinuousLFPDataset


def test_finalspark_dataset(tmp_path):
    h5_path = tmp_path / "fs437_raw.hdf5"
    with h5py.File(h5_path, 'w') as f:
        grp = f.create_group("fs437_wholelife_raw")
        # Create a structured array matching the FinalSpark format
        dt = np.dtype([('index', '<i8'), ('values_block_0', '<f8', (1,)), ('time', '<i8'), ('electrode', '<i8')])
        data = np.zeros(2048, dtype=dt)
        grp.create_dataset("table", data=data)
        
    ds = FinalSparkDataset(data_path=str(h5_path), seq_len=1024)
    assert len(ds) == 2
    batch = ds[0]
    assert batch.shape == (1024, 1)
    assert batch.dtype == torch.float32

def test_pharma_shock_dataset(tmp_path):
    base_path = tmp_path
    h5_path = base_path / "Drug_2953_control.raw.h5"
    with h5py.File(h5_path, 'w') as f:
        # 1028 channels, 2048 time steps
        f.create_dataset("sig", data=np.zeros((1028, 2048), dtype=np.uint16))
        
    ds = PharmacologicalShockDataset(condition="control", base_path=str(base_path), seq_len=1024)
    assert len(ds) == 2
    batch = ds[0]
    # Expecting time x channels, and max 1024 channels
    assert batch.shape == (1024, 1024)
    assert batch.dtype == torch.float32

def test_hdmea_dataset(tmp_path):
    brw_path = tmp_path / "hdmea_neuropulse.brw"
    with h5py.File(brw_path, 'w') as f:
        grp = f.create_group("3BData")
        # 4096 channels * 2048 frames = 8388608 elements
        grp.create_dataset("Raw", data=np.zeros(8388608, dtype=np.uint16))
        
    ds = HDMEADataset(data_path=str(brw_path), seq_len=1024)
    assert len(ds) == 2
    batch = ds[0]
    assert batch.shape == (1024, 4096)
    assert batch.dtype == torch.float32

def test_maxwell_dataset(tmp_path):
    h5_path = tmp_path / "test.raw.h5"
    with h5py.File(h5_path, 'w') as f:
        # 1028 channels, 2048 time steps
        f.create_dataset("sig", data=np.zeros((1028, 2048), dtype=np.uint16))
        
    ds = MaxWellHDMEADataset(file_path=str(h5_path), sequence_length=1024, target_channels=512)
    # The maxwell dataloader likely uses standard Dataset properties
    assert len(ds) == 2
    batch = ds[0]
    assert batch.shape == (1024, 512)
    assert batch.dtype == torch.float32

def test_spike_prophecy_dataset(tmp_path):
    meta_path = tmp_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "m_max": 100,
            "num_sessions": 1,
            "sessions": {"0": {"num_trials": 1, "split_boundaries": {"train_end": 100, "val_end": 150}}}
        }, f)
        
    np.save(tmp_path / "session_000.npy", np.zeros((100, 200), dtype=np.float32)) # (n_units, total_bins)
    
    ds = SpikeProphecyDataset(time_steps=50, split="train", data_dir=str(tmp_path))
    iterator = iter(ds)
    tensor_window = next(iterator)
    assert tensor_window.shape == (50, 100) # time_steps, m_max

def test_continuous_lfp_dataset():
    ds = ContinuousLFPDataset(time_steps=50, grid_size=32)
    iterator = iter(ds)
    E, visual_embedding = next(iterator)
    # LFP yields a continuous wave [time_steps, 2, grid_size, grid_size] and a 768-D visual stimulus
    assert E.shape == (50, 2, 32, 32)
    assert E.dtype == torch.float32
    assert visual_embedding.shape == (768,)
