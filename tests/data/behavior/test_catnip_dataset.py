import pytest
import tempfile
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.data.behavior.catnip_dataset import CatnipContinuousDataset

def test_catnip_dataset_parsing_and_filtering():
    # ARRANGE
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "test_features.h5"
        
        # Create mock data
        num_samples = 30
        features = np.random.randn(num_samples, 5)
        features_df = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(5)])
        
        # Trace metadata: associates each run with mouse_id and age
        # m1: 5 young, 5 old
        # m2: 10 old
        # m3: 10 young
        trace_meta_df = pd.DataFrame({
            "mouse_id": ["m1"] * 10 + ["m2"] * 10 + ["m3"] * 10,
            "age_months": [4] * 5 + [26] * 5 + [25] * 10 + [3] * 10
        })
        
        # Mouse metadata: associates each mouse with lifespan
        mouse_meta_df = pd.DataFrame({
            "mouse_id": ["m1", "m2", "m3"],
            "lifespan": [30, 28, 35]
        })
        
        # Save to HDF5
        features_df.to_hdf(h5_path, key="all features")
        trace_meta_df.to_hdf(h5_path, key="trace metadata")
        mouse_meta_df.to_hdf(h5_path, key="mouse metadata")
        
        # ACT - "all" cohort
        dataset_all = CatnipContinuousDataset(h5_path=str(h5_path), sequence_length=5, cohort="all")
        
        # ASSERT
        assert len(dataset_all) == 6  # 30 samples / 5 = 6 chunks
        tensor = dataset_all[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (5, 5)
        
        # ACT - "young" cohort
        dataset_young = CatnipContinuousDataset(h5_path=str(h5_path), sequence_length=5, cohort="young")
        
        # ASSERT - m1 (first 5 = 1 chunk), m3 (10 = 2 chunks) -> 3 chunks
        assert len(dataset_young) == 3
        
        # ACT - "old" cohort
        dataset_old = CatnipContinuousDataset(h5_path=str(h5_path), sequence_length=5, cohort="old")
        
        # ASSERT - m1 (last 5 = 1 chunk), m2 (10 = 2 chunks) -> 3 chunks
        assert len(dataset_old) == 3

def test_catnip_dataset_missing_file():
    # ARRANGE
    missing_path = "non_existent_file.h5"
    
    # ACT
    dataset = CatnipContinuousDataset(h5_path=missing_path)
    
    # ASSERT
    assert len(dataset) == 0

def test_catnip_dataset_zscore_normalization():
    # ARRANGE
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "test_features.h5"
        
        num_samples = 10
        # Create features with specific mean and std
        features = np.zeros((num_samples, 2))
        features[:, 0] = np.arange(10)  # mean 4.5, std ~ 2.87
        features[:, 1] = np.ones(10) * 5  # constant feature, std 0
        
        features_df = pd.DataFrame(features, columns=["feat_0", "feat_1"])
        
        trace_meta_df = pd.DataFrame({
            "mouse_id": ["m1"] * 10,
            "age_months": [4] * 10
        })
        
        mouse_meta_df = pd.DataFrame({
            "mouse_id": ["m1"],
            "lifespan": [30]
        })
        
        features_df.to_hdf(h5_path, key="all features")
        trace_meta_df.to_hdf(h5_path, key="trace metadata")
        mouse_meta_df.to_hdf(h5_path, key="mouse metadata")
        
        # ACT
        dataset = CatnipContinuousDataset(h5_path=str(h5_path), sequence_length=10, cohort="all")
        
        # ASSERT
        chunk = dataset[0]
        
        # Check z-score normalization
        # First feature should have mean roughly 0 and std roughly 1
        assert torch.allclose(chunk[:, 0].mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(chunk[:, 0].std(unbiased=False), torch.tensor(1.0), atol=1e-5)
        
        # Second feature should be 0 because of division by 1 and subtracting mean
        assert torch.allclose(chunk[:, 1].mean(), torch.tensor(0.0), atol=1e-5)
