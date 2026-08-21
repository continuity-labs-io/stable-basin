import pytest
from src.data.sim2real.epigenetic_entropy_dataloader import EpigeneticEntropyLoader

def test_epigenetic_entropy_dataloader():
    dataset_45 = EpigeneticEntropyLoader(biological_age=45, size=1, seq_len=10)
    dataset_50 = EpigeneticEntropyLoader(biological_age=50, size=1, seq_len=10)
    
    t_45 = dataset_45[0]["cpg_tensor"]
    t_50 = dataset_50[0]["cpg_tensor"]
    
    assert t_45.shape == (10, 1000, 10000)
    assert t_50.shape == (10, 1000, 10000)
