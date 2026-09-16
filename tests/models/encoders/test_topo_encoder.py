import torch
import torch.nn as nn
from src.models.encoders.topo_encoder import TopoEncoder
from src.data.ephys.uhd_lfp_dataset import ContinuousLFPDataset

def test_topo_encoder_integration():
    """
    Test that the ContinuousLFPDataset correctly applies the TopoEncoder
    on-the-fly to compress the 4D electrophysiology field into a 1D sequence or vector.
    """
    # Intentionally basic for the unit test
    ssm = nn.Identity() 
    
    encoder = TopoEncoder(ssm=ssm, d_model=768)
    
    # Test return_hidden=True (Sequence out)
    dataset = ContinuousLFPDataset(time_steps=5, grid_size=64, encoder=encoder, return_hidden=True)
    iterator = iter(dataset)
    E, stim = next(iterator)
    
    assert E.shape == (5, 768)
    assert stim.shape == (768,)
    
    # Test return_hidden=False (Final state out)
    dataset = ContinuousLFPDataset(time_steps=5, grid_size=64, encoder=encoder, return_hidden=False)
    iterator = iter(dataset)
    E, stim = next(iterator)
    
    assert E.shape == (768,)
    assert stim.shape == (768,)

if __name__ == "__main__":
    test_topo_encoder_integration()
    print("All tests passed!")
