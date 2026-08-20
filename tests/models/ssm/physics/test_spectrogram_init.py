import math
import torch
import pytest

from src.models.ssm.physics.spectrogram_init import BiologicalSpectrogramInit


def test_spectrogram_init_default():
    d_state = 16
    min_freq = 0.0001
    max_freq = 20000.0
    
    model = BiologicalSpectrogramInit(d_state=d_state, min_freq=min_freq, max_freq=max_freq)
    
    A = model()
    
    # Check shape
    assert A.shape == (d_state,)
    
    # Check strict negativity (stability guarantee)
    assert torch.all(A < 0)
    
    # Check frequency bounds
    # Since A = -f, the highest A (closest to 0) should be -min_freq
    # The lowest A (most negative) should be -max_freq
    assert torch.isclose(A[0], torch.tensor(-min_freq), rtol=1e-4)
    assert torch.isclose(A[-1], torch.tensor(-max_freq), rtol=1e-4)
    

def test_spectrogram_init_angular():
    d_state = 8
    min_freq = 1.0
    max_freq = 100.0
    
    model = BiologicalSpectrogramInit(d_state=d_state, min_freq=min_freq, max_freq=max_freq, use_angular_freq=True)
    
    A = model()
    
    assert A.shape == (d_state,)
    assert torch.all(A < 0)
    
    assert torch.isclose(A[0], torch.tensor(-2 * math.pi * min_freq), rtol=1e-4)
    assert torch.isclose(A[-1], torch.tensor(-2 * math.pi * max_freq), rtol=1e-4)
    

def test_spectrogram_init_validation():
    with pytest.raises(ValueError):
        BiologicalSpectrogramInit(d_state=4, min_freq=-1.0)
        
    with pytest.raises(ValueError):
        BiologicalSpectrogramInit(d_state=4, min_freq=10.0, max_freq=5.0)
