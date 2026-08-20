import math
import torch
import pytest

from src.models.ssm.physics.matrix_a_factory import create_a_matrix


def test_factory_log_spaced_1d():
    d_state = 16
    min_freq = 0.0001
    max_freq = 20000.0
    
    model = create_a_matrix("log_spaced", shape=(d_state,), min_freq=min_freq, max_freq=max_freq)
    
    A = model()
    
    # Check shape
    assert A.shape == (d_state,)
    
    # Check strict negativity (stability guarantee)
    assert torch.all(A < 0)
    
    # Check frequency bounds
    assert torch.isclose(A[0], torch.tensor(-min_freq), rtol=1e-4)
    assert torch.isclose(A[-1], torch.tensor(-max_freq), rtol=1e-4)
    

def test_factory_log_spaced_2d():
    d_model = 8
    d_state = 16
    min_freq = 1.0
    max_freq = 100.0
    
    model = create_a_matrix("log_spaced", shape=(d_model, d_state), min_freq=min_freq, max_freq=max_freq, use_angular_freq=True)
    
    A = model()
    
    assert A.shape == (d_model, d_state)
    assert torch.all(A < 0)
    
    # Because it is broadcasted over d_model, any item in dim 0 should match
    assert torch.isclose(A[0, 0], torch.tensor(-2 * math.pi * min_freq), rtol=1e-4)
    assert torch.isclose(A[0, -1], torch.tensor(-2 * math.pi * max_freq), rtol=1e-4)
    

def test_factory_random_1d():
    d_state = 16
    
    model = create_a_matrix("random", shape=(d_state,))
    
    A = model()
    
    assert A.shape == (d_state,)
    assert torch.all(A < 0)


def test_factory_random_2d():
    d_model = 8
    d_state = 16
    
    model = create_a_matrix("random", shape=(d_model, d_state))
    
    A = model()
    
    assert A.shape == (d_model, d_state)
    assert torch.all(A < 0)


def test_factory_validation():
    with pytest.raises(ValueError):
        create_a_matrix("unknown", shape=(4,))
        
    with pytest.raises(ValueError):
        create_a_matrix("log_spaced", shape=(4,), min_freq=-1.0)
        
    with pytest.raises(ValueError):
        create_a_matrix("log_spaced", shape=(4,), min_freq=10.0, max_freq=5.0)
