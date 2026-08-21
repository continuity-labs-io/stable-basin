import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import math


class SyntheticWaddingtonDataset(Dataset):
    """
    Synthetic biological dataset representing a cell moving through a phase transition
    (the Waddington landscape).

    Generates synthetic sequences comprising:
    - y_true: The 1D target tracking continuous phase transitions (FitzHugh-Nagumo fast variable).
    - x_raw: A 30-dimensional tensor composed of two modalities:
      - Modality 0 (20D): Continuous projection of the slow recovery variable w.
      - Modality 1 (10D): Sparse projection of the fast spiking variable v (masked 95% of the time).
    - mask: A 2-dimensional tensor representing the observability of the two modalities.
    
    Parameters:
    - size (int): The total number of unique sequences generated per epoch.
    - seq_len (int): The total time steps (length) of each sequence generated. 
    - density (float): The probability (0.0 to 1.0) that a Modality 1 sensor is active at any given step.
    
    Mathematics (FitzHugh-Nagumo Model):
    The data is generated dynamically on-the-fly using the FitzHugh-Nagumo oscillator equations.
    It is a 2D simplification of the Hodgkin-Huxley model of neural spiking:
      dv/dt = v - (v^3) / 3 - w + I_ext
      dw/dt = (v + a - b * w) / tau
      
    Where:
    - v (fast variable): Represents membrane voltage (action potentials).
    - w (slow variable): Represents recovery / gating kinetics.
    - I_ext (0.5): External stimulus current driving the system.
    - a (0.7), b (0.8): Kinetics parameters controlling the nullclines.
    - tau (10000.0): The time-scale separation parameter making 'w' much slower than 'v'.
    """

    def __init__(self, size: int = 100, seq_len: int = 500, density: float = 0.05,
                 dim_slow: int = 20, dim_fast: int = 10, noise_std: float = 0.05,
                 tau: float = 10000.0, a: float = 0.7, b: float = 0.8, I_ext: float = 0.5):
        self.size = size
        self.seq_len = seq_len
        self.density = density
        self.dim_slow = dim_slow
        self.dim_fast = dim_fast
        self.noise_std = noise_std
        
        # FHN Parameters
        self.tau = tau
        self.a = a
        self.b = b
        self.I_ext = I_ext
        self.dt = 0.1
        self.sub_steps = 200
        self.init_v_scale = 0.5
        self.init_w_scale = 0.1

        # We must ensure that the "biological layout" (the W_0 and W_1 sensor projections)
        # is absolutely identical across all test runs and seeds. However, we don't want
        # to ruin the global random seed used for generating the actual stochastic simulation data.
        # Solution: Capture the global state, force a hardcoded biological seed, then perfectly restore.
        global_rng_state = torch.get_rng_state()
        
        biological_mapping_seed = 42
        torch.manual_seed(biological_mapping_seed)
        
        self.W_0 = torch.randn(1, self.dim_slow)
        self.W_1 = torch.randn(1, self.dim_fast)
        
        # Restore the global state so __getitem__ still respects the external runner's seed
        torch.set_rng_state(global_rng_state)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        v = torch.zeros(self.seq_len, 1)
        w = torch.zeros(self.seq_len, 1)

        # Randomize initial conditions slightly to vary sequences
        v_curr = torch.randn(1).item() * self.init_v_scale
        w_curr = torch.randn(1).item() * self.init_w_scale

        # Euler Integration Loop
        for i in range(self.seq_len):
            for _ in range(self.sub_steps):
                dv = v_curr - (v_curr**3) / 3.0 - w_curr + self.I_ext
                dw = (v_curr + self.a - self.b * w_curr) / self.tau
                v_curr += dv * self.dt
                w_curr += dw * self.dt
            v[i, 0] = v_curr
            w[i, 0] = w_curr

        # Target is the fast variable
        y_true = v

        # Modality 0 (Continuous slow variable)
        modality_0 = w @ self.W_0 + torch.randn(self.seq_len, self.dim_slow) * self.noise_std

        # Modality 1 (Sparse fast variable)
        modality_1 = v @ self.W_1 + torch.randn(self.seq_len, self.dim_fast) * self.noise_std

        # The Mask
        mask_0 = torch.ones(self.seq_len, 1)
        # density is the fraction of active sensors
        mask_1 = (torch.rand(self.seq_len, 1) < self.density).float()

        # CRITICAL ZERO-PADDING
        modality_1 = modality_1 * mask_1

        # Combine masks
        mask = torch.cat([mask_0, mask_1], dim=1)

        # Output
        x_raw = torch.cat([modality_0, modality_1], dim=1)
        return {"x_raw": x_raw, "mask": mask, "y_true": y_true}


