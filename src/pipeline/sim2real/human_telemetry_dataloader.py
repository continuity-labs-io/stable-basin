import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
import numpy as np

class HumanTelemetryLoader(Dataset):
    """
    Continuous Human Telemetry Dataloader for the "Stable Basin" framework.
    Mocks a middle-aged human (50 years old) with multi-scale wearable data.
    
    Frequencies aligned to 250Hz Master Clock:
    - HRV: 250Hz (step=1) -> 1D
    - Actigraphy: 50Hz (step=5) -> 3D
    - Core Temp: 1Hz (step=250) -> 1D
    - CGM: 1/60Hz (step=15000) -> 1D
    Total Features: 6
    """
    def __init__(self, size: int = 100, seq_len: int = 30000):
        super().__init__()
        self.size = size
        self.seq_len = seq_len
        self.master_hz = 250
        
        # Steps in master clock ticks
        self.steps = {
            "hrv": 1,
            "actigraphy": 5,
            "temp": 250,
            "cgm": 15000
        }
        
    def __len__(self):
        return self.size
        
    def _generate_base_signals(self):
        """Generate smooth continuous signals at 250Hz"""
        t = torch.arange(self.seq_len, dtype=torch.float32) / self.master_hz
        
        # HRV: Mean 800ms + respiratory sinus arrhythmia (0.25Hz) + noise
        base_hrv = 800.0 + 50.0 * torch.sin(2 * math.pi * 0.25 * t) + 10.0 * torch.randn(self.seq_len)
        base_hrv = base_hrv.unsqueeze(1)
        
        # Actigraphy: 3D (X, Y, Z) - mostly resting, occasional bursts
        # We use a smoothed random walk for movement
        base_act = torch.randn(self.seq_len, 3) * 0.1
        # Add a movement burst
        burst_center = torch.randint(0, self.seq_len, (1,)).item()
        burst_width = 250 * 5  # 5 seconds
        burst = torch.exp(-0.5 * ((t - burst_center/self.master_hz) / 2.0)**2).unsqueeze(1)
        base_act = base_act + (torch.randn(self.seq_len, 3) * 2.0) * burst
        
        # Core Temp: Mean 37.0 C + very slow circadian drift
        base_temp = 37.0 + 0.5 * torch.sin(2 * math.pi * (1.0 / 86400.0) * t + torch.rand(1).item() * 2 * math.pi)
        base_temp = base_temp.unsqueeze(1)
        
        # CGM: Mean 100 mg/dL + slow post-prandial waves (e.g. 2 hour period)
        base_cgm = 100.0 + 30.0 * torch.sin(2 * math.pi * (1.0 / 7200.0) * t + torch.rand(1).item() * 2 * math.pi)
        base_cgm = base_cgm.unsqueeze(1)
        
        return base_hrv, base_act, base_temp, base_cgm
        
    def _align_to_master_clock(self, base_hrv, base_act, base_temp, base_cgm):
        """Sample and forward-fill variables according to their native frequencies"""
        seq_idx = torch.arange(self.seq_len)
        
        # Masks
        mask_hrv = (seq_idx % self.steps["hrv"] == 0).float().unsqueeze(1)
        mask_act = (seq_idx % self.steps["actigraphy"] == 0).float().unsqueeze(1).repeat(1, 3)
        mask_temp = (seq_idx % self.steps["temp"] == 0).float().unsqueeze(1)
        mask_cgm = (seq_idx % self.steps["cgm"] == 0).float().unsqueeze(1)
        
        # Forward-fill Indices
        idx_hrv = (seq_idx // self.steps["hrv"]) * self.steps["hrv"]
        idx_act = (seq_idx // self.steps["actigraphy"]) * self.steps["actigraphy"]
        idx_temp = (seq_idx // self.steps["temp"]) * self.steps["temp"]
        idx_cgm = (seq_idx // self.steps["cgm"]) * self.steps["cgm"]
        
        # Apply forward-fill
        ff_hrv = base_hrv[idx_hrv]
        ff_act = base_act[idx_act]
        ff_temp = base_temp[idx_temp]
        ff_cgm = base_cgm[idx_cgm]
        
        # Construct raw matrix and mask
        x_raw = torch.cat([ff_hrv, ff_act, ff_temp, ff_cgm], dim=1)
        mask = torch.cat([mask_hrv, mask_act, mask_temp, mask_cgm], dim=1)
        
        return x_raw, mask
        
    def apply_therapy_shock(self, x_raw, mask, time_minutes):
        """
        Inject massive D_0 noise starting at time_minutes to simulate 
        ECM-breaking enzymes triggering systemic stress.
        """
        shock_idx = int(time_minutes * 60 * self.master_hz)
        if shock_idx >= self.seq_len:
            return x_raw
            
        # Spike HRV variance (Feature 0)
        hrv_noise = torch.randn(self.seq_len - shock_idx) * 150.0
        x_raw[shock_idx:, 0] += hrv_noise
        
        # Drive Actigraphy to near 0 (patient is incapacitated/resting) (Features 1,2,3)
        x_raw[shock_idx:, 1:4] = x_raw[shock_idx:, 1:4] * 0.1
        
        # Spike Core Temp (Feature 4)
        x_raw[shock_idx:, 4] += 1.5  # Fever response
        
        return x_raw

    def __getitem__(self, idx):
        # Generate base
        b_hrv, b_act, b_tmp, b_cgm = self._generate_base_signals()
        
        # Align & Forward Fill
        x_raw, mask = self._align_to_master_clock(b_hrv, b_act, b_tmp, b_cgm)
        
        # Randomly apply therapy shock to 20% of the dataset
        if torch.rand(1).item() < 0.2:
            # Apply shock halfway through
            shock_mins = (self.seq_len / self.master_hz / 60.0) / 2.0
            x_raw = self.apply_therapy_shock(x_raw, mask, shock_mins)
            
        # Target could be predicting biological state Z, we'll return a mock target for now
        # so it plugs directly into the existing runners.
        y_true = x_raw[:, 0:1] # Just auto-regress HRV for compatibility
        
        return {"x_raw": x_raw, "mask": mask, "y_true": y_true}

if __name__ == "__main__":
    # Verification
    # 2 minutes of data at 250Hz = 30000 steps
    dataset = HumanTelemetryLoader(size=1, seq_len=30000)
    batch = dataset[0]
    
    x = batch["x_raw"]
    m = batch["mask"]
    
    print(f"Data Shape: {x.shape}")
    print(f"Mask Shape: {m.shape}")
    
    # Assert feature dimensions
    assert x.shape[1] == 6, "Expected 6 features"
    
    # Check CGM masking (should only have 2 active ticks in 2 minutes: at t=0 and t=1min)
    cgm_active_count = m[:, 5].sum().item()
    print(f"CGM Active Ticks (expected 2): {cgm_active_count}")
    
    print("Verification Passed!")
