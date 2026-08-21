import pytest
from src.data.sim2real.human_telemetry_dataloader import HumanTelemetryLoader

def test_human_telemetry_dataloader():
    # 2 minutes of data at 250Hz = 30000 steps
    dataset = HumanTelemetryLoader(size=1, seq_len=30000)
    batch = dataset[0]
    
    x = batch["x_raw"]
    m = batch["mask"]
    
    # Assert feature dimensions
    assert x.shape == (30000, 6)
    assert m.shape == (30000, 6)
    
    # Check CGM masking (should only have 2 active ticks in 2 minutes: at t=0 and t=1min)
    cgm_active_count = m[:, 5].sum().item()
    assert cgm_active_count == 2
