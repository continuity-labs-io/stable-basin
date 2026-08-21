import numpy as np
import pytest
from src.data.sim2real.omega_bioelectric_dataloader import OmegaBioelectricLoader

def test_omega_bioelectric_dataloader():
    master_clock_ms = []
    time_minutes = []

    for minute in [0, 5, 10, 15]:
        # 4.5 seconds of 500Hz = 2250 frames per burst
        burst = np.linspace(minute * 60000, minute * 60000 + 4500, 2250)
        master_clock_ms.extend(burst)
        time_minutes.extend([minute + (t / 60000) for t in np.linspace(0, 4500, 2250)])

    master_clock_ms = np.array(master_clock_ms)
    time_minutes = np.array(time_minutes)
    
    loader = OmegaBioelectricLoader(sample_rate_hz=500, crash_minute=10)
    raw_spikes = loader.fetch_gevi_traces(total_frames=len(master_clock_ms))
    df_omega = loader.apply_hardware_physics(raw_spikes, time_minutes)
    final_df = loader.align_to_master_clock(df_omega, master_clock_ms)
    
    assert final_df is not None
    assert len(final_df) == len(master_clock_ms)
