import os
import torch
import numpy as np
import jax.numpy as jnp
from unittest.mock import patch
import tempfile
import pytest

from src.echo.benchmarks.waddington_collapse import run_waddington_collapse_benchmark

def test_waddington_collapse_ebet():
    """
    Synthetic Crash Test:
    Mocks the HD-MEA dataset to simulate a crash, forces the Hessian trace 
    to collapse earlier, and asserts the Energy Basin Escape Time (EBET) is computed correctly.
    """
    seq_len = 200
    input_dim = 1024
    
    # 1. Synthetic Crash Test Data
    # Frames 0-100: high-variance sinusoidal noise
    t1 = torch.linspace(0, 10 * np.pi, 100).unsqueeze(1).repeat(1, input_dim)
    noise = torch.randn(100, input_dim) * 0.5
    healthy_data = torch.sin(t1) + noise
    
    # Frames 100-200: flat zeros
    crash_data = torch.zeros(100, input_dim)
    
    synthetic_data = torch.cat([healthy_data, crash_data], dim=0)
    
    # 2. Pre-calculate where the electrical crash frame will be detected
    data_np = synthetic_data.numpy()
    rolling_var = np.array([np.var(data_np[max(0, i-50):i+1]) for i in range(len(data_np))])
    baseline_var = np.mean(rolling_var[50:100])
    var_threshold = 0.5 * baseline_var
    
    electrical_crash_frame = -1
    for i in range(100, len(rolling_var)):
        if rolling_var[i] < var_threshold:
            electrical_crash_frame = i
            break
            
    if electrical_crash_frame == -1:
        pytest.fail("Failed to detect synthetic electrical crash.")

    # 3. Force thermodynamic collapse 20 frames before the electrical crash
    thermo_collapse = electrical_crash_frame - 20
    
    mock_trace = np.ones(seq_len) * 100.0
    mock_trace[thermo_collapse:] = 10.0  # Drop well below 50% threshold
    
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(tmpdir, "waddington_collapse.png")
        
        # Patch the tracker's batch method so we don't rely on random weights aligning with the crash
        with patch('src.echo.benchmarks.waddington_collapse.HessianCurvatureTracker.batch_calculate_curvature') as mock_batch_calc:
            mock_batch_calc.return_value = {
                "hessian_trace": jnp.array(mock_trace)
            }
            
            # Execute benchmark
            ebet = run_waddington_collapse_benchmark(synthetic_data, output_plot=plot_path)
            
            # Assert EBET is precisely 20
            assert ebet == 20, f"Expected EBET of 20, got {ebet}"
            
            # Assert script executed without throwing shape errors and generated the plot
            assert os.path.exists(plot_path), "Plot file was not generated"

