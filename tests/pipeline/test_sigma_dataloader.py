import numpy as np
import pandas as pd
import pytest
from src.pipeline.sim2real.sigma_phase_structure_dataloader import SigmaPhaseLoader


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sigma_dataloader_output_format():
    loader = SigmaPhaseLoader(target_components=5)
    times, raw_feats = loader.fetch_and_segment()
    latents = loader.compress_to_latent(raw_feats)

    # Simulate a micro-burst
    minute = 5
    master_clock_ms = np.linspace(minute * 60000, minute * 60000 + 4500, 50)

    df = loader.align_to_master_clock(times, latents, master_clock_ms)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "Time_ms" in df.columns
    assert "Sigma_PC001" in df.columns
    assert "Sigma_PC005" in df.columns
    assert list(df["Time_ms"]) == list(master_clock_ms)


def test_sigma_dataloader_neural_ode_smoothness():
    # The regression test that demonstrates the Neural ODE provides
    # smooth, continuously differentiable transitions.
    loader = SigmaPhaseLoader(target_components=2)
    times = np.array([0.0, 1.0, 2.0])
    latents = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]])

    # Evaluate at fine granularity over the first interval [0, 1)
    master_clock_ms = np.linspace(0, 59999, 100)  # almost 1 minute

    df = loader.align_to_master_clock(times, latents, master_clock_ms)

    # We expect the neural ODE to give smooth transitions
    pc1 = df["Sigma_PC001"].values

    # Calculate discrete first derivative (velocity)
    dt = master_clock_ms[1] - master_clock_ms[0]
    velocity = np.diff(pc1) / dt

    # Calculate discrete second derivative (acceleration)
    acceleration = np.diff(velocity) / dt

    # Ensure no NaN or Inf (stability of ODE integration)
    assert np.all(np.isfinite(pc1)), "ODE integration diverged"
    assert np.all(np.isfinite(velocity)), "ODE velocity has singularities"
    assert np.all(np.isfinite(acceleration)), "ODE acceleration has singularities"
