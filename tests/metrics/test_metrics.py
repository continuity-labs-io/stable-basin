import pytest
import torch
from src.metrics.metrics import ThermodynamicMetrics


@pytest.fixture
def metrics_engine():
    return ThermodynamicMetrics()


def create_dummy_signal(time_steps, embed_dim):
    t = torch.linspace(0, 10, time_steps).unsqueeze(1)
    signal = torch.sin(t) * torch.ones(1, embed_dim)
    noise = 0.01 * torch.randn(time_steps, embed_dim)
    return signal + noise


def test_calculate_csd(metrics_engine):
    # Shape: [Time, Embed_Dim]
    z_seq = create_dummy_signal(20, 16)
    csd_scores = metrics_engine.calculate_csd(z_seq, window_size=5)

    assert len(csd_scores) == 20
    assert all(isinstance(score, float) for score in csd_scores)


def test_calculate_csd_zero_variance(metrics_engine):
    """
    Test that calculate_csd gracefully handles channels with zero variance
    (e.g., dead/quiescent biological channels) without returning NaN or 
    artificially inflated noise scores.
    """
    time_steps = 20
    embed_dim = 16
    z_seq = torch.zeros(time_steps, embed_dim)
    
    # Active channel 0 has a sine wave
    t = torch.linspace(0, 10, time_steps)
    z_seq[:, 0] = torch.sin(t)
    
    # The other 15 channels are completely flat (0.0 variance)
    csd_scores = metrics_engine.calculate_csd(z_seq, window_size=5)
    
    # Ensure it doesn't crash or return NaNs
    assert len(csd_scores) == 20
    assert not any(torch.isnan(torch.tensor(csd_scores)))
    assert all(isinstance(score, float) for score in csd_scores)


def test_calculate_csd_all_zero_variance(metrics_engine):
    """
    Test the extreme case where all channels are completely dead.
    """
    z_seq = torch.zeros(20, 16)
    csd_scores = metrics_engine.calculate_csd(z_seq, window_size=5)
    
    assert len(csd_scores) == 20
    assert not any(torch.isnan(torch.tensor(csd_scores)))
    # For all zero channels, it should return 0.0 according to our logic
    assert all(score == 0.0 for score in csd_scores)


@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part")
def test_calculate_ksm(metrics_engine):
    z_seq = create_dummy_signal(20, 16)
    ksm_scores = metrics_engine.calculate_ksm(z_seq, window_size=5)

    assert len(ksm_scores) == 20
    assert all(0.0 <= score <= 1.0 for score in ksm_scores)


def test_calculate_hysteresis(metrics_engine):
    z_baseline = create_dummy_signal(20, 16)
    z_perturbed = create_dummy_signal(20, 16) + 1.0

    area, divergence = metrics_engine.calculate_hysteresis(z_baseline, z_perturbed)

    assert isinstance(area, float)
    assert len(divergence) == 20
    assert area >= 0.0


def test_calculate_lle(metrics_engine):
    z_seq = create_dummy_signal(20, 16)
    lle_scores = metrics_engine.calculate_lle(z_seq, window_size=5)

    assert len(lle_scores) == 20
    assert all(isinstance(score, float) for score in lle_scores)


def test_calculate_cka(metrics_engine):
    z_seq1 = create_dummy_signal(20, 16)
    z_seq2 = create_dummy_signal(20, 16)

    cka_score = metrics_engine.calculate_cka(z_seq1, z_seq2)
    assert isinstance(cka_score, float)
