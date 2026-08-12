import pytest
import torch
from unittest.mock import MagicMock
from src.metrics.autopsy_engine import ThermodynamicAutopsyEngine


def test_autopsy_engine_generation():
    # Mock model that returns a static attribution matrix
    mock_model = MagicMock()
    # The expected shape for attribution matrix is [1, Time, 114]
    mock_attribution = torch.ones(1, 50, 114)
    # Give high importance to feature 10 to test if it detects it
    mock_attribution[0, :, 10] = 100.0
    
    from src.metrics.attribution_engine import AttributionEngine
    AttributionEngine.get_instance().set_strategy(lambda m, x, t: mock_attribution)

    engine = ThermodynamicAutopsyEngine(mock_model)
    x_sequence = torch.randn(1, 50, 114)
    crash_time_step = 45

    report = engine.generate_autopsy(x_sequence, crash_time_step)

    assert report["status"] == "CRITICAL_FAILURE_PREDICTED"
    assert report["predicted_crash_time"] == "T=45"
    assert "anomaly_ontology" in report
    assert report["anomaly_ontology"]["primary_latent_driver"] == engine.feature_names[10]

    # Cleanup
    AttributionEngine.get_instance().reset_strategy()

    causal_trace = report["anomaly_ontology"]["causal_trace"]
    assert len(causal_trace) == 3  # Top 3 critical steps prior to crash
    for step in causal_trace:
        assert step["flagged_input"] == engine.feature_names[10]
