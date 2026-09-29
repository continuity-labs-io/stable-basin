import pytest
from src.core.threshold_monitor import MetricThresholdMonitor
from src.metrics import TimeSeriesStabilityMetrics
from src.harness.sensor_fusion_predictor import SensorFusionPredictor


def test_threshold_monitor_state_machine():
    # Initialize components
    engine = SensorFusionPredictor("masr_ssm", modality_dims=[6], d_model=32)
    metrics = TimeSeriesStabilityMetrics()
    controller = MetricThresholdMonitor(engine, metrics, hysteresis_frames=3)

    # Testing Nominal State
    res = controller.evaluate_safety_margins(ksm_score=0.95, csd_score=1.0, plv_score=1.0)
    assert res["action"] == "OK"

    # Testing Instability Spike (Hysteresis Protection)
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0, plv_score=1.0)
    assert res["action"] == "WARNING"  # Frame 1
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0, plv_score=1.0)
    assert res["action"] == "WARNING"  # Frame 2

    # Testing Full Bifurcation Abort
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0, plv_score=1.0)
    assert res["action"] == "ALARM"  # Frame 3 (Threshold hit)
