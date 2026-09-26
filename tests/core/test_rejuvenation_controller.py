import pytest
from src.core.rejuvenation_controller import RejuvenationFlightController
from src.metrics import ThermodynamicMetrics
from src.harness.sensor_fusion_predictor import SensorFusionPredictor


def test_rejuvenation_controller_state_machine():
    # Initialize components
    engine = SensorFusionPredictor("masr_ssm", modality_dims=[6], d_model=32)
    metrics = ThermodynamicMetrics()
    controller = RejuvenationFlightController(engine, metrics, hysteresis_frames=3)

    # Testing Nominal State
    res = controller.evaluate_safety_margins(ksm_score=0.95, csd_score=1.0, plv_score=1.0)
    assert res["action"] == "MAINTAIN_INFUSION"

    # Testing Instability Spike (Hysteresis Protection)
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0, plv_score=1.0)
    assert res["action"] == "WARNING"  # Frame 1
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0, plv_score=1.0)
    assert res["action"] == "WARNING"  # Frame 2

    # Testing Full Bifurcation Abort
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0, plv_score=1.0)
    assert res["action"] == "EMERGENCY_ABORT"  # Frame 3 (Threshold hit)
