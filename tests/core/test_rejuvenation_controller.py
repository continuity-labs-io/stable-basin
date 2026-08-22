import pytest
from src.core.rejuvenation_controller import RejuvenationFlightController
from src.metrics.metrics import ThermodynamicMetrics
from src.icebox.models.ssm.masr_mamba import MaskAwareMamba

def test_rejuvenation_controller_state_machine():
    # Initialize components
    engine = MaskAwareMamba(input_dim=6, d_model=32, mask_aware=True)
    metrics = ThermodynamicMetrics()
    controller = RejuvenationFlightController(engine, metrics, hysteresis_frames=3)
    
    # Testing Nominal State
    res = controller.evaluate_safety_margins(ksm_score=0.95, csd_score=1.0)
    assert res["action"] == "MAINTAIN_INFUSION"
    
    # Testing Instability Spike (Hysteresis Protection)
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0)
    assert res["action"] == "WARNING" # Frame 1
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0)
    assert res["action"] == "WARNING" # Frame 2
    
    # Testing Full Bifurcation Abort
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0)
    assert res["action"] == "EMERGENCY_ABORT" # Frame 3 (Threshold hit)
