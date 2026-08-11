import torch
import logging

# Configure logger
logger = logging.getLogger("RejuvenationFlightController")
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

class RejuvenationFlightController:
    """
    Biological Flight Computer for Closed-Loop Rejuvenation Therapies.
    Monitors thermodynamic stability and physically actuates the IV pump 
    to prevent saddle-node bifurcations.
    """
    def __init__(self, engine, metrics, hysteresis_frames=3):
        self.engine = engine
        self.metrics = metrics
        
        # Dampening / Hysteresis State
        self.hysteresis_frames = hysteresis_frames
        self.critical_count = 0
        
        # Safety Thresholds
        self.CRITICAL_KSM_THRESHOLD = 0.85
        self.MAX_CSD_VARIANCE = 3.0
        
        # Actuation state
        self.current_state = "STATE_NOMINAL"
        
    def process_telemetry_chunk(self, x_raw, mask):
        """
        Pushes multi-modal telemetry through the physics engine.
        x_raw shape: [Time, Features]
        mask shape: [Time, Features]
        """
        # Engine expects batch dimension: [1, Time, Features]
        x_nan = x_raw.clone()
        x_nan[mask == 0] = float('nan')
        x_nan = x_nan.unsqueeze(0)
        
        # Extract Latent State
        z_batch = self.engine.get_hidden_states(x_nan)
        z_seq = z_batch[0] # [Time, Embed_Dim]
        
        # Calculate Thermodynamic Metrics
        # These methods return lists of length Time. We just need the latest frame.
        csd_scores = self.metrics.calculate_csd(z_seq)
        ksm_scores = self.metrics.calculate_ksm(z_seq)
        
        latest_csd = csd_scores[-1]
        latest_ksm = ksm_scores[-1]
        
        return latest_ksm, latest_csd

    def _actuate_iv_pump(self, action, ksm_score, csd_score):
        """
        Hardware Webhook to physically control payload delivery.
        """
        metrics_str = f"[Koopman Stability Metric (KSM): {ksm_score:.3f} | Critical Slowing Down (CSD): {csd_score:.3f}]"
        
        if action == "EMERGENCY_ABORT":
            logger.critical(f"[IV_PUMP] Emergency abort triggered. Terminating therapy. {metrics_str}")
        elif action == "MAINTAIN_INFUSION":
            logger.info(f"[IV_PUMP] Infusion running nominally. Stabilizing... {metrics_str}")
        elif action == "WARNING":
            logger.warning(f"[IV_PUMP] Instability detected. Holding flow rate... {metrics_str}")
            
    def evaluate_safety_margins(self, ksm_score, csd_score):
        """
        PID / State Machine logic to determine hardware actuation.
        """
        result = {}
        
        if ksm_score < self.CRITICAL_KSM_THRESHOLD or csd_score > self.MAX_CSD_VARIANCE:
            self.critical_count += 1
            if self.critical_count >= self.hysteresis_frames:
                self.current_state = "STATE_BIFURCATION_DANGER"
                result = {
                    "action": "EMERGENCY_ABORT",
                    "status": "CRITICAL",
                    "reason": "Saddle-node bifurcation imminent."
                }
            else:
                result = {
                    "action": "WARNING",
                    "status": "DEGRADING",
                    "reason": f"Instability frames: {self.critical_count}/{self.hysteresis_frames}"
                }
        else:
            if ksm_score > 0.92:
                # Reset counter if strongly nominal
                self.critical_count = 0
            else:
                # Slowly decay counter if borderline
                self.critical_count = max(0, self.critical_count - 1)
                
            self.current_state = "STATE_NOMINAL"
            result = {
                "action": "MAINTAIN_INFUSION",
                "status": "SAFE",
                "reason": "Homeostasis intact."
            }
            
        self._actuate_iv_pump(result["action"], ksm_score, csd_score)
        return result

if __name__ == "__main__":
    from src.metrics.metrics import ThermodynamicMetrics
    from src.models.ssm.meld_engine import MeldEngine
    
    # Initialize components
    engine = MeldEngine(input_dim=6, d_model=32, mask_aware=True)
    metrics = ThermodynamicMetrics()
    controller = RejuvenationFlightController(engine, metrics, hysteresis_frames=3)
    
    print("--- Testing Nominal State ---")
    res = controller.evaluate_safety_margins(ksm_score=0.95, csd_score=1.0)
    assert res["action"] == "MAINTAIN_INFUSION"
    
    print("--- Testing Instability Spike (Hysteresis Protection) ---")
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0)
    assert res["action"] == "WARNING" # Frame 1
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0)
    assert res["action"] == "WARNING" # Frame 2
    
    print("--- Testing Full Bifurcation Abort ---")
    res = controller.evaluate_safety_margins(ksm_score=0.70, csd_score=4.0)
    assert res["action"] == "EMERGENCY_ABORT" # Frame 3 (Threshold hit)
    
    print("\nVerification Passed!")
