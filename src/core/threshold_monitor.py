import torch
import logging
from src.metrics.spectral import SpectralMetrics

# Configure logger
logger = logging.getLogger("MetricThresholdMonitor")
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class MetricThresholdMonitor:
    """
    Monitors thermodynamic stability and checks metrics against safety thresholds.
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
        self.CRITICAL_PLV_THRESHOLD = 0.65

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
        x_nan[mask == 0] = float("nan")
        x_nan = x_nan.unsqueeze(0)

        # Extract Latent State
        z_batch = self.engine.get_hidden_states(x_nan)
        z_seq = z_batch[0]  # [Time, Embed_Dim]

        # Calculate Thermodynamic Metrics
        # These methods return lists of length Time. We just need the latest frame.
        csd_scores = self.metrics.calculate_csd(z_seq)
        ksm_scores = self.metrics.calculate_ksm(z_seq)

        # Calculate Spectral Metrics (PLV across adjacent spatial dimensions)
        spectral = SpectralMetrics()
        plv_array = spectral.calculate_plv(z_seq[:, :-1], z_seq[:, 1:])
        plv_score = plv_array.mean().item()

        latest_csd = csd_scores[-1]
        latest_ksm = ksm_scores[-1]

        return latest_ksm, latest_csd, plv_score

    def _log_decision(self, action, ksm_score, csd_score, plv_score):
        """
        Log the status of thermodynamic metrics.
        """
        metrics_str = f"[KSM: {ksm_score:.3f} | CSD: {csd_score:.3f} | PLV: {plv_score:.3f}]"

        if action == "ALARM":
            logger.critical(f"Critical instability thresholds exceeded. {metrics_str}")
        elif action == "OK":
            logger.info(f"Metrics within nominal range. {metrics_str}")
        elif action == "WARNING":
            logger.warning(f"Borderline metrics detected. {metrics_str}")

    def evaluate_safety_margins(self, ksm_score, csd_score, plv_score):
        """
        Check metrics against predefined safety margins.
        """
        result = {}

        if (
            ksm_score < self.CRITICAL_KSM_THRESHOLD
            or csd_score > self.MAX_CSD_VARIANCE
            or plv_score < self.CRITICAL_PLV_THRESHOLD
        ):
            self.critical_count += 1
            if self.critical_count >= self.hysteresis_frames:
                self.current_state = "STATE_ALARM"
                result = {
                    "action": "ALARM",
                    "status": "CRITICAL",
                    "reason": "Instability or spectral decoherence detected.",
                }
            else:
                result = {
                    "action": "WARNING",
                    "status": "DEGRADING",
                    "reason": f"Instability frames: {self.critical_count}/{self.hysteresis_frames}",
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
                "action": "OK",
                "status": "SAFE",
                "reason": "Metrics are stable.",
            }

        self._log_decision(result["action"], ksm_score, csd_score, plv_score)
        return result
