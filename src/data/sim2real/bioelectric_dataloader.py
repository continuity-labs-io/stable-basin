import numpy as np
import pandas as pd
import scipy.signal as signal

import logging

logger = logging.getLogger(__name__)


class BioelectricLoader:
    def __init__(self, sample_rate_hz=500, crash_minute=10):
        """
        Bioelectric Dataloader V1
        Scaffold for ingesting high-speed kilohertz voltage imaging (GEVI) traces,
        applying ratiometric motion-cancellation, and detecting Variance Explosions.
        """
        self.sample_rate_hz = sample_rate_hz
        self.crash_minute = crash_minute
        # Target proxy: Neurodata Without Borders (NWB) GEVI datasets or Allen Brain Observatory
        self.source_url = "s3://stanford-lin-lab/asap6c-in-vivo-traces.nwb"
        logger.info(
            f"[INIT] Omega Dataloader targeting proxy: {self.source_url} at {self.sample_rate_hz}Hz"
        )

    def fetch_gevi_traces(self, total_frames):
        """
        [TASK 1] DATA ENGINEER: THE NWB INGEST
        Connect to a public Neurodata Without Borders (.nwb) file containing real patch-clamp
        or high-speed optical voltage traces. Extract the raw action potential waveforms.
        """
        logger.info("[NETWORK] Mocking kilohertz voltage trace ingestion...")

        # Simulating baseline membrane potential with sparse action potentials

        # Base membrane voltage (normalized to 1.0)
        base_voltage = np.ones(total_frames)

        # Simulate sparse action potentials (spikes) using a thresholded random uniform
        spikes = np.where(np.random.uniform(0, 1, total_frames) > 0.99, 0.4, 0)

        # Convolve with a quick exponential decay to make it look like a biological spike
        kernel = signal.windows.exponential(int(self.sample_rate_hz * 0.05), tau=3.0)
        biological_spikes = signal.convolve(spikes, kernel, mode="same")

        return base_voltage + biological_spikes

    def apply_hardware_physics(self, pure_voltage, time_minutes):
        """
        [TASK 2] DATA ENGINEER: RATIOMETRIC MOTION ARTIFACTS
        Organoids move and swell. We simulate this macroscopic wobble, apply it to BOTH
        the Green (ASAP6c) and Red (mScarlet3) channels, and prove that division cancels it out.
        """
        logger.info(
            "[PHYSICS] Injecting mechanical tissue drift and thermodynamic crash variance..."
        )

        # 1. The Mechanical Wobble (Cells shifting out of focus)
        # A slow, wandering sine wave that hits BOTH color channels equally
        wobble = 0.2 * np.sin(2 * np.pi * time_minutes / 3.0)

        # 2. The Thermodynamic Crash (Variance Explosion)
        # As the Phase (Sigma) loses binding energy, the voltage baseline goes chaotic
        is_crashing = time_minutes >= self.crash_minute
        baseline_jitter = np.where(
            is_crashing,
            np.random.normal(0, 0.25, len(pure_voltage)),
            np.random.normal(0, 0.01, len(pure_voltage)),
        )

        # Construct the 2 Dimensions
        df = pd.DataFrame()
        # RED (D002): Baseline fluorophore. Blind to voltage. Only sees wobble.
        df["VoltRed"] = 1.0 + wobble

        # GREEN (D001): Voltage sensor. Sees voltage + wobble + crash jitter.
        df["VoltGrn"] = pure_voltage + wobble + baseline_jitter

        # [TASK 3] EDGE COMPUTE SIMULATION
        # The AI doesn't get raw green. It gets Normalized Input = Green / Red.
        # Notice how the `wobble` mathematically vanishes.
        df["VoltNormalized"] = df["VoltGrn"] / df["VoltRed"]

        return df

    def align_to_master_clock(self, df_omega, master_time_ms):
        """
        Extracts the exact 4.5-second micro-bursts to match the MELD timing.
        """
        df_omega["Time_ms"] = master_time_ms
        # Reorder columns
        return df_omega[["Time_ms", "VoltGrn", "VoltRed", "VoltNormalized"]]


# ==========================================
# EXECUTION (Drop this in the Jupyter Notebook)
# ==========================================
