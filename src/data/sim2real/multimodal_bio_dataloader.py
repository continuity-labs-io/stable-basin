import numpy as np
import pandas as pd
import torch

import logging

logger = logging.getLogger(__name__)


def generate_sim2real_stub(total_minutes=15, crash_minute=10):
    """
    Sim2Real Sim2Real Engine
    Generates a 15-minute multi-scale tensor for 1 cell.
    Demonstrates the Hierarchical Entrainment Crash.
    """
    logger.info("Booting Sim2Real Physics Engine...")

    # 1. TIME CONSTANTS (The Multi-Scale Problem)
    burst_freq_hz = 500
    burst_duration_sec = 4.5
    frames_per_burst = int(burst_duration_sec * burst_freq_hz)  # 2250 frames per burst
    interval_min = 5
    total_intervals = total_minutes // interval_min  # 3 intervals in 15 mins

    time_ms = []
    current_time = 0.0
    for _ in range(total_intervals):
        # Generate the exact 4.5-second burst window
        burst_times = np.linspace(
            current_time,
            current_time + (burst_duration_sec * 1000),
            frames_per_burst,
            endpoint=False,
        )
        time_ms.extend(burst_times)
        current_time += interval_min * 60 * 1000  # Advance 5 mins

    df = pd.DataFrame({"Time_ms": time_ms})
    df["Minute"] = df["Time_ms"] / 60000.0
    total_rows = len(df)

    # ---------------------------------------------------------
    # [TASK 1] SIGMA (Phase Structure - 100D)
    # ML TEAM: Replace random drift with CZ Biohub real PCA embeddings
    # ---------------------------------------------------------
    logger.info("Generating Sigma (100D)...")
    phase_cols = [f"PC{i:03d}" for i in range(1, 101)]
    # Synthetic: smooth drift, accelerating after crash
    drift = np.cumsum(np.random.normal(0, 0.005, (total_rows, 100)), axis=0)
    crash_penalty = np.where(df["Minute"].values[:, None] >= crash_minute, -0.02, 0)
    phase_df = pd.DataFrame(drift + crash_penalty, columns=phase_cols)

    # ---------------------------------------------------------
    # [TASK 2] OMEGA (Voltage - 2D)
    # ML TEAM: Replace with Michael Lin ASAP6c/mScarlet3 real trace distributions
    # ---------------------------------------------------------
    logger.info("Generating Omega (2D)...")
    # Red Baseline (Stable)
    volt_red = np.random.normal(1.0, 0.01, total_rows)

    # Green Active (Variance Explosion at crash)
    is_crashing = df["Minute"] >= crash_minute
    # Jitter spikes from 0.05 to 0.50 (10x Variance Explosion)
    jitter = np.where(
        is_crashing, np.random.normal(0, 0.50, total_rows), np.random.normal(0, 0.05, total_rows)
    )
    volt_grn = 1.0 + jitter
    bioelectric_df = pd.DataFrame({"VoltRed": volt_red, "VoltGrn": volt_grn})

    # ---------------------------------------------------------
    # [TASK 3] PSI (RNA Software - 12D)
    # ML TEAM: Replace Poisson generator with Gao/Wyss-Coray real scRNA-seq counts
    # ---------------------------------------------------------
    logger.info("Generating Psi (12D)...")
    rna_cols = [
        "RNA_NFE2L2",
        "RNA_TP53",
        "RNA_CDKN2A",
        "RNA_TREM2",
        "RNA_APOE",
        "RNA_IL6",
        "RNA_GFAP",
        "RNA_MAPT",
        "RNA_NANOG",
        "RNA_CASP3",
        "RNA_CAS13",
        "RNA_GAPDH",
    ]

    # Fill with NaNs (The shutter is closed 99% of the time)
    rna_df = pd.DataFrame(np.nan, index=df.index, columns=rna_cols)

    # Only sample at the very first frame of each 5-min burst
    sample_indices = np.arange(0, total_rows, frames_per_burst)

    for idx in sample_indices:
        minute = df.loc[idx, "Minute"]
        if minute < crash_minute:
            # Healthy baseline (Normal Poisson counts)
            rna_df.loc[idx, rna_cols] = np.random.poisson(lam=5, size=12)
        else:
            # Transcriptomic Hysteresis (Panic genes get stuck ON)
            rna_df.loc[idx, rna_cols] = np.random.poisson(lam=5, size=12)
            # TP53, IL6, and CASP3 spike massively
            rna_df.loc[idx, ["RNA_TP53", "RNA_IL6", "RNA_CASP3"]] = np.random.poisson(
                lam=50, size=3
            )

    # Reorder for clean UX
    cols = ["Time_ms", "Minute", "VoltGrn", "VoltRed"] + rna_cols + phase_cols
    df = pd.concat([df, bioelectric_df, rna_df, phase_df], axis=1)[cols]

    logger.info(f"Success! Generated Sim2Real Tensor: {total_rows} rows x {len(cols)} columns.")
    return df


class MultimodalBioDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that wraps the Sim2Real stub generator.
    It returns sequences of length `sequence_length` containing:
    - Phase Structure (Sigma)
    - Bioelectric Voltage (Omega)
    - Transcriptomic Counts (Psi)
    """
    def __init__(self, total_minutes=15, crash_minute=10, sequence_length=100):
        super().__init__()
        self.sequence_length = sequence_length
        self.df = generate_sim2real_stub(total_minutes, crash_minute)
        
        # Pre-extract columns
        self.phase_cols = [c for c in self.df.columns if c.startswith("PC")]
        self.volt_cols = ["VoltGrn", "VoltRed"]
        self.rna_cols = [c for c in self.df.columns if c.startswith("RNA_")]
        
    def __len__(self):
        return len(self.df) - self.sequence_length + 1
        
    def __getitem__(self, idx):
        window = self.df.iloc[idx : idx + self.sequence_length]
        
        phase_tensor = torch.tensor(window[self.phase_cols].values, dtype=torch.float32)
        volt_tensor = torch.tensor(window[self.volt_cols].values, dtype=torch.float32)
        rna_tensor = torch.tensor(window[self.rna_cols].values, dtype=torch.float32)
        
        return {
            "phase": phase_tensor,
            "voltage": volt_tensor,
            "rna": rna_tensor
        }
