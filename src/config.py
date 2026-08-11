"""
Stable Basin Global Configuration (MeldEngine Default)
"""

import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MeldSettings(BaseSettings):
    # --- TEMPORAL RESOLUTION CONSTANTS ---
    OPTICS_HZ: int = Field(default=100, description="Optical imaging framerate")
    GEVI_HZ: int = Field(default=20000, description="High-frequency bioelectric sampling rate")
    EPHYS_BIN_MS: int = Field(default=50, description="Electrophysiology bin size in milliseconds")

    # --- THERMODYNAMIC METRIC WINDOWS ---
    KSM_WINDOW_SIZE: int = Field(default=4, description="Dynamic Mode Decomposition sliding window")
    CSD_WINDOW_SIZE: int = Field(default=3, description="Critical Slowing Down sliding window")
    LLE_WINDOW_SIZE: int = Field(default=4, description="Local Lyapunov Exponent sliding window")

    # --- PHYSICS LOSS PARAMETERS ---
    LIPSCHITZ_CONSTANT: float = Field(
        default=1.5, description="Upper bound for thermodynamic state change"
    )
    MELD_ALPHA: float = Field(default=1.0, description="Forecast MSE Weight")
    MELD_BETA: float = Field(default=0.1, description="Lipschitz Penalty Weight")
    MELD_GAMMA: float = Field(default=0.5, description="Time-Reversal Error Weight")

    # --- ARCHITECTURE DIMENSIONS ---
    MAMBA_D_MODEL: int = Field(default=256, description="Hidden dimension of the Mamba-2 block")
    MAMBA_D_STATE: int = Field(default=64, description="State dimension size")

    # --- DEMO SPECIFIC CONSTANTS ---
    DURATION_SECONDS: float = Field(default=1.0, description="Simulation duration in seconds")
    OPTICS_DIM: int = Field(default=32, description="Optical spatial dimension")
    GEVI_DIM: int = Field(default=1, description="GEVI spatial dimension")

    PUMP_ARTIFACT_HZ: float = Field(default=2.0)
    OPTICS_PUMP_AMPLITUDE: float = Field(default=2.0)
    GEVI_PUMP_AMPLITUDE: float = Field(default=50.0)

    SPIKE_AMPLITUDE: float = Field(default=100.0)
    SPIKE_PROBABILITY_PER_WINDOW: float = Field(default=0.15)

    EVENT_BOUNDARY_OPTICS: int = Field(default=50)

    TOXIC_SHOCK_NOISE_STD: float = Field(default=5.0)
    CORROSION_DRIFT_STD: float = Field(default=2.0)

    GEVI_COMPRESSOR_OUT_CHANNELS: int = Field(default=16)
    GEVI_COMPRESSOR_KERNEL_SIZE: int = Field(default=200)
    GEVI_COMPRESSOR_STRIDE: int = Field(default=200)

    TRAIN_ITERATIONS: int = Field(default=15)
    TRAIN_BATCH_SIZE: int = Field(default=4)
    LEARNING_RATE: float = Field(default=1e-3)

    @property
    def OPTICS_FRAMES(self) -> int:
        return int(self.OPTICS_HZ * self.DURATION_SECONDS)

    @property
    def GEVI_FRAMES(self) -> int:
        return int(self.GEVI_HZ * self.DURATION_SECONDS)

    @property
    def SPIKE_WINDOW_STEPS(self) -> int:
        return int(self.GEVI_HZ / self.OPTICS_HZ)

    @property
    def SPIKE_WIDTH_STEPS(self) -> int:
        return int(self.GEVI_HZ * 0.001)

    @property
    def EVENT_BOUNDARY_GEVI(self) -> int:
        return self.EVENT_BOUNDARY_OPTICS * self.SPIKE_WINDOW_STEPS

    model_config = SettingsConfigDict(
        env_file=".env" if os.path.exists(".env") else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = MeldSettings()
