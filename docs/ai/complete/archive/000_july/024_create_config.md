Next, we establish src/config.py. This centralizes the scattered dimensional
variables into a single physics manifest.

```python
"""
MELD Architecture Global Configuration
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

    model_config = SettingsConfigDict(
        env_file=".env" if os.path.exists(".env") else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = MeldSettings()
```

nb this is an extraction so also remove the hardcoded constants from the model
definition files.
