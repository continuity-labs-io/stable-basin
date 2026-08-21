import torch
import torch.nn as nn
from src.config import settings

class GEVIEncoder(nn.Module):
    """
    Temporally pools high-frequency bioelectric data streams (GEVI) down to 
    match a lower frequency optical framerate using a 1D Convolution.
    """

    def __init__(
        self,
        gevi_sample_rate=settings.GEVI_HZ,
        target_clock_hz=settings.OPTICS_HZ,
        gevi_dim=settings.MAMBA_D_STATE,
    ):
        super().__init__()
        self.compression_ratio = int(gevi_sample_rate / target_clock_hz)
        
        self.compressor = nn.Conv1d(
            in_channels=1,
            out_channels=gevi_dim,
            kernel_size=self.compression_ratio,
            stride=self.compression_ratio,
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, 1, Total_Steps) representing high-frequency GEVI data
        Returns:
            compressed: Tensor of shape (Batch, Target_Time_Steps, gevi_dim)
        """
        compressed = self.compressor(x)
        return compressed.transpose(1, 2)
