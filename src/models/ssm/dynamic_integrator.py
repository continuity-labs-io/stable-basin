import torch
import torch.nn as nn
from src.models.ssm.async_ssm import pytorch_fused_scan

class DynamicIntegrator(nn.Module):
    """
    Pure PyTorch reference implementation for asynchronous continuous-time SSMs.
    Used strictly for gradcheck validation against the upcoming Triton Kernel.
    """
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # SSM Core Parameters (Diagonal Form)
        # A dictates memory decay. Bound strictly negative for stability.
        self.A_log = nn.Parameter(torch.log(torch.rand(dim, d_state) * 0.5 + 0.1))
        # B projects the incoming scalar event into the hidden state
        self.B_proj = nn.Parameter(torch.randn(dim, d_state) * 0.1)

    def forward(self, events: torch.Tensor, event_mask: torch.Tensor):
        """
        events: [Batch, Max_Events, 3] -> [Value, Sensor_ID, Timestamp]
        event_mask: [Batch, Max_Events] -> boolean mask for valid events
        """
        return pytorch_fused_scan(events, event_mask, self.A_log, self.B_proj, self.dim, self.d_state)
