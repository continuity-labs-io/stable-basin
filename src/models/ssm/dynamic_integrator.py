import torch
import torch.nn as nn
from src.models.ssm.async_ssm import pytorch_fused_scan
from .physics import create_a_matrix

class DynamicIntegrator(nn.Module):
    """
    Pure PyTorch reference implementation for asynchronous continuous-time SSMs.
    Used strictly for gradcheck validation against the upcoming Triton Kernel.
    """
    def __init__(self, dim: int, d_state: int = 16, a_init_type: str = "random"):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # SSM Core Parameters (Diagonal Form)
        # A dictates memory decay. Bound strictly negative for stability.
        self.A_init = create_a_matrix(init_type=a_init_type, shape=(dim, d_state))
        # B projects the incoming scalar event into the hidden state
        self.B_proj = nn.Parameter(torch.randn(dim, d_state) * 0.1)

    def forward(self, events: torch.Tensor, event_mask: torch.Tensor):
        """
        events: [Batch, Max_Events, 3] -> [Value, Sensor_ID, Timestamp]
        event_mask: [Batch, Max_Events] -> boolean mask for valid events
        """
        return pytorch_fused_scan(events, event_mask, self.A_init.A_log, self.B_proj, self.dim, self.d_state)
