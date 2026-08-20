import torch
import torch.nn as nn
from src.models.ssm.triton_fused_scan import HAS_TRITON, MaskAwareFusedScan
from .physics import create_a_matrix

def pytorch_fused_scan(
    events: torch.Tensor, 
    event_mask: torch.Tensor, 
    A_log: torch.Tensor, 
    B_proj: torch.Tensor, 
    dim: int, 
    d_state: int
) -> torch.Tensor:
    """
    Pure PyTorch implementation of the Mask Aware Fused Scan.
    """
    B, max_events, _ = events.shape
    device = events.device
    
    # The true Biological Manifold: [Batch, Dim, D_State]
    h = torch.zeros(B, dim, d_state, device=device)
    
    # Track the exact timestamp each sensor was last updated: [Batch, Dim]
    last_t = torch.zeros(B, dim, device=device)
    
    A = -torch.exp(A_log) # Ensure A is negative [Dim, D_State]
    h_sequence = []

    for i in range(max_events):
        vals = events[:, i, 0]              # [Batch]
        sensor_ids = events[:, i, 1].long() # [Batch]
        timestamps = events[:, i, 2]        # [Batch]
        valid = event_mask[:, i]            # [Batch]

        # Prevent Out-Of-Bounds Gather on invalid padded elements
        safe_sensor_ids = torch.clamp(sensor_ids, 0, dim - 1)

        # 1. Calculate Dynamic Delta T
        last_t_sensor = last_t.gather(1, safe_sensor_ids.unsqueeze(1)).squeeze(1)
        dt = timestamps - last_t_sensor
        
        # 2. Continuous-Time Discretization (Zero-Order Hold)
        A_active = A[safe_sensor_ids]             # [Batch, d_state]
        B_active = B_proj[safe_sensor_ids]        # [Batch, d_state]
        
        A_bar = torch.exp(A_active * dt.unsqueeze(1))
        B_bar = ((A_bar - 1.0) / (A_active - 1e-8)) * B_active * vals.unsqueeze(1)
        
        # 3. Apply State Update (Orthogonal Routing)
        h_active = h.gather(1, safe_sensor_ids.unsqueeze(1).unsqueeze(2).expand(-1, -1, d_state)).squeeze(1)
        h_new = A_bar * h_active + B_bar
        
        # 4. Vectorized Scatter (Safely mapping back into full tensor)
        h_scatter = h.scatter(1, safe_sensor_ids.unsqueeze(1).unsqueeze(2).expand(-1, -1, d_state), h_new.unsqueeze(1))
        h = torch.where(valid.unsqueeze(1).unsqueeze(2), h_scatter, h)
        
        # Scatter new timestamp
        last_t_scatter = last_t.scatter(1, safe_sensor_ids.unsqueeze(1), timestamps.unsqueeze(1))
        last_t = torch.where(valid.unsqueeze(1), last_t_scatter, last_t)
        
        h_sequence.append(h.clone())
        
    if max_events > 0:
        return torch.stack(h_sequence, dim=1) # [Batch, Max_Events, Dim, D_State]
    else:
        return torch.empty(B, 0, dim, d_state, device=device)


class AsyncMaskAwareSSM(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, a_init_type: str = "random"):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # SSM Core Parameters (Diagonal Form)
        self.A_init = create_a_matrix(init_type=a_init_type, shape=(dim, d_state))
        self.B_proj = nn.Parameter(torch.randn(dim, d_state) * 0.1)

    def forward(self, events: torch.Tensor, event_mask: torch.Tensor):
        A_log = self.A_init.A_log
        if HAS_TRITON and events.device.type == "cuda":
            return MaskAwareFusedScan.apply(events, event_mask, A_log, self.B_proj)
        else:
            return pytorch_fused_scan(events, event_mask, A_log, self.B_proj, self.dim, self.d_state)
