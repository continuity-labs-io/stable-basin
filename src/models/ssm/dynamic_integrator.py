import torch
import torch.nn as nn

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
        B, max_events, _ = events.shape
        device = events.device
        
        # The true Biological Manifold: [Batch, Dim, D_State]
        h = torch.zeros(B, self.dim, self.d_state, device=device)
        
        # Track the exact timestamp each sensor was last updated: [Batch, Dim]
        last_t = torch.zeros(B, self.dim, device=device)
        
        A = -torch.exp(self.A_log) # Ensure A is negative [Dim, D_State]
        h_sequence = []

        for i in range(max_events):
            vals = events[:, i, 0]              # [Batch]
            sensor_ids = events[:, i, 1].long() # [Batch]
            timestamps = events[:, i, 2]        # [Batch]
            valid = event_mask[:, i]            # [Batch]

            # Prevent Out-Of-Bounds Gather on invalid padded elements
            safe_sensor_ids = torch.clamp(sensor_ids, 0, self.dim - 1)

            # 1. Calculate Dynamic Delta T
            last_t_sensor = last_t.gather(1, safe_sensor_ids.unsqueeze(1)).squeeze(1)
            dt = timestamps - last_t_sensor
            
            # 2. Continuous-Time Discretization (Zero-Order Hold)
            A_active = A[safe_sensor_ids]             # [Batch, d_state]
            B_active = self.B_proj[safe_sensor_ids]   # [Batch, d_state]
            
            A_bar = torch.exp(A_active * dt.unsqueeze(1))
            B_bar = ((A_bar - 1.0) / (A_active - 1e-8)) * B_active * vals.unsqueeze(1)
            
            # 3. Apply State Update (Orthogonal Routing)
            h_active = h.gather(1, safe_sensor_ids.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_state)).squeeze(1)
            h_new = A_bar * h_active + B_bar
            
            # 4. Vectorized Scatter (Safely mapping back into full tensor)
            h_scatter = h.scatter(1, safe_sensor_ids.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_state), h_new.unsqueeze(1))
            h = torch.where(valid.unsqueeze(1).unsqueeze(2), h_scatter, h)
            
            # Scatter new timestamp
            last_t_scatter = last_t.scatter(1, safe_sensor_ids.unsqueeze(1), timestamps.unsqueeze(1))
            last_t = torch.where(valid.unsqueeze(1), last_t_scatter, last_t)
            
            h_sequence.append(h.clone())
            
        if max_events > 0:
            return torch.stack(h_sequence, dim=1) # [Batch, Max_Events, Dim, D_State]
        else:
            return torch.empty(B, 0, self.dim, self.d_state, device=device)
