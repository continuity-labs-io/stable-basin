import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .physics import create_a_matrix

def mamba_masr_reference_scan(
    x: Float[Tensor, "batch seq d_model"],
    dt: Float[Tensor, "batch seq d_model"],
    mask: Float[Tensor, "batch seq d_model"],
    A: Float[Tensor, "d_model d_state"],
    B: Float[Tensor, "batch seq d_state"],
    C: Float[Tensor, "batch seq d_state"],
    D: Float[Tensor, "d_model"]
) -> Float[Tensor, "batch seq d_model"]:
    """
    Pure PyTorch reference implementation of the Mask-Aware Subspace Routing (MASR) Mamba scan.
    
    Args:
        x: (batch_size, seq_len, d_model) - Input sequence data
        dt: (batch_size, seq_len, d_model) - Continuous dynamic time deltas (Δt)
        mask: (batch_size, seq_len, d_model) - Boolean sparsity mask (1 = observed, 0 = missing)
        A: (d_model, d_state) - Continuous A matrix
        B: (batch_size, seq_len, d_state) - Continuous B matrix (data-dependent)
        C: (batch_size, seq_len, d_state) - Continuous C matrix (data-dependent)
        D: (d_model,) - D skip connection
        
    Returns:
        y: (batch_size, seq_len, d_model) - Output sequence
    """
    batch_size, seq_len, d_model = x.shape
    _, d_state = A.shape
    
    # Initialize hidden state h_0
    h = torch.zeros(batch_size, d_model, d_state, device=x.device, dtype=x.dtype)
    y = torch.zeros_like(x)
    
    for t in range(seq_len):
        x_t = x[:, t, :] # (batch_size, d_model)
        dt_t = dt[:, t, :] # (batch_size, d_model)
        mask_t = mask[:, t, :] # (batch_size, d_model)
        B_t = B[:, t, :] # (batch_size, d_state)
        C_t = C[:, t, :] # (batch_size, d_state)
        
        # Latent Stasis: multiply continuous Δt by the boolean mask
        # If mask is 0 (missing), Δt goes to 0
        dt_masked_t = dt_t * mask_t # (batch_size, d_model)
        
        # Calculate discrete Ā and B̄
        # Ā = exp(Δt * A)
        # B̄ ≈ Δt * B (simplified Mamba ZOH approximation)
        dt_masked_t_exp = dt_masked_t.unsqueeze(-1) # (batch_size, d_model, 1)
        
        # Prevent division by zero during ZOH discretization
        A_safe = torch.where(A.abs() <= 1e-8, torch.full_like(A, -1e-8), A)
        
        A_bar = torch.exp(dt_masked_t_exp * A) # (batch_size, d_model, d_state)
        
        # Swapped exp(dt*A)-1.0 to PyTorch's mathematically identical but numerically 
        # stable torch.expm1(dt*A) to fix catastrophic cancellation in float32 
        # when dt*A is very small, which was injecting massive noise during ZOH discretization.
        B_bar = torch.expm1(dt_masked_t_exp * A) / A_safe * B_t.unsqueeze(1) # (batch_size, d_model, d_state)
        
        # Update hidden state h_t = Ā * h_{t-1} + B̄ * x_t
        # Perfectly freezing the hidden state (h_t = h_{t-1}) when mask is 0
        h = A_bar * h + B_bar * x_t.unsqueeze(-1) # (batch_size, d_model, d_state)
        
        # Compute output y_t = C_t * h_t + D * x_t
        y_t = (C_t.unsqueeze(1) * h).sum(dim=-1) + D * x_t # (batch_size, d_model)
        y[:, t, :] = y_t
        
    return y

class PyTorchMambaMASR(nn.Module):
    def __init__(self, d_model, d_state=16, a_init_type: str = "random"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Continuous state parameters
        # Increased the A initialization bounds (a_scale=0.5, a_shift=0.1) 
        # to enforce a minimum baseline memory leak, preventing the hidden state 
        # from acting as a pure integrator that drifts over 5,000 extrapolation steps.
        self.A_init = create_a_matrix(init_type=a_init_type, shape=(d_model, d_state), a_scale=0.5, a_shift=0.1)
        
        # Initialized the residual shortcut D to zeros (instead of ones) 
        # to force the network to rely on and learn a robust continuous recurrent 
        # state from epoch 1, rather than relying on the feedforward path.
        self.D = nn.Parameter(torch.zeros(d_model))
        
        # Data-dependent parameter projections
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Clamped the initialization of dt_proj.weight to a near-zero uniform 
        # distribution to prevent the continuous time step from fluctuating wildly 
        # on out-of-distribution sequences during extrapolation.
        nn.init.uniform_(self.dt_proj.weight, -1e-4, 1e-4)
        self.dt_proj.bias.data.uniform_(math.log(0.001), math.log(0.1))
        
    def forward(self, x, mask):
        """
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len, d_model)
        """
        # Sever the backward pass for missing data by masking input immediately
        x_masked = x * mask
        
        # Compute continuous data-dependent parameters
        B = self.B_proj(x_masked) # (batch_size, seq_len, d_state)
        C = self.C_proj(x_masked) # (batch_size, seq_len, d_state)
        
        # Softplus ensures Δt is strictly positive
        dt = F.softplus(self.dt_proj(x_masked)) # (batch_size, seq_len, d_model)
        
        A = self.A_init()
        y = mamba_masr_reference_scan(x_masked, dt, mask, A, B, C, self.D)
        return y

class MaskAwareMamba(nn.Module):
    """
    Mask-Aware Mamba-2 Engine for multimodal telemetry.
    
    Comparable to `MaskAwareSSM`, but uses a PyTorch reference MASR Mamba backbone 
    instead of a baseline ZOH continuous-time formulation. 
    """
    def __init__(self, input_dim: int, d_model: int = 256, d_state: int = 64, mask_aware: bool = False, a_init_type: str = "random"):
        super().__init__()
        self.mask_aware = mask_aware
        
        if mask_aware:
            # double the input dim for concatenated sensor failure mask
            in_features = input_dim * 2
            self.input_proj = nn.Sequential(
                nn.Linear(in_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            in_features = input_dim
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            
        self.mamba = PyTorchMambaMASR(d_model=d_model, d_state=d_state, a_init_type=a_init_type)

        # Standard causal forecasting head (predicts x_{t+1})
        self.forward_head = nn.Linear(d_model, input_dim)
        # Reverse head for thermodynamic loss (reconstructs x_t)
        self.reverse_head = nn.Linear(d_model, input_dim)

    def forward(self, x, mask=None, return_hidden=False):
        if self.mask_aware:
            if mask is None:
                mask = torch.isnan(x).float()
            x_safe = torch.nan_to_num(x, nan=0.0)
            x_in = torch.cat([x_safe, mask], dim=-1)
            h = self.input_proj(x_in)
        else:
            h = self.input_proj(x)

        if self.mask_aware and mask is not None:
            # Pass the per-channel mask directly!
            hidden_states = self.mamba(h, mask)
        else:
            hidden_states = self.mamba(h, torch.ones_like(h))

        # positive values in prediction
        pred_t_plus_1 = F.softplus(self.forward_head(hidden_states))
        reconstructed_t = F.softplus(self.reverse_head(hidden_states))

        if return_hidden:
            return pred_t_plus_1, reconstructed_t, hidden_states
        return pred_t_plus_1, reconstructed_t

    def get_hidden_states(self, x, mask=None):
        _, _, hidden_states = self.forward(x, mask=mask, return_hidden=True)
        return hidden_states


