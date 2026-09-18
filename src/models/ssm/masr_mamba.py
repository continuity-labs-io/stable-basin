"""
[ICEBOXED] - Architectural Pivot

These modules represent an attempt to force classical, deterministic architectures to handle continuous-time biological realities (e.g., Latent Stasis, Triton kernel optimizations, and deterministic LRP). Moving forward, Stable Basin relies on natively probabilistic, energy-based thermodynamic frameworks where missing data is naturally imputed and physics-based hardware minimization renders these hacks obsolete.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor

from src.models.ssm.physics import create_a_matrix

def pscan(A, X):
    """
    Parallel associative scan (Kogge-Stone).
    A: (batch, seq, d_model, d_state)
    X: (batch, seq, d_model, d_state)
    Computes H_t = A_t * H_{t-1} + X_t
    """
    B_sz, L, d_model, d_state = X.shape
    L_pad = 2 ** math.ceil(math.log2(L))
    pad_len = L_pad - L
    if pad_len > 0:
        A = F.pad(A, (0, 0, 0, 0, 0, pad_len), value=1.0)
        X = F.pad(X, (0, 0, 0, 0, 0, pad_len), value=0.0)
        
    for i in range(int(math.log2(L_pad))):
        shift = 2 ** i
        
        X_shifted = X[:, :-shift]
        A_shifted = A[:, :-shift]
        
        X_update = X[:, shift:] + A[:, shift:] * X_shifted
        A_update = A[:, shift:] * A_shifted
        
        X_new = X.clone()
        A_new = A.clone()
        
        X_new[:, shift:] = X_update
        A_new[:, shift:] = A_update
        
        X = X_new
        A = A_new
        
    return X[:, :L]

@jaxtyped(typechecker=beartype)
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
    """
    acc_dtype = torch.promote_types(x.dtype, torch.float32)
    
    # Latent Stasis: multiply continuous Δt by the boolean mask
    dt_masked = dt * mask # (batch, seq, d_model)
    dt_masked_exp = dt_masked.unsqueeze(-1) # (batch, seq, d_model, 1)
    
    A_safe = torch.where(A.abs() <= 1e-8, torch.full_like(A, -1e-8), A)
    
    A_bar = torch.exp(dt_masked_exp * A) # (batch, seq, d_model, d_state)
    
    # B is (batch, seq, d_state), unsqueeze to (batch, seq, 1, d_state)
    B_bar = torch.expm1(dt_masked_exp * A) / A_safe * B.unsqueeze(2) # (batch, seq, d_model, d_state)
    
    X_in = B_bar.to(acc_dtype) * x.unsqueeze(-1).to(acc_dtype)
    A_bar = A_bar.to(acc_dtype)
    
    # Parallel associative scan replaces the O(N) loop
    H = pscan(A_bar, X_in) # (batch, seq, d_model, d_state)
    
    # C is (batch, seq, d_state), unsqueeze to (batch, seq, 1, d_state)
    y = (C.unsqueeze(2).to(acc_dtype) * H).sum(dim=-1) + D.to(acc_dtype) * x.to(acc_dtype)
    
    return y.to(x.dtype)

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
            if mask.shape[-1] != h.shape[-1]:
                mamba_mask = (mask.sum(dim=-1, keepdim=True) > 0).float().expand_as(h)
            else:
                mamba_mask = mask
            hidden_states = self.mamba(h, mamba_mask)
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


