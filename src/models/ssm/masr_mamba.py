import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import create_a_matrix

def mamba_masr_reference_scan(x, dt, mask, A, B, C, D):
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
        
        # When mask is 0, dt_masked_t is 0, so Ā = exp(0) = 1 (Identity), and B̄ = 0
        A_bar = torch.exp(dt_masked_t_exp * A) # (batch_size, d_model, d_state)
        B_bar = (dt_masked_t_exp * B_t.unsqueeze(1)) # (batch_size, d_model, d_state)
        
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
        self.A_init = create_a_matrix(init_type=a_init_type, shape=(d_model, d_state), a_scale=0.1, a_shift=0.0)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Data-dependent parameter projections
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask):
        """
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len, d_model)
        """
        # Compute continuous data-dependent parameters
        B = self.B_proj(x) # (batch_size, seq_len, d_state)
        C = self.C_proj(x) # (batch_size, seq_len, d_state)
        
        # Softplus ensures Δt is strictly positive
        dt = F.softplus(self.dt_proj(x)) # (batch_size, seq_len, d_model)
        
        A = self.A_init()
        y = mamba_masr_reference_scan(x, dt, mask, A, B, C, self.D)
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
            self.input_proj = nn.Linear(input_dim, d_model)
            
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
            # Create a global mask for the entire layer: 
            # if any sensor is present, it's not totally missing.
            global_mask = mask.amax(dim=-1, keepdim=True).expand(-1, -1, self.mamba.d_model)
            hidden_states = self.mamba(h, global_mask)
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Execution block to generate synthetic dataset and visualize Latent Stasis
    seq_len = 200
    d_model = 2
    d_state = 1
    
    # Synthetic dataset mimicking asynchronous multi-rate biology
    # Modality 0: Fast signal
    # Modality 1: Slow signal with dropouts
    t = torch.linspace(0, 20, seq_len)
    fast_signal = torch.sin(t * 5)
    slow_signal = torch.sin(t * 1) + 1.5 # Offset so exponential decay is visually obvious
    
    x_true = torch.zeros(1, seq_len, d_model)
    x_true[0, :, 0] = fast_signal
    x_true[0, :, 1] = slow_signal
    
    mask = torch.ones(1, seq_len, d_model)
    # Introduce blocks of missing data for the slow signal (simulate asynchronous multi-rate dropouts)
    drop_indices = ((t > 4) & (t < 9)) | ((t > 13) & (t < 17))
    mask[0, drop_indices, 1] = 0.0
    
    # Zero-padded input: replace missing values with 0
    x_padded = x_true * mask
    
    model = PyTorchMambaMASR(d_model=d_model, d_state=d_state)
    
    # Manually initialize weights to make the untrained model act as a basic exponential filter
    # This clearly demonstrates the Latent Stasis (freeze) vs Baseline (decay) dynamics
    with torch.no_grad():
        nn.init.constant_(model.A, -0.5)      # Continuous decay rate (Ā < 1)
        nn.init.constant_(model.D, 0.0)       # No skip connection, purely state-driven output
        nn.init.constant_(model.B_proj.weight, 0.0)
        nn.init.constant_(model.B_proj.bias, 0.5)  # B = 0.5
        nn.init.constant_(model.C_proj.weight, 0.0)
        nn.init.constant_(model.C_proj.bias, 1.0)  # C = 1.0
        nn.init.constant_(model.dt_proj.weight, 0.0)
        nn.init.constant_(model.dt_proj.bias, 0.5) # dt offset
    
    # Run MASR (with sparsity mask)
    with torch.no_grad():
        y_masr = model(x_padded, mask)
        
        # Run Baseline (Zero-padded, standard Mamba without Mask-Aware stasis)
        # Standard Mamba assumes all steps are observed (mask = 1)
        mask_ones = torch.ones_like(mask)
        y_baseline = model(x_padded, mask_ones)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(t.numpy(), x_true[0, :, 1].numpy(), label="True Slow Signal", color='black', alpha=0.3, linewidth=2, linestyle='--')
    axes[0].plot(t.numpy(), x_padded[0, :, 1].numpy(), label="Zero-Padded Input (Observed)", color='red')
    axes[0].set_title("Input Data (Slow Signal with Blocks of Missing Data)")
    axes[0].legend()
    
    axes[1].plot(t.numpy(), y_baseline[0, :, 1].numpy(), label="Baseline (Zero-Padded) Output", color='orange', linewidth=2)
    axes[1].set_title("Standard Mamba (Zero-padded: Hidden State Decays toward 0 during dropouts)")
    axes[1].legend()
    
    axes[2].plot(t.numpy(), y_masr[0, :, 1].numpy(), label="MASR Output", color='green', linewidth=2)
    axes[2].set_title("MASR (Latent Stasis: Hidden State perfectly Freezes during dropouts)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("masr_vs_baseline.png")
    print("Saved graph to masr_vs_baseline.png")
