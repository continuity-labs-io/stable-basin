import torch
import torch.nn as nn

class GRUDModel(nn.Module):
    """
    GRU-D baseline that decays the hidden state based on the time elapsed
    since the sparse causal driver was last observed.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable decay parameter (gamma)
        self.gamma = nn.Parameter(torch.zeros(d_model))
        
        # Standard GRUCell
        self.gru_cell = nn.GRUCell(d_model, d_model)
        
    def forward(self, x: torch.Tensor, delta_t: torch.Tensor):
        """
        Args:
            x: (B, L, d_model) - the latent features
            delta_t: (B, L, 1) - time elapsed since Modality 1 was last observed
        """
        B, L, _ = x.shape
        
        h = torch.zeros(B, self.d_model, device=x.device)
        h_seq = []
        
        # Bound gamma to be non-negative
        # gamma_decay = exp(-relu(gamma) * delta_t)
        # shape: (B, L, d_model)
        gamma_decay = torch.exp(-torch.relu(self.gamma) * delta_t)
        
        for t in range(L):
            # Decay hidden state based on time elapsed
            h = h * gamma_decay[:, t, :]
            
            # Apply GRU update
            h = self.gru_cell(x[:, t, :], h)
            h_seq.append(h.unsqueeze(1))
            
        return torch.cat(h_seq, dim=1)
