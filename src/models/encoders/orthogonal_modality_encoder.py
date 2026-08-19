import torch
import torch.nn as nn


class OrthogonalModalityEncoder(nn.Module):
    """
    Gated Modality Fusion Layer (Mask-to-Gate Projector).
    Implements Proportional Orthogonal Routing.
    """
    def __init__(self, d_in: int, modality_dims: list[int], d_model: int):
        super().__init__()
        self.W_proj = nn.Linear(d_in, d_model, bias=False)
        
        n_modalities = len(modality_dims)
        if d_model < n_modalities:
            raise ValueError("d_model must be >= the number of modalities.")
            
        self.W_gate = nn.Linear(n_modalities, d_model, bias=True)
        
        d_sensor_total = sum(modality_dims)

        with torch.no_grad():
            # Absolute Stasis Default: sigmoid(-10) = ~4.5e-5.
            # Guarantees memory survives 1000+ step voids when a sensor is missing.
            self.W_gate.bias.fill_(-10.0)
            self.W_gate.weight.fill_(0.0)
            
            # Proportional Orthogonal Routing
            current_idx = 0
            remaining_d_model = d_model
            
            for i, dim in enumerate(modality_dims):
                remaining_modalities = n_modalities - i
                
                if i == n_modalities - 1:
                    # Last modality gets all remaining dimensions to avoid rounding gaps
                    chunk_size = remaining_d_model
                else:
                    proportion = dim / d_sensor_total
                    chunk_size = int(d_model * proportion)
                    
                    # Guarantee at least 1 dimension per modality
                    # but leave enough for the remaining modalities
                    max_allowed = remaining_d_model - (remaining_modalities - 1)
                    chunk_size = max(1, min(chunk_size, max_allowed))
                
                start_idx = current_idx
                end_idx = current_idx + chunk_size
                
                # Map this modality's mask bit to its proportional latent chunk
                self.W_gate.weight[start_idx:end_idx, i] = 20.0
                
                current_idx = end_idx
                remaining_d_model -= chunk_size

        self.W_gate.weight.requires_grad = False
        self.W_gate.bias.requires_grad = False

    def forward(self, x_raw: torch.Tensor, mask: torch.Tensor):
        latent_x = self.W_proj(x_raw)
        latent_gate = torch.sigmoid(self.W_gate(mask))
        return latent_x, latent_gate
