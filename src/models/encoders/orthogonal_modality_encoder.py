import torch
import torch.nn as nn


class OrthogonalModalityEncoder(nn.Module):
    """
    Gated Modality Fusion Layer (Mask-to-Gate Projector).

    This module maps a low-dimensional boolean mask indicating sensor presence into
    a continuous gating tensor used by downstream Mask-Aware SSMs. It performs two specific roles:
    1. Feature Projection: Linearly projects the raw input features into a latent dimension (`d_model`).
    2. Orthogonal Subspace Routing: Uses a hardcoded, non-trainable prior to map the mask to the latent space.
       It explicitly partitions the latent space such that Modality 0 controls the first half, and
       Modality 1 controls the second half. Missing sensors result in a nearly-zero gate, which freezes time
       for that specific half of the latent space.
    """
    def __init__(self, d_sensor_total: int, n_modalities: int, d_model: int):
        super().__init__()
        self.W_cart = nn.Linear(d_sensor_total, d_model, bias=False)
        self.W_gate = nn.Linear(n_modalities, d_model, bias=True)

        # true orthogonal subspace routing prior
        half = d_model // 2
        with torch.no_grad():
            # Absolute Stasis Default: We initialize the bias to -10.0 so that when passed 
            # through the sigmoid in the forward pass, it evaluates to ~4.5e-5.
            # This near-zero gate guarantees memory survives 1000+ step voids when a sensor is missing.
            self.W_gate.bias.fill_(-10.0)
            self.W_gate.weight.fill_(0.0)
            
            # Orthogonal Routing
            # Modality 0 (Voltage) strictly controls latent dimensions 0 to 31
            self.W_gate.weight[:half, 0] = 20.0 
            
            # Modality 1 (Epigenetics) strictly controls latent dimensions 32 to 63
            self.W_gate.weight[half:, 1] = 20.0 

        self.W_gate.weight.requires_grad = False
        self.W_gate.bias.requires_grad = False

    def forward(self, x_raw: torch.Tensor, mask: torch.Tensor):
        latent_x = self.W_cart(x_raw)
        latent_gate = torch.sigmoid(self.W_gate(mask))
        return latent_x, latent_gate
