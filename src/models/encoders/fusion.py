import torch
import torch.nn as nn


class BiologicalCartridgeFusion(nn.Module):
    def __init__(self, d_cartridge: int, n_modalities: int, d_model: int):
        super().__init__()
        self.W_cart = nn.Linear(d_cartridge, d_model, bias=False)
        self.W_gate = nn.Linear(n_modalities, d_model, bias=True)

        # TRUE ORTHOGONAL SUBSPACE ROUTING PRIOR
        half = d_model // 2
        with torch.no_grad():
            # 1. Absolute Stasis Default: sigmoid(-10) = 4.5e-5. 
            # Guarantees memory survives 1000+ step voids.
            self.W_gate.bias.fill_(-10.0)
            self.W_gate.weight.fill_(0.0)
            
            # 2. Orthogonal Routing
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
