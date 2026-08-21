import torch
import torch.nn as nn

from .physics import create_a_matrix


class MaskAwareSSM(nn.Module):
    """
    Mask-Aware SSM that implements a time-freezing 'Physics Hack'.

    During inference, the model modulates the time-step based on the latent
    gate. When a sensor is masked (g_t ≈ 0), dt is forced to 0. This causes
    A_bar = exp(0) = 1, perfectly freezing the hidden state in time and
    preventing any memory decay while observations are missing.  
    """

    def __init__(self, d_model: int, A_scale: float = 0.5, A_shift: float = 0.1, a_init_type: str = "random"):
        super().__init__()
        
        # Initialize A using the specified factory
        self.A_init = create_a_matrix(init_type=a_init_type, shape=(d_model,), a_scale=A_scale, a_shift=A_shift)
        self.B_proj = nn.Linear(d_model, d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)

        import math
        self.dt_proj.bias.data.uniform_(math.log(0.001), math.log(0.1))

    def _apply_masking(self, dt_base: torch.Tensor, B_base: torch.Tensor, g_t: torch.Tensor):
        """
        Time freezing: By modulating the time-step with the gate, a masked sensor 
        (g_t ≈ 0) forces dt to 0. This causes A_bar = exp(0) = 1, perfectly freezing 
        the hidden state in time and preventing any memory decay while observations 
        are missing.
        
        Explicit input gating: Block offline sensors from adding noise to the state.
        """
        dt_gated = dt_base * g_t + 1e-8
        B_gated = B_base * g_t
        return dt_gated, B_gated

    def forward(self, latent_x: torch.Tensor, latent_gate: torch.Tensor):
        """
        Args:
            latent_x: Tensor of shape [batch, seq_len, d_model] 
            latent_gate: Tensor of shape [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = latent_x.size()
        h_prev = torch.zeros(batch, d_model, device=latent_x.device)

        A = self.A_init()

        hidden_states = []
        for t in range(seq_len):
            x_t = latent_x[:, t, :]
            g_t = latent_gate[:, t, :]

            dt_base = torch.nn.functional.softplus(self.dt_proj(x_t))
            B_base = self.B_proj(x_t)

            dt_gated, B = self._apply_masking(dt_base, B_base, g_t)

            # Resumed SSM update
            A_bar = torch.exp(A * dt_gated)
            B_bar = (A_bar - 1.0) / (A - 1e-8) * B

            h_prev = A_bar * h_prev + B_bar
            hidden_states.append(h_prev)

        return torch.stack(hidden_states, dim=1)
