import torch
import torch.nn as nn

from .physics import create_a_matrix


class BaselineSSM(nn.Module):
    """
    A baseline continuous-time State Space Model (SSM) that processes all
    timesteps sequentially without any explicit masking mechanism (i.e., it
    ingests padded or masked states blindly). 
    
    It serves as a control group against mask-aware variants. The model uses
    Zero-Order Hold (ZOH) discretization with an input-dependent time step
    (`dt`) and input projection (`B`), while the state transition `A` is a
    learned global parameter restricted to negative values for stability.
    """

    def __init__(self, d_model: int, d_state: int = 16, A_scale: float = 0.5, A_shift: float = 0.1, a_init_type: str = "random"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_init = create_a_matrix(init_type=a_init_type, shape=(d_model,), a_scale=A_scale, a_shift=A_shift)
        self.B_proj = nn.Linear(d_model, d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)
        
        import math
        self.dt_proj.bias.data.uniform_(math.log(0.001), math.log(0.1))

    def forward(self, latent_x: torch.Tensor):
        """
        Args:
            latent_x: Tensor of shape [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = latent_x.size()
        h_prev = torch.zeros(batch, d_model, device=latent_x.device)

        A = self.A_init()

        hidden_states = []
        for t in range(seq_len):
            x_t = latent_x[:, t, :]

            dt = torch.nn.functional.softplus(self.dt_proj(x_t))
            B = self.B_proj(x_t)

            A_bar = torch.exp(A * dt)
            B_bar = (A_bar - 1.0) / (A - 1e-8) * B

            h_prev = A_bar * h_prev + B_bar
            hidden_states.append(h_prev)

        return torch.stack(hidden_states, dim=1)
