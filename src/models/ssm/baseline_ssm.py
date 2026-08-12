import torch
import torch.nn as nn


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

    def __init__(self, d_model: int, A_scale: float = 0.5, A_shift: float = 0.1):
        super().__init__()
        # Initialize A uniformly in [- (A_scale + A_shift), -A_shift] (since A = -exp(A_log)). The
        # shift bounds A away from 0 to prevent infinite memory, and the
        # scale ensures memory doesn't decay instantly, providing diverse
        # timescales.
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model) * A_scale + A_shift))
        self.B_proj = nn.Linear(d_model, d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)

    def forward(self, latent_x: torch.Tensor):
        """
        Args:
            latent_x: Tensor of shape [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = latent_x.size()
        h_prev = torch.zeros(batch, d_model, device=latent_x.device)

        A = -torch.exp(self.A_log)

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
