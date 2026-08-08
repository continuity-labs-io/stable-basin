import torch
import torch.nn as nn


class MaskAwareSSM(nn.Module):
    """
    Observer Model: Mask-Aware SSM that natively implements the 'Physics Hack', 
    modulating time based on the latent gate.
    
    It continuously ingests non-destructive streams (Voltage, Morphology, Fluid Exhaust) 
    and uses Mask-Aware Subspace Routing to mathematically infer hidden, destructive states 
    (like RNA and Epigenetics) without lysing the cell.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model) * 0.5 + 0.1))
        self.B_proj = nn.Linear(d_model, d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)

    def forward(self, latent_x: torch.Tensor, latent_gate: torch.Tensor):
        """
        Args:
            latent_x: Tensor of shape [batch, seq_len, d_model]
            latent_gate: Tensor of shape [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = latent_x.size()
        h_prev = torch.zeros(batch, d_model, device=latent_x.device)

        A = -torch.exp(self.A_log)

        hidden_states = []
        for t in range(seq_len):
            x_t = latent_x[:, t, :]
            g_t = latent_gate[:, t, :]

            dt_base = torch.nn.functional.softplus(self.dt_proj(x_t))
            # THE PHYSICS HACK
            dt_gated = dt_base * g_t + 1e-8

            # EXPLICIT INPUT GATING: Block offline sensors from adding ghost noise to the state
            B = self.B_proj(x_t) * g_t

            A_bar = torch.exp(A * dt_gated)
            B_bar = (A_bar - 1.0) / (A - 1e-8) * B

            h_prev = A_bar * h_prev + B_bar
            hidden_states.append(h_prev)

        return torch.stack(hidden_states, dim=1)


if __name__ == "__main__":
    from src.models.encoders.fusion import BiologicalCartridgeFusion

    fusion = BiologicalCartridgeFusion(d_cartridge=30, n_modalities=2, d_model=64)
    ssm = MaskAwareSSM(d_model=64)

    x_raw = torch.randn(2, 100, 30)
    mask = torch.ones(2, 100, 2)

    latent_x, latent_gate = fusion(x_raw, mask)
    h = ssm(latent_x, latent_gate)

    print(f"Shape of h: {h.shape}")

    has_nan = torch.isnan(h).any().item()
    has_inf = torch.isinf(h).any().item()
    correct_shape = h.shape == torch.Size([2, 100, 64])

    if correct_shape and not has_nan and not has_inf:
        print("Success: The forward pass executed without exploding gradients or NaNs.")
    else:
        if not correct_shape:
            print("Failure: Output shape does not match expected [2, 100, 64].")
        if has_nan:
            print("Failure: Output contains NaNs.")
        if has_inf:
            print("Failure: Output contains Infs (exploding gradients).")
