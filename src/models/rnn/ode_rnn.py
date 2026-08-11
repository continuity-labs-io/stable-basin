import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, t, h):
        return self.net(h)

class ODERNNModel(nn.Module):
    """
    ODE-RNN baseline that continuously integrates the hidden state between observations
    using a Neural ODE, then applies a standard GRU update at observation times.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # ODE Solver Configuration
        self.ode_solver = 'rk4'
        self.ode_step_size = 0.1
        
        self.ode_func = ODEFunc(d_model)
        self.gru_cell = nn.GRUCell(d_model, d_model)
        
    def forward(self, x: torch.Tensor, delta_t: torch.Tensor):
        """
        Args:
            x: (B, L, d_model) - the latent features
            delta_t: (B, L, 1) - ignored in this uniform step implementation, 
                                 as integration step is always 1 unit.
        """
        B, L, _ = x.shape
        
        h = torch.zeros(B, self.d_model, device=x.device)
        h_seq = []
        
        for t in range(L):
            # Integrate hidden state from t-1 to t (interval of 1)
            t_span = torch.tensor([0.0, 1.0], device=x.device)
            
            # odeint returns shape (2, B, d_model)
            h_trajectory = odeint(
                self.ode_func, 
                h, 
                t_span, 
                method=self.ode_solver, 
                options={'step_size': self.ode_step_size}
            )
            h = h_trajectory[1]
            
            # Apply GRU update with the current observation
            h = self.gru_cell(x[:, t, :], h)
            h_seq.append(h.unsqueeze(1))
            
        return torch.cat(h_seq, dim=1)
