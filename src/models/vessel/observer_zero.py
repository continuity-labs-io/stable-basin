import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from einops import rearrange

class ObserverZero(nn.Module):
    """
    Observer Zero: A 2D continuous-time physics simulation using a 
    Reaction-Diffusion system (FitzHugh-Nagumo) to model self-sustaining 
    vortices and Markov Blankets.
    """
    def __init__(
        self,
        size: int = 128,
        dt: float = 0.05,
        D_u: float = 0.16,
        D_v: float = 0.08,
        epsilon: float = 0.01,
        gamma: float = 0.5,
        sigma: float = 0.05
    ):
        super().__init__()
        self.size = size
        self.dt = dt
        self.D_u = D_u
        self.D_v = D_v
        self.epsilon = epsilon
        self.gamma = gamma
        self.sigma = sigma

        # Initialize internal state variables with high frequency random uniform noise
        # Shape: (batch=1, channels=1, height=size, width=size)
        self.register_buffer("u", torch.rand(1, 1, size, size) * 2 - 1)
        self.register_buffer("v", torch.rand(1, 1, size, size) * 2 - 1)

        # 3x3 Laplacian kernel for spatial coupling (gap junctions)
        # Sums to 0 to conserve mass
        kernel = torch.tensor([
            [0.05, 0.20, 0.05],
            [0.20, -1.00, 0.20],
            [0.05, 0.20, 0.05]
        ], dtype=torch.float32)
        # Reshape for conv2d: (out_channels, in_channels, H, W)
        kernel = rearrange(kernel, "h w -> 1 1 h w")
        self.register_buffer("laplacian_kernel", kernel)

    def _laplacian(self, state: Float[torch.Tensor, "batch channels height width"]) -> Float[torch.Tensor, "batch channels height width"]:
        """
        Computes the spatial Laplacian ∇² using a 2D convolution.
        Uses circular padding to create a toroidal (wrap-around) space.
        """
        # F.pad format for 4D tensor is (pad_left, pad_right, pad_top, pad_bottom)
        padded = F.pad(state, (1, 1, 1, 1), mode='circular')
        return F.conv2d(padded, self.laplacian_kernel)

    def forward(self) -> tuple[Float[torch.Tensor, "batch channels height width"], Float[torch.Tensor, "batch channels height width"]]:
        """
        Executes one continuous-time update step.
        """
        lap_u = self._laplacian(self.u)
        lap_v = self._laplacian(self.v)

        # Brownian noise (thermodynamic entropy)
        dW = torch.randn_like(self.u)

        # Reaction-Diffusion Physics (FitzHugh-Nagumo)
        # u(t+Δt) = u(t) + Δt · [D_u ∇² u + u - u³ - v + σdW]
        # v(t+Δt) = v(t) + Δt · [D_v ∇² v + ε(u - γv)]
        
        du = self.D_u * lap_u + self.u - (self.u ** 3) - self.v + self.sigma * dW
        dv = self.D_v * lap_v + self.epsilon * (self.u - self.gamma * self.v)

        self.u = self.u + self.dt * du
        self.v = self.v + self.dt * dv

        return self.u, self.v

    def inject_wound(self, x: int, y: int, radius: int = 5, u_val: float = 3.0, v_val: float = -1.0):
        """
        Injects a massive localized spike of energy into the substrate at (x, y)
        to simulate an external perturbation (sensation).
        """
        # Create a coordinate grid
        Y, X = torch.meshgrid(
            torch.arange(self.size, device=self.u.device), 
            torch.arange(self.size, device=self.u.device), 
            indexing='ij'
        )
        
        # Calculate distance squared from the center
        dist_sq = (X - x) ** 2 + (Y - y) ** 2
        
        # Create mask for the circular region
        mask = dist_sq <= radius ** 2
        
        # Reshape mask to match state tensor shape (1, 1, H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Inject the wound using torch.where for gradient stability
        u_tensor = torch.tensor(u_val, device=self.u.device, dtype=self.u.dtype)
        v_tensor = torch.tensor(v_val, device=self.v.device, dtype=self.v.dtype)
        
        self.u = torch.where(mask, u_tensor, self.u)
        self.v = torch.where(mask, v_tensor, self.v)
