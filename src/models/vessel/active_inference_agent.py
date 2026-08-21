import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from einops import rearrange

class ActiveInferenceAgent(nn.Module):
    """
    Active Inference Agent: Upgrades the Observer Zero Reaction-Diffusion system
    with Fristonian Free Energy minimization and chemotaxis.
    """
    def __init__(
        self,
        size: int = 128,
        dt: float = 0.01,
        D_u: float = 0.16,
        D_v: float = 0.08,
        epsilon: float = 0.01,
        gamma: float = 0.5,
        sigma: float = 0.001,
        decay_rate: float = 0.005,
        chi: float = 0.5,        # Chemotactic sensitivity
        absorption: float = 0.01 # Rate of nutrient absorption
    ):
        super().__init__()
        self.size = size
        self.dt = dt
        self.D_u = D_u
        self.D_v = D_v
        self.epsilon = epsilon
        self.gamma = gamma
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.chi = chi
        self.absorption = absorption

        # Initialize internal state variables with zero, except for a localized "agent" patch
        self.register_buffer("u", torch.zeros(1, 1, size, size))
        self.register_buffer("v", torch.zeros(1, 1, size, size))
        
        # Localize the initial agent (primordial soup) to the top-left quadrant
        s_min, s_max = int(size * 0.15), int(size * 0.35)
        self.u[..., s_min:s_max, s_min:s_max] = torch.rand(1, 1, s_max-s_min, s_max-s_min) * 2 - 1
        self.v[..., s_min:s_max, s_min:s_max] = torch.rand(1, 1, s_max-s_min, s_max-s_min) * 2 - 1
        
        # Nutrient field (static Gaussian)
        N = self._create_gaussian_nutrient(size)
        self.register_buffer("N", N)

        # 3x3 Laplacian kernel
        kernel_lap = torch.tensor([
            [0.05, 0.20, 0.05],
            [0.20, -1.00, 0.20],
            [0.05, 0.20, 0.05]
        ], dtype=torch.float32)
        self.register_buffer("lap_kernel", rearrange(kernel_lap, "h w -> 1 1 h w"))
        
        # Sobel kernels for spatial gradients
        # Scaled down by 1/8 to represent true derivative on grid spacing
        sobel_x = torch.tensor([
            [-0.125, 0.0, 0.125],
            [-0.250, 0.0, 0.250],
            [-0.125, 0.0, 0.125]
        ], dtype=torch.float32)
        self.register_buffer("grad_x_kernel", rearrange(sobel_x, "h w -> 1 1 h w"))
        
        sobel_y = torch.tensor([
            [-0.125, -0.250, -0.125],
            [ 0.000,  0.000,  0.000],
            [ 0.125,  0.250,  0.125]
        ], dtype=torch.float32)
        self.register_buffer("grad_y_kernel", rearrange(sobel_y, "h w -> 1 1 h w"))

    def _create_gaussian_nutrient(self, size: int) -> Float[torch.Tensor, "1 1 height width"]:
        # Place the food source off-center (at 75% width, 75% height)
        Y, X = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        center_x, center_y = int(size * 0.75), int(size * 0.75)
        # Gaussian distribution
        sigma = size * 0.15
        N = torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        return N.unsqueeze(0).unsqueeze(0).float() * 3.0  # Scale intensity for a strong gradient

    def _apply_conv(self, state: Float[torch.Tensor, "batch channels height width"], kernel: torch.Tensor) -> Float[torch.Tensor, "batch channels height width"]:
        """Applies a 3x3 convolution with circular padding."""
        padded = F.pad(state, (1, 1, 1, 1), mode='circular')
        return F.conv2d(padded, kernel)

    def compute_gradient(self, state: Float[torch.Tensor, "batch channels height width"]) -> tuple[Float[torch.Tensor, "batch channels height width"], Float[torch.Tensor, "batch channels height width"]]:
        """Calculates spatial gradient ∇ = (∂x, ∂y)."""
        grad_x = self._apply_conv(state, self.grad_x_kernel)
        grad_y = self._apply_conv(state, self.grad_y_kernel)
        return grad_x, grad_y

    def compute_divergence(self, fx: Float[torch.Tensor, "batch channels height width"], fy: Float[torch.Tensor, "batch channels height width"]) -> Float[torch.Tensor, "batch channels height width"]:
        """Calculates divergence ∇·(fx, fy) = ∂x(fx) + ∂y(fy)."""
        div_x = self._apply_conv(fx, self.grad_x_kernel)
        div_y = self._apply_conv(fy, self.grad_y_kernel)
        return div_x + div_y

    def forward(self) -> tuple[Float[torch.Tensor, "batch channels height width"], Float[torch.Tensor, "batch channels height width"]]:
        """Executes one continuous-time update step."""
        # 1. Passive Diffusion
        lap_u = self._apply_conv(self.u, self.lap_kernel)
        lap_v = self._apply_conv(self.v, self.lap_kernel)

        # 2. Sensation: Evaluate the external nutrient gradient
        N_grad_x, N_grad_y = self.compute_gradient(self.N)
        
        # 3. Active Inference: Chemotaxis Advection (Keller-Segel model)
        # The agent dynamically couples its spread to move UP the nutrient gradient
        # Velocity field V = χ ∇N
        # Advection = -∇·(u V) = -χ ∇·(u ∇N)
        flux_x = self.u * N_grad_x
        flux_y = self.u * N_grad_y
        advection_u = self.compute_divergence(flux_x, flux_y)
        
        # Thermodynamic noise
        dW = torch.randn_like(self.u)
        
        # Energy Replenishment
        # Only active u pixels physically overlapping with the field can absorb nutrients
        replenishment = self.absorption * self.u * self.N

        # Reaction-Diffusion Physics + Chemotaxis + Metabolism
        du = (
            self.D_u * lap_u 
            - self.chi * advection_u 
            + self.u - (self.u ** 3) 
            - self.v 
            - self.decay_rate * self.u 
            + replenishment 
            + self.sigma * dW
        )
        
        dv = self.D_v * lap_v + self.epsilon * (self.u - self.gamma * self.v)

        self.u = self.u + self.dt * du
        self.v = self.v + self.dt * dv

        return self.u, self.v
