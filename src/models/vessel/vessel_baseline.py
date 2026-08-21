import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jaxtyping import Float
from einops import rearrange

class VesselBaseline(nn.Module):
    """
    A Controlled Experimental Vessel establishing a baseline for an Observer.
    Simulates a 2D observer enclosed in a Markov Blanket, in a thermal bath
    using a spatially heterogeneous Reaction-Diffusion (FitzHugh-Nagumo) system.
    """
    def __init__(
        self,
        size: int = 200,
        dt: float = 0.05,
        radius_internal: float = 30.0,
        thickness_wall: float = 5.0,
        D_u_base: float = 0.16,
        D_v_base: float = 0.08,
        epsilon: float = 0.01,
        gamma: float = 0.5,
        sigma_ext: float = 0.5, 
        v_wall_threshold: float = 2.0
    ):
        super().__init__()
        self.size = size
        self.dt = dt
        self.radius_internal = radius_internal
        self.thickness_wall = thickness_wall
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.sigma_ext = sigma_ext
        self.v_wall_threshold = v_wall_threshold
        
        # Spatial Masks setup
        y, x = torch.meshgrid(
            torch.arange(size, dtype=torch.float32), 
            torch.arange(size, dtype=torch.float32),
            indexing='ij'
        )
        center_x, center_y = size / 2.0, size / 2.0
        dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Masks: Shape (1, 1, H, W)
        # Using strict inequalities to ensure mutually exclusive regions
        mask_internal = dist <= radius_internal
        mask_wall = (dist > radius_internal) & (dist <= (radius_internal + thickness_wall))
        mask_external = dist > (radius_internal + thickness_wall)
        
        # Use einops for reshape instead of native .view() as per enforcing shape discipline
        mask_internal = rearrange(mask_internal, "h w -> 1 1 h w")
        mask_wall = rearrange(mask_wall, "h w -> 1 1 h w")
        mask_external = rearrange(mask_external, "h w -> 1 1 h w")

        self.register_buffer("mask_internal", mask_internal)
        self.register_buffer("mask_wall", mask_wall)
        self.register_buffer("mask_external", mask_external)
        
        # Heterogeneous parameters
        D_u = torch.full((1, 1, size, size), D_u_base, dtype=torch.float32)
        D_v = torch.full((1, 1, size, size), D_v_base, dtype=torch.float32)
        
        # Reduce diffusion coefficient significantly in this ring to prevent leakage
        D_u = torch.where(self.mask_wall, torch.tensor(D_u_base * 0.01), D_u)
        D_v = torch.where(self.mask_wall, torch.tensor(D_v_base * 0.01), D_v)
        
        self.register_buffer("D_u", D_u)
        self.register_buffer("D_v", D_v)
        
        # Initial states
        u_init = torch.zeros(1, 1, size, size, dtype=torch.float32)
        # Random noise in external state, calm in internal
        u_init = torch.where(self.mask_external, torch.rand_like(u_init) * 2 - 1, u_init)
        
        v_init = torch.zeros(1, 1, size, size, dtype=torch.float32)
        v_init = torch.where(self.mask_external, torch.rand_like(v_init) * 2 - 1, v_init)
        # Wall is rigidly set to high threshold
        v_init = torch.where(self.mask_wall, torch.tensor(self.v_wall_threshold), v_init)
        
        self.register_buffer("u", u_init)
        self.register_buffer("v", v_init)
        
        # 3x3 Laplacian kernel
        kernel = torch.tensor([
            [0.05, 0.20, 0.05],
            [0.20, -1.00, 0.20],
            [0.05, 0.20, 0.05]
        ], dtype=torch.float32)
        kernel = rearrange(kernel, "h w -> 1 1 h w")
        self.register_buffer("laplacian_kernel", kernel)

    def _laplacian(self, state: Float[torch.Tensor, "batch channels height width"]) -> Float[torch.Tensor, "batch channels height width"]:
        padded = F.pad(state, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, self.laplacian_kernel)

    def forward(self) -> tuple[Float[torch.Tensor, "batch channels height width"], Float[torch.Tensor, "batch channels height width"], Float[torch.Tensor, ""]]:
        """
        Executes one continuous-time update step of the FitzHugh-Nagumo RD system.
        """
        lap_u = self._laplacian(self.u)
        lap_v = self._laplacian(self.v)
        
        # Spatially heterogeneous Brownian noise σdW
        dW = torch.randn_like(self.u)
        noise = torch.zeros_like(self.u)
        # Inject high Brownian noise (σdW) in the Heat Bath
        noise = torch.where(self.mask_external, self.sigma_ext * dW, noise)
        
        # Reaction-Diffusion Physics (FitzHugh-Nagumo)
        # u(t+Δt) = u(t) + Δt · [D_u ∇² u + u - u³ - v + σdW]
        # v(t+Δt) = v(t) + Δt · [D_v ∇² v + ε(u - γv)]
        
        du = self.D_u * lap_u + self.u - (self.u ** 3) - self.v + noise
        dv = self.D_v * lap_v + self.epsilon * (self.u - self.gamma * self.v)
        
        self.u = self.u + self.dt * du
        self.v = self.v + self.dt * dv
        
        # Enforce wall conditions rigidly (v is permanently high in the wall to dampen noise)
        self.v = torch.where(self.mask_wall, torch.tensor(self.v_wall_threshold, device=self.v.device, dtype=self.v.dtype), self.v)
        
        # Calculate Metric (Internal Variance/Entropy)
        # Variance of the `u` tensor strictly within the Internal State circle
        u_internal = self.u[self.mask_internal]
        # Avoid NaN by ensuring more than 1 element
        if u_internal.numel() > 1:
            internal_variance = torch.var(u_internal)
        else:
            internal_variance = torch.tensor(0.0, device=self.u.device)
            
        return self.u, self.v, internal_variance

def run_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesselBaseline(size=200).to(device)
    
    # 2-panel figure: Left is heatmap, Right is line graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title("Observer Baseline (Utopia)")
    
    # Left Panel: 2D Heatmap
    heatmap = ax1.imshow(model.u.cpu().squeeze().numpy(), cmap='magma', vmin=-2.5, vmax=2.5)
    ax1.set_title("Vessel Baseline (Observer Zero)\nu: Activator State", color='white')
    ax1.axis('off')
    
    # Right Panel: Live updating line graph
    variances = []
    times = []
    line, = ax2.plot([], [], lw=2, color='cyan')
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 0.5)
    ax2.set_title("Internal Variance (Entropy) over Time", color='white')
    ax2.set_xlabel("Time Step (t)", color='white')
    ax2.set_ylabel("Variance (Var[u_internal])", color='white')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_facecolor('#1e1e1e')
    ax2.tick_params(colors='white')
    
    fig.patch.set_facecolor('#121212')
    plt.tight_layout()
    
    step = [0]
    
    def update(frame):
        # Multiple steps per frame for smooth animation
        for _ in range(5):
            u, v, internal_var = model()
            step[0] += 1
            
            variances.append(internal_var.item())
            times.append(step[0])
            
        heatmap.set_data(u.cpu().squeeze().numpy())
        
        line.set_data(times, variances)
        if step[0] > ax2.get_xlim()[1]:
            # Auto-scroll x axis
            ax2.set_xlim(0, step[0] * 1.5)
            
        # Dynamically adjust y axis if variance spikes
        max_var = max(variances)
        if max_var > ax2.get_ylim()[1]:
            ax2.set_ylim(0, max_var * 1.5)
            
        return heatmap, line
        
    ani = FuncAnimation(fig, update, frames=500, interval=40, blit=False)
    plt.show()

if __name__ == "__main__":
    run_simulation()
