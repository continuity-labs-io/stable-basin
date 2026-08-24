"""
The Waddington Prior: Fristonian NESS Simulation

Simulates and visualizes aging as the thermodynamic flattening of a biological Prior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def potential_energy(x: torch.Tensor, mu: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """U(x) = 0.5 * (x - μ)^T @ Π @ (x - μ)"""
    diff = x - mu
    return 0.5 * diff @ Pi @ diff

def main():
    # -------------------------------------------------------------------------
    # The Mathematical Setup
    # -------------------------------------------------------------------------
    # 2D state vector configuration
    mu = torch.tensor([0.0, 0.0])
    
    # Solenoidal flow matrix (rotational dynamics)
    Q = torch.tensor([[0.0, 1.0], 
                      [-1.0, 0.0]])
                      
    # Dissipation matrix (friction / entropy generation)
    Gamma = torch.tensor([[0.1, 0.0], 
                          [0.0, 0.1]])
    
    # The Two Conditions (Youth vs. Aged)
    # Steep geometric basin
    Pi_youth = torch.tensor([[5.0, 0.0], 
                             [0.0, 5.0]])
                             
    # Flattened, shallow basin
    Pi_aged = torch.tensor([[0.5, 0.0], 
                             [0.0, 0.5]])
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    # Initialize states near origin
    x_youth = torch.tensor([1.0, 1.0])
    x_aged = torch.tensor([1.0, 1.0])
    
    # dt=0.03 ensures mathematical stability for the high-precision youth matrix
    dt = 0.03
    
    # -------------------------------------------------------------------------
    # Geometry Preparation (Contour Backgrounds)
    # -------------------------------------------------------------------------
    grid_points = 100
    X_grid = np.linspace(-3.0, 3.0, grid_points)
    Y_grid = np.linspace(-3.0, 3.0, grid_points)
    X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
    
    Z_youth = np.zeros_like(X_mesh)
    Z_aged = np.zeros_like(X_mesh)
    
    for i in range(grid_points):
        for j in range(grid_points):
            pt = torch.tensor([X_mesh[i, j], Y_mesh[i, j]], dtype=torch.float32)
            Z_youth[i, j] = potential_energy(pt, mu, Pi_youth).item()
            Z_aged[i, j] = potential_energy(pt, mu, Pi_aged).item()
            
    # -------------------------------------------------------------------------
    # Setup Figure and Subplots
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('The Waddington Prior: Youth vs Aged')
    
    # Plot Potential Energy Basins
    levels = np.linspace(0, 10, 20)
    ax1.contour(X_mesh, Y_mesh, Z_youth, levels=levels, cmap='viridis', alpha=0.5)
    ax2.contour(X_mesh, Y_mesh, Z_aged, levels=levels, cmap='viridis', alpha=0.5)
    
    for ax, title in zip([ax1, ax2], ["Youth (High Precision Π)", "Aged (Low Precision Π)"]):
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # -------------------------------------------------------------------------
    # Particle and Trail Artists
    # -------------------------------------------------------------------------
    trail_length = 50
    youth_trail = np.zeros((trail_length, 2))
    aged_trail = np.zeros((trail_length, 2))
    
    youth_trail[:] = x_youth.numpy()
    aged_trail[:] = x_aged.numpy()
    
    # Particles
    youth_scatter = ax1.scatter([], [], c='red', s=50, zorder=3, edgecolors='black')
    youth_line, = ax1.plot([], [], c='red', alpha=0.4, zorder=2, linewidth=2)
    
    aged_scatter = ax2.scatter([], [], c='blue', s=50, zorder=3, edgecolors='black')
    aged_line, = ax2.plot([], [], c='blue', alpha=0.4, zorder=2, linewidth=2)
    
    # Telemetry Overlays
    text_youth = ax1.text(-2.8, 2.8, "", fontsize=10, family='monospace', 
                          verticalalignment='top',
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    text_aged = ax2.text(-2.8, 2.8, "", fontsize=10, family='monospace', 
                          verticalalignment='top',
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # -------------------------------------------------------------------------
    # The Continuous-Time Update Loop
    # -------------------------------------------------------------------------
    def update(frame):
        nonlocal x_youth, x_aged, youth_trail, aged_trail
        
        # Inject the exact same random noise seed into both steps
        noise = torch.randn(2)
        diffusion = torch.sqrt(2 * Gamma * dt) @ noise
        
        # Euler-Maruyama integration step:
        # dx = dt * ((Q - Gamma) @ Π @ (x - μ)) + sqrt(2 * Gamma * dt) @ dW
        drift_youth = dt * ((Q - Gamma) @ Pi_youth @ (x_youth - mu))
        drift_aged = dt * ((Q - Gamma) @ Pi_aged @ (x_aged - mu))
        
        x_youth = x_youth + drift_youth + diffusion
        x_aged = x_aged + drift_aged + diffusion
        
        # Update Fading Trails
        youth_trail = np.roll(youth_trail, -1, axis=0)
        youth_trail[-1] = x_youth.numpy()
        
        aged_trail = np.roll(aged_trail, -1, axis=0)
        aged_trail[-1] = x_aged.numpy()
        
        # Update Visual Artists
        youth_scatter.set_offsets(x_youth.numpy())
        youth_line.set_data(youth_trail[:, 0], youth_trail[:, 1])
        
        aged_scatter.set_offsets(x_aged.numpy())
        aged_line.set_data(aged_trail[:, 0], aged_trail[:, 1])
        
        # Update Telemetry Display
        u_youth = potential_energy(x_youth, mu, Pi_youth).item()
        dist_youth = torch.norm(x_youth - mu).item()
        
        u_aged = potential_energy(x_aged, mu, Pi_aged).item()
        dist_aged = torch.norm(x_aged - mu).item()
        
        text_youth.set_text(
            f"Trace(Π): {torch.trace(Pi_youth).item():.1f}\n"
            f"U(x)    : {u_youth:.2f}\n"
            f"||x - μ|| : {dist_youth:.2f}"
        )
        
        text_aged.set_text(
            f"Trace(Π): {torch.trace(Pi_aged).item():.1f}\n"
            f"U(x)    : {u_aged:.2f}\n"
            f"||x - μ|| : {dist_aged:.2f}"
        )
        
        return youth_scatter, youth_line, text_youth, aged_scatter, aged_line, text_aged

    ani = animation.FuncAnimation(fig, update, frames=200, interval=26, blit=True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
