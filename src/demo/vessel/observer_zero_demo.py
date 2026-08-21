import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add the project root to the path so we can import the model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.vessel.observer_zero import ObserverZero

def main():
    # Set device to CPU/Metal backend
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize the model
    size = 128
    model = ObserverZero(size=size, dt=0.01, D_u=0.16, D_v=0.08, epsilon=0.01, 
                         gamma=0.5, sigma=0.0001).to(device)
    model.eval()
    
    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title('Observer Zero')
    ax.set_title("Observer Zero (Reaction-Diffusion)")
    ax.axis("off")
    
    # Extract initial state for plotting
    u_state = model.u.detach().cpu().squeeze().numpy()
    im = ax.imshow(u_state, cmap='magma', vmin=-1.5, vmax=1.5)
    
    # Number of micro-steps per visual frame to speed up the visual evolution
    micro_steps = 100
    
    def update(frame):
        with torch.no_grad():
            for _ in range(micro_steps):
                model()
        
        # Update the image
        u_state_np = model.u.detach().cpu().squeeze().numpy()
        im.set_array(u_state_np)
        return [im]

    # Create the animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=None, 
        interval=20, 
        blit=True,
        cache_frame_data=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
