import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from src.models.vessel.active_inference_agent import ActiveInferenceAgent

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize the model
    size = 128
    # We set a low sigma to allow clear structure formation and strong chi for obvious chemotaxis
    model = ActiveInferenceAgent(
        size=size, dt=0.01, D_u=0.16, D_v=0.08, epsilon=0.01, 
        gamma=0.5, sigma=0.0, decay_rate=0.005, chi=1.5, absorption=0.05
    ).to(device)
    model.eval()
    
    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title('Active Inference Agent')
    ax.set_title("Active Inference Agent (Chemotaxis & Metabolism)")
    ax.axis("off")
    
    # The nutrient field is static, so we can overlay it using a green colormap
    N_state = model.N.detach().cpu().squeeze().numpy()
    
    # Create a custom colormap for the nutrient field (transparent to solid green)
    colors = [(0, 1, 0, 0), (0, 1, 0, 0.4)] # Green with alpha
    cmap_nutrient = LinearSegmentedColormap.from_list("nutrient", colors)
    
    # Extract initial state for plotting
    u_state = model.u.detach().cpu().squeeze().numpy()
    
    # Plot the agent
    im_u = ax.imshow(u_state, cmap='magma', vmin=-1.5, vmax=1.5)
    
    # Plot the nutrient field on top
    im_n = ax.imshow(N_state, cmap=cmap_nutrient, vmin=0, vmax=3.0)
    
    # Number of micro-steps per visual frame
    micro_steps = 100
    
    def update(frame):
        with torch.no_grad():
            for _ in range(micro_steps):
                model()
        
        # Update the image
        u_state_np = model.u.detach().cpu().squeeze().numpy()
        im_u.set_array(u_state_np)
        return [im_u, im_n]

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
