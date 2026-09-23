import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import matplotlib.colors as mcolors
import logging

logger = logging.getLogger(__name__)

def generate_worm_shape(eigenworms_t: np.ndarray, num_segments: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs the worm (x, y) body coordinates from 6D eigenworm modes.
    If the true basis isn't available, we use Fourier-like basis modes
    to create a biologically plausible shape.
    """
    s = np.linspace(-1, 1, num_segments)
    
    # Synthetic basis for the 6 eigenworms (commonly sine/cosine waves of increasing frequency)
    basis = np.array([
        np.ones_like(s),           # mode 1: mean curvature
        np.sin(np.pi * s),         # mode 2: undulation
        np.cos(np.pi * s),         # mode 3: undulation phase
        np.sin(2 * np.pi * s),     # mode 4: higher freq
        np.cos(2 * np.pi * s),     # mode 5: higher freq phase
        np.sin(3 * np.pi * s),     # mode 6: highest freq
    ])
    
    # theta is the tangent angle along the body
    theta = np.dot(eigenworms_t[:6], basis)
    
    # Reconstruct (x,y) by integrating the tangent angles
    x = np.cumsum(np.cos(theta))
    y = np.cumsum(np.sin(theta))
    
    # Center the worm
    x -= np.mean(x)
    y -= np.mean(y)
    
    return x, y

def create_worm_gait_animation(
    young_data: np.ndarray, 
    old_data: np.ndarray, 
    output_path: str,
    frames: int = 500,
    fps: int = 30
):
    """
    Creates a split-screen animation comparing young vs old worm gait.
    Left side: Physical 2D worm shape.
    Right side: 3D phase space (Strange Attractor) of the first 3 eigenworms.
    """
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(16, 9), facecolor='black')
    
    # Grid setup
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Young Worm Axes
    ax_young_shape = fig.add_subplot(gs[0, 0])
    ax_young_phase = fig.add_subplot(gs[0, 1], projection='3d')
    
    # Old Worm Axes
    ax_old_shape = fig.add_subplot(gs[1, 0])
    ax_old_phase = fig.add_subplot(gs[1, 1], projection='3d')
    
    axes = [ax_young_shape, ax_young_phase, ax_old_shape, ax_old_phase]
    for ax in axes:
        ax.set_facecolor('black')
        if hasattr(ax, 'zaxis'):
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_zticks([])
        # Make grid invisible for cyber aesthetic
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_young_shape.set_title("Young Worm (Limit Cycle)", color='cyan', fontsize=16)
    ax_young_phase.set_title("Young Attractor", color='cyan', fontsize=16)
    
    ax_old_shape.set_title("Aged Worm (Thermodynamic Degradation)", color='magenta', fontsize=16)
    ax_old_phase.set_title("Aged Attractor", color='magenta', fontsize=16)

    # Setup limits
    # Assuming z-scored data, range is approx -3 to 3 for phase
    for ax in [ax_young_phase, ax_old_phase]:
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([-4, 4])
        
    for ax in [ax_young_shape, ax_old_shape]:
        ax.set_xlim([-60, 60])
        ax.set_ylim([-60, 60])
        ax.set_aspect('equal')
        
    # Lines
    line_young_shape, = ax_young_shape.plot([], [], lw=4, color='cyan', alpha=0.8)
    line_old_shape, = ax_old_shape.plot([], [], lw=4, color='magenta', alpha=0.8)
    
    line_young_phase, = ax_young_phase.plot([], [], [], lw=1.5, color='cyan', alpha=0.9)
    line_old_phase, = ax_old_phase.plot([], [], [], lw=1.5, color='magenta', alpha=0.9)
    
    # Particle head for phase space
    head_young_phase, = ax_young_phase.plot([], [], [], marker='o', color='white', markersize=8)
    head_old_phase, = ax_old_phase.plot([], [], [], marker='o', color='white', markersize=8)

    # Trail length for phase space
    trail_len = 100

    def init():
        line_young_shape.set_data([], [])
        line_old_shape.set_data([], [])
        line_young_phase.set_data([], [])
        line_young_phase.set_3d_properties([])
        line_old_phase.set_data([], [])
        line_old_phase.set_3d_properties([])
        head_young_phase.set_data([], [])
        head_young_phase.set_3d_properties([])
        head_old_phase.set_data([], [])
        head_old_phase.set_3d_properties([])
        return (line_young_shape, line_old_shape, 
                line_young_phase, line_old_phase,
                head_young_phase, head_old_phase)

    def update(frame):
        # Update Young
        x_y, y_y = generate_worm_shape(young_data[frame])
        line_young_shape.set_data(x_y, y_y)
        
        start_idx = max(0, frame - trail_len)
        trail_y = young_data[start_idx:frame+1]
        if len(trail_y) > 0:
            line_young_phase.set_data(trail_y[:, 0], trail_y[:, 1])
            line_young_phase.set_3d_properties(trail_y[:, 2])
            head_young_phase.set_data([trail_y[-1, 0]], [trail_y[-1, 1]])
            head_young_phase.set_3d_properties([trail_y[-1, 2]])
            
        # Update Old
        x_o, y_o = generate_worm_shape(old_data[frame])
        line_old_shape.set_data(x_o, y_o)
        
        trail_o = old_data[start_idx:frame+1]
        if len(trail_o) > 0:
            line_old_phase.set_data(trail_o[:, 0], trail_o[:, 1])
            line_old_phase.set_3d_properties(trail_o[:, 2])
            head_old_phase.set_data([trail_o[-1, 0]], [trail_o[-1, 1]])
            head_old_phase.set_3d_properties([trail_o[-1, 2]])
            
        # Add slight rotation to 3D plots
        ax_young_phase.view_init(elev=20, azim=frame * 0.5)
        ax_old_phase.view_init(elev=20, azim=frame * 0.5)

        return (line_young_shape, line_old_shape, 
                line_young_phase, line_old_phase,
                head_young_phase, head_old_phase)

    ani = animation.FuncAnimation(
        fig, update, frames=frames,
        init_func=init, blit=False, interval=1000/fps
    )
    
    logger.info(f"Saving animation to {output_path}...")
    try:
        ani.save(output_path, fps=fps)
        logger.info("Animation saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save animation: {e}")
    plt.close(fig)
