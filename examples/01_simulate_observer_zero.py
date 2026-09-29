"""
FitzHugh-Nagumo reaction-diffusion system with an insulated boundary.
"""
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.models.vessel.vessel_baseline import VesselBaseline

def run_simulation(size, sigma_ext, v_wall_threshold, save, seed, steps):
    if seed is not None:
        torch.manual_seed(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesselBaseline(
        size=size,
        sigma_ext=sigma_ext,
        v_wall_threshold=v_wall_threshold
    ).to(device)

    # 2-panel figure: Left is heatmap, Right is line graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title("FitzHugh-Nagumo Reaction-Diffusion")

    # Left Panel: 2D Heatmap
    heatmap = ax1.imshow(model.u.cpu().squeeze().numpy(), cmap="magma", vmin=-2.5, vmax=2.5)
    ax1.set_title("FitzHugh-Nagumo System\nu: Activator State", color="white")
    ax1.axis("off")

    # Right Panel: Live updating line graph
    variances = []
    times = []
    (line,) = ax2.plot([], [], lw=2, color="cyan")
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 0.5)
    ax2.set_title("Internal variance over Time", color="white")
    ax2.set_xlabel("Time Step (t)", color="white")
    ax2.set_ylabel("Variance (Var[u_internal])", color="white")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.set_facecolor("#1e1e1e")
    ax2.tick_params(colors="white")

    fig.patch.set_facecolor("#121212")
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

    frames = steps if save else 500
    ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    
    if save:
        ani.save(save, writer="ffmpeg")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FitzHugh-Nagumo reaction-diffusion system.")
    parser.add_argument("--size", type=int, default=200, help="Simulation size")
    parser.add_argument("--sigma-ext", type=float, default=0.5, help="External noise sigma")
    parser.add_argument("--v-wall-threshold", type=float, default=2.0, help="Wall threshold")
    parser.add_argument("--save", type=str, default=None, help="Save to mp4 path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=500, help="Number of frames for animation")
    
    args = parser.parse_args()
    print("Running FitzHugh-Nagumo reaction-diffusion system with an insulated boundary...")
    run_simulation(
        size=args.size,
        sigma_ext=args.sigma_ext,
        v_wall_threshold=args.v_wall_threshold,
        save=args.save,
        seed=args.seed,
        steps=args.steps
    )
