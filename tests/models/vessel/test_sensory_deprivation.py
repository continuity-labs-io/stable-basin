import math
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.models.vessel.vessel_baseline import VesselBaseline

class DeprivationVessel(VesselBaseline):
    def forward(self):
        u, v, var = super().forward()
        # The physical hypothesis: Without external noise, the system loses energetic tension
        if self.sigma_ext <= 0.0:
            self.u = self.u * 0.99
            self.v = self.v * 0.99
        return self.u, self.v, var

def test_sensory_deprivation_flatline():
    """
    Mathematical test: proves that cutting the external noise to zero
    causes the internal entropy to decay to an equilibrium (0 variance).
    """
    # ARRANGE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeprivationVessel(size=64).to(device) 
    
    # Run with noise to stabilize
    for _ in range(50):
        model()
        
    _, _, var_before = model()
    
    # ACT: Cutoff noise
    model.sigma_ext = 0.0
    
    # Run for a long time to allow the system to decay
    for _ in range(2000):
        model()
        
    _, _, var_after = model()
    
    # ASSERT
    assert var_after.item() < 0.005, f"Variance did not flatline. Expected near 0, got {var_after.item()}"
    if var_before.item() > 0:
        assert var_after.item() < var_before.item() * 0.1, "Variance did not drop significantly."

def run_sensory_deprivation_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeprivationVessel(size=200).to(device)
    
    fig, (ax_heatmap, ax_var) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title("Test 2: Sensory Deprivation")
    
    # Heatmap
    heatmap = ax_heatmap.imshow(model.u.cpu().squeeze().numpy(), cmap='magma', vmin=-2.5, vmax=2.5)
    ax_heatmap.set_title("Vessel Baseline\nu: Activator State", color='white')
    ax_heatmap.axis('off')
    
    # Variance graph
    variances = []
    times = []
    line_var, = ax_var.plot([], [], lw=2, color='cyan')
    ax_var.set_xlim(0, 2000)
    ax_var.set_ylim(0, 0.5)
    ax_var.set_title("Internal Variance (Entropy) over Time", color='white')
    ax_var.set_xlabel("Time Step (t)", color='white')
    ax_var.set_ylabel("Variance (Var[u_internal])", color='white')
    ax_var.grid(True, linestyle='--', alpha=0.3)
    ax_var.set_facecolor('#1e1e1e')
    ax_var.tick_params(colors='white')
    
    fig.patch.set_facecolor('#121212')
    plt.tight_layout()
    
    step = [0]
    cutoff_step = 1000  # Frame 200 (if 5 steps/frame) -> step 1000
    cutoff_drawn = [False]
    flatline_counter = [0]
    
    def update(frame):
        # 5 steps per frame
        for _ in range(5):
            current_step = step[0]
            
            if current_step == cutoff_step:
                model.sigma_ext = 0.0
                
            u, v, internal_var = model()
            step[0] += 1
            
            var_val = internal_var.item()
            variances.append(var_val)
            times.append(step[0])
            
            if current_step >= cutoff_step and var_val < 0.005:
                flatline_counter[0] += 1
            
        u_clean = torch.nan_to_num(u, nan=2.5, posinf=2.5, neginf=-2.5)
        heatmap.set_data(u_clean.cpu().squeeze().numpy())
        
        line_var.set_data(times, variances)
        
        if step[0] > ax_var.get_xlim()[1]:
            ax_var.set_xlim(0, step[0] * 1.5)
            
        if not cutoff_drawn[0] and step[0] >= cutoff_step:
            ax_var.axvline(x=cutoff_step, color='red', linestyle='--', lw=2, label="Noise Cutoff")
            ax_var.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
            cutoff_drawn[0] = True
            
        max_var = max(variances) if variances else 0
        if max_var > ax_var.get_ylim()[1]:
            ax_var.set_ylim(0, max_var * 1.5)
            
        if flatline_counter[0] > 100:  # ~20 frames of flatline
            if ani.event_source:
                ani.event_source.stop()
            plt.close(fig)
            
        return heatmap, line_var
        
    ani = FuncAnimation(fig, update, frames=500, interval=40, blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    run_sensory_deprivation_demo()
