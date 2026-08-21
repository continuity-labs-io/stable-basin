import math
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from src.models.vessel.vessel_baseline import VesselBaseline

def inject_wound(model):
    """
    Injects a 10x10 energy spike (u = 5.0) that intersects the wall and breaches the interior.
    The center of the wound is placed exactly on the inner edge of the wall.
    """
    center_y = model.size // 2
    # Place wound on the right side, at radius_internal boundary
    center_x = int(model.size / 2 + model.radius_internal)
    half_size = 5 # 10x10 wound
    
    # Apply to u
    model.u.data[..., center_y-half_size:center_y+half_size, center_x-half_size:center_x+half_size] = 5.0

def test_equilibrium_wound_recovery():
    """
    Mathematical test: proves that the system can recover from a massive energy spike
    that ruptures the Markov blanket.
    """
    # ARRANGE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesselBaseline(size=64, radius_internal=20).to(device)
    
    # Stabilize
    for _ in range(100):
        model()
        
    _, _, var_baseline = model()
    
    # ACT: Inject Wound
    inject_wound(model)
    
    # Capture immediate spike before u^3 collapse
    var_spike = torch.var(model.u[model.mask_internal])
    
    # Let it recover
    for _ in range(1000):
        model()
        
    _, _, var_recovered = model()
    
    # ASSERT
    assert var_spike.item() > var_baseline.item() * 5, "Entropy did not spike after the wound."
    assert var_recovered.item() < var_spike.item() * 0.5, "Entropy did not decay back toward baseline."

def run_equilibrium_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use 128 size for a better view of the 10x10 wound
    model = VesselBaseline(size=128).to(device)
    
    fig, (ax_heatmap, ax_var) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title("Test 3: Equilibrium (The Wound)")
    
    # Heatmap
    heatmap = ax_heatmap.imshow(model.u.cpu().squeeze().numpy(), cmap='magma', vmin=-2.5, vmax=5.0)
    ax_heatmap.set_title("Vessel Baseline\nu: Activator State", color='white')
    ax_heatmap.axis('off')
    
    # Variance graph
    variances = []
    times = []
    line_var, = ax_var.plot([], [], lw=2, color='lime')
    ax_var.set_xlim(0, 4000)
    ax_var.set_ylim(0, 1.0)
    ax_var.set_title("Internal Variance (Entropy) over Time", color='white')
    ax_var.set_xlabel("Time Step (t)", color='white')
    ax_var.set_ylabel("Variance (Var[u_internal])", color='white')
    ax_var.grid(True, linestyle='--', alpha=0.3)
    ax_var.set_facecolor('#1e1e1e')
    ax_var.tick_params(colors='white')
    
    fig.patch.set_facecolor('#121212')
    plt.tight_layout()
    
    step = [0]
    wound_step = 1500  # Frame 300
    wound_drawn = [False]
    
    def update(frame):
        # 5 steps per frame
        for _ in range(5):
            current_step = step[0]
            
            if current_step == wound_step:
                inject_wound(model)
                
            u, v, internal_var = model()
            step[0] += 1
            
            var_val = internal_var.item()
            if math.isnan(var_val) or math.isinf(var_val):
                var_val = 50.0 
                
            variances.append(var_val)
            times.append(step[0])
            
        u_clean = torch.nan_to_num(u, nan=2.5, posinf=5.0, neginf=-2.5)
        heatmap.set_data(u_clean.cpu().squeeze().numpy())
        
        line_var.set_data(times, variances)
        
        if step[0] > ax_var.get_xlim()[1]:
            ax_var.set_xlim(0, step[0] * 1.5)
            
        if not wound_drawn[0] and step[0] >= wound_step:
            ax_var.axvline(x=wound_step, color='red', linestyle='--', lw=2, label="Wound Injection")
            ax_var.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
            wound_drawn[0] = True
            
        max_var = max(variances) if variances else 0
        if max_var > ax_var.get_ylim()[1]:
            ax_var.set_ylim(0, max_var * 1.5)
            
        return heatmap, line_var
        
    ani = FuncAnimation(fig, update, frames=800, interval=40, blit=False, repeat=False)
    
    # Pause button
    is_paused = [False]
    def toggle_pause(event):
        if is_paused[0]:
            ani.event_source.start()
            is_paused[0] = False
            bpause.label.set_text("Pause")
        else:
            ani.event_source.stop()
            is_paused[0] = True
            bpause.label.set_text("Resume")
            
    ax_pause = plt.axes([0.85, 0.02, 0.1, 0.05])
    bpause = Button(ax_pause, 'Pause', color='#1e1e1e', hovercolor='#3a3a3a')
    bpause.label.set_color('white')
    bpause.on_clicked(toggle_pause)
    
    plt.show()

if __name__ == "__main__":
    run_equilibrium_demo()
