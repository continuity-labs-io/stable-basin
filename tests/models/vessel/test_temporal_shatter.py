import math
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from src.models.vessel.vessel_baseline import VesselBaseline

def test_temporal_shatter_destabilization():
    """
    Mathematical invariant test: proves that exceeding the critical dt threshold
    will cause the differential equations to destabilize and internal entropy to spike.
    """
    # ARRANGE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use smaller size for faster testing
    model = VesselBaseline(size=64).to(device) 
    model.dt = 0.01
    
    # Stabilize
    for _ in range(10):
        model()
        
    _, _, var_safe = model()
    
    # ACT
    model.dt = 1.0 # Critical latency
    for _ in range(100):
        model()
        
    _, _, var_unsafe = model()
    
    # ASSERT
    assert not torch.isnan(var_safe), "Simulation started with NaNs."
    # If the simulation totally shatters, variance could become NaN or Inf.
    # We treat NaN/Inf as a successful shatter.
    if math.isnan(var_unsafe.item()) or math.isinf(var_unsafe.item()):
        pass # Successfully shattered
    else:
        assert var_unsafe.item() > var_safe.item() * 10, "Temporal shatter failed: Entropy did not catastrophically spike."
        assert var_unsafe.item() > 1.0, f"Expected massive entropy spike, got {var_unsafe.item()}"

def run_temporal_shatter_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesselBaseline(size=200).to(device)
    
    # 3 panels: left is heatmap, right top is dt, right bottom is variance
    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title("Test 1: Temporal Shatter")
    
    ax_heatmap = plt.subplot(1, 2, 1)
    ax_dt = plt.subplot(2, 2, 2)
    ax_var = plt.subplot(2, 2, 4, sharex=ax_dt)
    
    # Heatmap
    heatmap = ax_heatmap.imshow(model.u.cpu().squeeze().numpy(), cmap='magma', vmin=-2.5, vmax=2.5)
    ax_heatmap.set_title("Vessel Baseline\nu: Activator State", color='white')
    ax_heatmap.axis('off')
    
    # dt graph
    dts = []
    times = []
    line_dt, = ax_dt.plot([], [], lw=2, color='orange')
    ax_dt.set_xlim(0, 1000)
    ax_dt.set_ylim(0, 0.6)
    ax_dt.set_title("Integration Step (Δt) over Time", color='white')
    ax_dt.set_ylabel("Δt", color='white')
    ax_dt.grid(True, linestyle='--', alpha=0.3)
    ax_dt.set_facecolor('#1e1e1e')
    ax_dt.tick_params(colors='white', labelbottom=False)
    
    # Variance graph
    variances = []
    line_var, = ax_var.plot([], [], lw=2, color='cyan')
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
    shatter_counter = [0]
    
    def update(frame):
        # 5 steps per frame
        for _ in range(5):
            current_step = step[0]
            
            # Linearly interpolate dt from 0.01 to 0.5 over 1000 steps
            if current_step < 1000:
                current_dt = 0.01 + (0.5 - 0.01) * (current_step / 1000.0)
            else:
                current_dt = 0.5
                
            model.dt = current_dt
            
            u, v, internal_var = model()
            step[0] += 1
            
            # To handle NaNs that might occur after shatter
            var_val = internal_var.item()
            if math.isnan(var_val) or math.isinf(var_val):
                # Lock variance high to show it exploded visually
                var_val = 50.0 
            
            if var_val > 1.0:
                shatter_counter[0] += 1
                
            dts.append(current_dt)
            variances.append(var_val)
            times.append(step[0])
            
        # Clean u of NaNs for plotting to avoid matplotlib crash
        u_clean = torch.nan_to_num(u, nan=2.5, posinf=2.5, neginf=-2.5)
        heatmap.set_data(u_clean.cpu().squeeze().numpy())
        
        line_dt.set_data(times, dts)
        line_var.set_data(times, variances)
        
        if step[0] > ax_dt.get_xlim()[1]:
            ax_dt.set_xlim(0, step[0] * 1.5)
            
        max_var = max(variances) if variances else 0
        if max_var > ax_var.get_ylim()[1]:
            ax_var.set_ylim(0, max_var * 1.5)
            
        if shatter_counter[0] > 100:  # e.g., 20 frames after shatter happens
            if ani.event_source:
                ani.event_source.stop()
            # Don't close the figure so the user can inspect the final state
            
        return heatmap, line_dt, line_var
        
    ani = FuncAnimation(fig, update, frames=200, interval=40, blit=False, repeat=False)
    
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
    run_temporal_shatter_demo()
