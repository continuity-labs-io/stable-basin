import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import time

from src.pipeline.sim2real.human_telemetry_dataloader import HumanTelemetryLoader
from src.pipeline.sim2real.epigenetic_entropy_dataloader import EpigeneticEntropyLoader
from src.models.ssm.mask_aware_mamba import MaskAwareMamba
from src.metrics.metrics import ThermodynamicMetrics
from src.core.rejuvenation_controller import RejuvenationFlightController

# Suppress PyDMD debug spam
logging.getLogger("DiagnosticLogger").setLevel(logging.WARNING)
logger = logging.getLogger("SimulationDemo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    print("="*60)
    print("🚀 INITIATING HUMAN DIGITAL TWIN REJUVENATION SIMULATION 🚀")
    print("="*60)
    
    device = torch.device("cpu")
    
    # 1. Initialize Components
    logger.info("[1/4] Booting Biological Flight Computer...")
    engine = MaskAwareMamba(input_dim=6, d_model=32, mask_aware=True).to(device)
    metrics = ThermodynamicMetrics()
    controller = RejuvenationFlightController(engine, metrics, hysteresis_frames=3)
    
    # 2. Generate 20-minute timeline (1200 seconds at 250Hz = 300,000 frames)
    logger.info("[2/4] Generating 20-minute patient telemetry...")
    seq_len = 300000
    dataset = HumanTelemetryLoader(size=1, seq_len=seq_len)
    batch = dataset[0]
    
    base_x = batch["x_raw"].to(device)
    mask = batch["mask"].to(device)
    
    # Apply therapy shock at minute 5
    shock_x = base_x.clone()
    shock_x = dataset.apply_therapy_shock(shock_x, mask, time_minutes=5.0)
    
    # --- PASS 1: Find the exact abort timestamp ---
    logger.info("[3/4] Running closed-loop safety evaluation...")
    shock_nan = shock_x.clone()
    shock_nan[mask == 0] = float('nan')
    z_shock = engine.get_hidden_states(shock_nan.unsqueeze(0))[0]
    
    abort_sec = None
    # Scan from min 5 to min 10
    for sec in range(5*60, 10*60):
        # 41-frame window ensures exactly 1 PyDMD calculation for speed
        z_win = z_shock[sec*250 : sec*250 + 41] 
        ksm = metrics.calculate_ksm(z_win, window_size=40)[-1]
        csd = metrics.calculate_csd(z_win, window_size=40)[-1]
        
        res = controller.evaluate_safety_margins(ksm, csd)
        if res["action"] == "EMERGENCY_ABORT":
            abort_sec = sec
            break
            
    if abort_sec is None:
        abort_sec = 8 * 60 # Fallback
        
    abort_min = abort_sec / 60.0
    logger.info(f"[*] Controller triggered EMERGENCY ABORT at minute {abort_min:.2f}!")
    
    # --- PASS 2: Construct Final Spliced Timeline ---
    final_x = base_x.clone()
    # Shock exists only between infusion start (min 5) and abort
    final_x[5*60*250 : abort_sec*250] = shock_x[5*60*250 : abort_sec*250]
    
    final_nan = final_x.clone()
    final_nan[mask == 0] = float('nan')
    z_final = engine.get_hidden_states(final_nan.unsqueeze(0))[0]
    
    logger.info("[4/4] Extracting Thermodynamic Metrics (1Hz resolution) & Epigenetic Entropy...")
    ksm_plot = []
    csd_plot = []
    z_plot = []
    
    # Downscaled epigenetic loaders for speed
    loader_50 = EpigeneticEntropyLoader(biological_age=50, size=1, seq_len=1, n_cells=100, n_cpgs=100)
    loader_45 = EpigeneticEntropyLoader(biological_age=45, size=1, seq_len=1, n_cells=100, n_cpgs=100)
    
    t_start = time.time()
    for sec in range(1200):
        # KSM / CSD
        z_win = z_final[sec*250 : sec*250 + 41]
        if len(z_win) < 41:
            ksm_plot.append(1.0)
            csd_plot.append(0.0)
        else:
            ksm_plot.append(metrics.calculate_ksm(z_win, window_size=40)[-1])
            csd_plot.append(metrics.calculate_csd(z_win, window_size=40)[-1])
            
        # Epigenetic Entropy Z (Step function at minute 15)
        if sec < 15 * 60:
            cpg_tensor = loader_50[0]["cpg_tensor"]
        else:
            cpg_tensor = loader_45[0]["cpg_tensor"]
            
        z_val = metrics.calculate_epigenetic_dispersion(cpg_tensor)[0]
        z_plot.append(z_val)
        
        if sec > 0 and sec % 300 == 0:
            logger.info(f"    ... processed {sec/60:.0f} simulated minutes.")
            
    logger.info(f"Metrics extracted in {time.time() - t_start:.2f} seconds.")
    
    # --- VISUALIZATION DASHBOARD ---
    logger.info("Generating Dark-Mode Dashboard...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    
    time_axis = np.arange(1200) / 60.0 # In minutes
    
    # Panel 1: Raw HRV
    ax = axes[0]
    hrv = final_x[:, 0].numpy()
    # Downsample HRV to 1Hz for plotting
    hrv_1hz = hrv[::250][:1200]
    ax.plot(time_axis, hrv_1hz, color='#cccccc', linewidth=1)
    ax.set_title("Continuous Wearable Telemetry (HRV)", color='white', fontsize=14, pad=10)
    ax.set_ylabel("RR Interval (ms)")
    
    # IV State Background
    iv_on = (time_axis >= 5) & (time_axis <= abort_min)
    ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1], where=iv_on, color='green', alpha=0.2, label='IV Flowing')
    ax.axvline(abort_min, color='red', linestyle='--', linewidth=2, label='EMERGENCY ABORT')
    ax.legend(loc="upper right")
    
    # Panel 2: KSM
    ax = axes[1]
    ax.plot(time_axis, ksm_plot, color='cyan', linewidth=2)
    ax.axhline(0.85, color='red', linestyle=':', linewidth=2, label='Critical Threshold (0.85)')
    ax.axvline(abort_min, color='red', linestyle='--', linewidth=2)
    ax.set_title("Koopman Stability Metric (ε_0)", color='cyan', fontsize=14, pad=10)
    ax.set_ylabel("Stability [0, 1]")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left")
    
    # Panel 3: CSD
    ax = axes[2]
    ax.plot(time_axis, csd_plot, color='magenta', linewidth=2)
    ax.axvline(abort_min, color='red', linestyle='--', linewidth=2)
    ax.set_title("Critical Slowing Down (D_0 Biological Noise)", color='magenta', fontsize=14, pad=10)
    ax.set_ylabel("Variance / AR1")
    
    # Panel 4: Epigenetic Entropy (Z)
    ax = axes[3]
    ax.plot(time_axis, z_plot, color='yellow', linewidth=2, drawstyle='steps-post')
    ax.axvline(15, color='white', linestyle=':', linewidth=2, label='Post-Recovery Measurement')
    ax.set_title("Configurational Entropy (Z - Biological Age)", color='yellow', fontsize=14, pad=10)
    ax.set_xlabel("Clinical Time (Minutes)", fontsize=12)
    ax.set_ylabel("Methylation Dispersion")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/10_human_rejuvenation_sim.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Dashboard saved to: output/10_human_rejuvenation_sim.png")
    print("="*60)
    print("Simulation Complete!")

if __name__ == "__main__":
    main()
