"""
Demo 05: Flight Recorder

This script simulates a real-time, asynchronous telemetry stream of the MaskAwareMamba's
internal physics state using the TelemetryExhaust bridge to the Rerun viewer.
It streams Fedichev Macrostates and the phase space geometry without blocking inference.
"""

import os
import time
import torch
import numpy as np
import logging

from src.models.ssm.mask_aware_mamba import MaskAwareMamba
from src.metrics.metrics import ThermodynamicMetrics
from src.metrics.telemetry_exhaust import TelemetryExhaust
from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("FlightRecorderDemo")

def main():
    device = get_optimal_device()
    logger.info("\n[*] BOOTING DEMO 05: THE FLIGHT RECORDER")
    
    # 1. Initialize our engine
    engine = MaskAwareMamba(input_dim=114, d_model=256, mask_aware=False).to(device)
    engine.eval()
    
    # 2. Initialize the metrics extractor and telemetry bridge
    metrics_engine = ThermodynamicMetrics()
    
    output_dir = "output/demo"
    os.makedirs(output_dir, exist_ok=True)
    rrd_path = os.path.join(output_dir, "05_flight_recorder.rrd")
    
    exhaust = TelemetryExhaust(mode="save", save_path=rrd_path)
    logger.info(f"[*] Telemetry stream initialized. Saving to: {rrd_path}")
    
    # 3. Simulate an incoming biological data stream
    # Let's say 200 time steps of 114-D data.
    logger.info("[*] Generating mock biological data stream...")
    total_steps = 200
    # Simulate a baseline state transitioning into a perturbed state
    baseline_data = torch.randn(total_steps, 114, device=device) * 0.1
    perturbed_data = baseline_data.clone()
    
    # Inject an anomaly starting at T=100
    perturbed_data[100:, :] += 2.0
    
    # For realistic "streaming", we process up to step t
    logger.info("[*] Commencing asynchronous telemetry stream...")
    dt = 0.05 # simulated physical time step in seconds
    
    # We need sequences for Fedichev Macrostates. We will accumulate the latent states.
    z_baseline_list = []
    z_perturbed_list = []
    
    for t in range(1, total_steps):
        # 1. Forward Pass (Inference)
        with torch.no_grad():
            # In a real streaming app, we'd process windowed chunks. Here we simulate it.
            # MaskAwareMamba expects [Batch, Seq, Features]
            batch_base = baseline_data[:t].unsqueeze(0)
            batch_pert = perturbed_data[:t].unsqueeze(0)
            
            _, _, hidden_base = engine.forward(batch_base, return_hidden=True)
            _, _, hidden_pert = engine.forward(batch_pert, return_hidden=True)
            
            # The hidden state is [Batch, Seq, d_model]. We want the latest state.
            z_base_t = hidden_base[0, -1, :] # [d_model]
            z_pert_t = hidden_pert[0, -1, :] # [d_model]
            
            z_baseline_list.append(z_base_t)
            z_perturbed_list.append(z_pert_t)
            
        # Only start extracting metrics once we have enough window history (e.g. 5 steps)
        if t > 5:
            # 2. Calculate Fedichev Macrostates using our new wrapper
            z_base_seq = torch.stack(z_baseline_list)
            z_pert_seq = torch.stack(z_perturbed_list)
            
            # Extract macrostates
            macrostates = metrics_engine.extract_fedichev_macrostates(
                z_baseline=z_base_seq, 
                z_perturbed=z_pert_seq, 
                window_size=4
            )
            
            # The macrostates return lists of length `t-1`. We log the latest one for streaming.
            z0 = macrostates["z0_volatility"][-1]
            Z = macrostates["Z_entropic_damage"][-1]
            eps0 = macrostates["epsilon_0_ksm"][-1]
            
            # We also compute LLE chaos
            lle_list = metrics_engine.calculate_lle(z_pert_seq, window_size=4, dt=dt)
            lle = lle_list[-1]
            
            # 3. Stream Telemetry via the Exhaust
            physical_time = t * dt
            exhaust.update_time(frame_idx=t, time_sec=physical_time)
            
            # Log scalars
            exhaust.log_fedichev_macrostates(
                z0_volatility=z0,
                Z_entropic_damage=Z,
                epsilon_0_ksm=eps0,
                lle_chaos=lle
            )
            
            # Log 3D Attractor Basin Geometry (plot all points up to t)
            # The exhaust requires [Num_Points, 3]. Slice the first 3 dimensions.
            exhaust.log_attractor_basin(z_pert_seq[:, :3])
            
            # Log infrastructure (mock VRAM and Perfusion)
            vram = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 1024.0 + np.sin(t*0.1)*50.0
            perfusion = 40.0 + np.random.randn() * 2.0
            exhaust.log_infrastructure(vram_mb=vram, perfusion_rate=perfusion)
            
    logger.info(f"[+] Demo 05 Complete. You can now open {rrd_path} in the Rerun viewer by running: rerun {rrd_path}")

if __name__ == "__main__":
    main()
