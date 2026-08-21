import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data.sim2real.gevi_dataloader import GEVIDataloader
from src.models.encoders.gevi_encoder import GEVIEncoder

class MockKinematicRouting(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 1D Conv acting as a finite difference estimator
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=4, padding=3)
        
        # Initialize kernel to a 3rd-order finite difference (jerk)
        # We want causal routing, so we'll pad on the left and truncate the right later
        # weight shape: (out_channels, in_channels, kernel_size)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[-1.0, 3.0, -3.0, 1.0]]]))
            self.conv.bias.fill_(0.0)
            
    def forward(self, x):
        """
        x: (batch, channels, seq_len)
        Returns:
        dt: (batch, seq_len)
        """
        # (batch, channels, seq_len + padding)
        dx = self.conv(x)
        # Truncate right padding to maintain causality
        dx = dx[:, :, :-3]
        
        # dt gating mechanism: softplus over absolute value of the finite difference
        # High inertia (low dt) during smooth regions, Low inertia (high dt) during spikes
        dt = F.softplus(torch.abs(dx))
        return dt.squeeze(1)

def test_inertial_routing_plot():
    dataloader = GEVIDataloader(
        gevi_sample_rate=1000, 
        target_clock_hz=1000, # 1:1 compression so we can see the raw wave
        baseline_mv=-70.0,
        noise_std=2.0,
        spike_prob=0.01,
        spike_mv=100.0,
        anomaly_start_frame=300,
        anomaly_noise_std=40.0
    )
    
    # Generate synthetic GEVI sequence (1 batch, 1 channel, 500 steps)
    raw_signal = dataloader.generate_synthetic_gevi(
        batch_size=1, 
        target_time_steps=500, 
        device=torch.device('cpu'), 
        is_healthy=False
    )
    
    # Run through our Mock Kinematic Routing
    routing = MockKinematicRouting(in_channels=1)
    
    with torch.no_grad():
        dt = routing(raw_signal)
        
    signal_np = raw_signal[0, 0].numpy()
    dt_np = dt[0].numpy()
    
    # Output path compliance
    output_dir = "output/tests/models/ssm/physics"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "inertia_dial_validation.png")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Top Panel: Raw biological signal
    ax1.plot(signal_np, color='blue', alpha=0.7)
    ax1.set_title("Raw Biological Signal (GEVI)")
    ax1.set_ylabel("Membrane Potential (mV)")
    
    # Bottom Panel: Delta t 'Inertial Dial'
    ax2.plot(dt_np, color='red', alpha=0.9)
    ax2.set_title("Δt 'Inertial Dial' (Kinematic Routing)")
    ax2.set_ylabel("Δt (Time-Step)")
    ax2.set_xlabel("Time Step")
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
    print(f"Saved Inertial Dial plot to {out_path}")
    
    # Assertions
    # Baseline phase: steps 0 to 299
    # Anomaly phase: steps 300 to 500
    baseline_dt = dt_np[10:290].mean() # ignore edge effects at start
    anomaly_dt = dt_np[310:490].mean()
    
    assert anomaly_dt > baseline_dt * 1.5, f"Inertial Dial failed to significantly spike during anomalies. Anomaly mean: {anomaly_dt:.2f}, Baseline mean: {baseline_dt:.2f}"
