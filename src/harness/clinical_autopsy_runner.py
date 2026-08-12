import os
import time
import json
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from src.data.ephys.pharma_shock_dataset import PharmacologicalShockDataset
from src.models.ssm.meld_engine import MeldEngine
from src.models.ssm.baseline_ssm import BaselineSSM
from src.models.attention.baseline_transformer import BaselineTransformer
from src.models.ssm.mask_aware_ssm import MaskAwareSSM
from src.metrics.metrics import ThermodynamicMetrics
from src.metrics.mamba_lrp import MambaLRPEpsilon
from src.metrics.autopsy_engine import ThermodynamicAutopsyEngine
from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ModelAdapter(nn.Module):
    """
    Adapter to give BaselineSSM, BaselineTransformer, and MaskAwareSSM
    the same interface as MeldEngine (input projection, forward forecasting head, etc).
    """
    def __init__(self, core_model, input_dim=1024, d_model=256, is_mask_aware=False):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.core = core_model
        self.forward_head = nn.Linear(d_model, input_dim)
        self.is_mask_aware = is_mask_aware

    def forward(self, x, return_hidden=False):
        h = self.input_proj(x)
        if self.is_mask_aware:
            latent_gate = torch.ones_like(h)
            hidden_states = self.core(h, latent_gate)
        else:
            hidden_states = self.core(h)
        pred_t_plus_1 = F.softplus(self.forward_head(hidden_states))
        
        # We don't strictly need reconstructed_t for residual MSE, but we match signature
        reconstructed_t = pred_t_plus_1 
        
        if return_hidden:
            return pred_t_plus_1, reconstructed_t, hidden_states
        return pred_t_plus_1, reconstructed_t

    def get_hidden_states(self, x):
        _, _, hidden_states = self.forward(x, return_hidden=True)
        return hidden_states


def main():
    parser = argparse.ArgumentParser(description="Clinical Autopsy Runner")
    parser.add_argument("--model-type", type=str, required=True, 
                        choices=["meld", "baseline", "transformer", "mask_aware"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=2000)
    parser.add_argument("--ksm-threshold", type=float, default=0.85)
    parser.add_argument("--png-name", type=str, required=True)
    parser.add_argument("--csv-name", type=str, required=True)
    args = parser.parse_args()

    device = get_optimal_device(verbose=True)
    
    # 1. Load Dataset
    logger.info(f"Loading PharmacologicalShockDataset (seq_len={args.seq_len})")
    # For a real run, this data needs to exist. Since we are creating the harness, we assume the data file is present
    # or will be placed at data/ephys/pharmacological_shock/Drug_2953_50uM.raw.h5
    try:
        dataset = PharmacologicalShockDataset(condition="50uM", seq_len=args.seq_len)
        # We extract just the first sequence as the continuous stream
        telemetry = dataset[0].unsqueeze(0).to(device)  # shape: [1, seq_len, 1024]
    except FileNotFoundError:
        logger.warning("Dataset file not found. Generating dummy telemetry for testing.")
        telemetry = torch.randn(1, args.seq_len, 1024, device=device)

    # 2. Initialize Model
    logger.info(f"Initializing {args.model_type} model")
    input_dim = 1024
    d_model = 256
    
    if args.model_type == "meld":
        model = MeldEngine(input_dim=input_dim, d_model=d_model, mask_aware=False).to(device)
    elif args.model_type == "baseline":
        core = BaselineSSM(d_model=d_model)
        model = ModelAdapter(core, input_dim=input_dim, d_model=d_model).to(device)
    elif args.model_type == "transformer":
        core = BaselineTransformer(d_model=d_model)
        model = ModelAdapter(core, input_dim=input_dim, d_model=d_model).to(device)
    elif args.model_type == "mask_aware":
        core = MaskAwareSSM(d_model=d_model)
        model = ModelAdapter(core, input_dim=input_dim, d_model=d_model, is_mask_aware=True).to(device)

    # Apply monkey patch for LRP attribution
    lrp_engine = MambaLRPEpsilon(model)
    model.compute_attribution = lrp_engine.attribute

    # 3. Burn-in Training Loop with Residual MSE
    # We use the early portion (e.g., first 500 frames) as the stable healthy tissue
    burn_in_len = min(500, args.seq_len // 4)
    x_train = telemetry[:, :burn_in_len, :]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    logger.info(f"Starting Burn-In Training for {args.epochs} epochs on first {burn_in_len} frames")
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Predict delta
        pred_t_plus_1, _, hidden = model(x_train, return_hidden=True)
        
        # Calculate residual MSE loss
        # L = || (x_hat_{t+1} - x_t) - (x_{t+1} - x_t) ||^2
        # Which algebraically is equivalent to predicting delta
        pred_delta = pred_t_plus_1[:, :-1, :] - x_train[:, :-1, :]
        true_delta = x_train[:, 1:, :] - x_train[:, :-1, :]
        
        loss = F.mse_loss(pred_delta, true_delta)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Residual MSE Loss: {loss.item():.6f}")

    # Calculate baseline KSM variance on training segment
    model.eval()
    with torch.no_grad():
        _, _, base_hidden = model(x_train, return_hidden=True)
        base_ksm = ThermodynamicMetrics(alpha=500.0).calculate_ksm(base_hidden)
        base_ksm_variance = torch.var(torch.tensor(base_ksm, dtype=torch.float32)).item()
        logger.info(f"Baseline Thermodynamic Stability (KSM Variance): {base_ksm_variance:.6e}")

    # 4. Dynamic Crash Detection
    logger.info("Running dynamic crash detection over full sequence")
    start_time = time.time()
    with torch.no_grad():
        pred_full, _, full_hidden = model(telemetry, return_hidden=True)
    inference_time = time.time() - start_time
    latency_ms = (inference_time / args.seq_len) * 1000
    logger.info(f"Inference Latency: {latency_ms:.2f} ms/frame")

    ksm_trajectory = ThermodynamicMetrics(alpha=500.0).calculate_ksm(full_hidden)
    ksm_np = np.array(ksm_trajectory)
    
    crash_frame = -1
    for t in range(len(ksm_np)):
        if ksm_np[t] < args.ksm_threshold:
            crash_frame = t
            break

    if crash_frame == -1:
        logger.warning(f"Crash frame not found! KSM never dropped below {args.ksm_threshold}.")
        crash_frame = len(ksm_np) - 1 # Default to end
    else:
        logger.info(f"Detected crash at frame {crash_frame} (KSM dropped below {args.ksm_threshold})")

    # 5. Causal Autopsy via MambaLRP
    logger.info("Executing Thermodynamic Autopsy Engine")
    # For autopsy engine, feature_names can be generic channel names
    feature_names = [f"Ch_{i}" for i in range(input_dim)]
    autopsy = ThermodynamicAutopsyEngine(model, feature_names=feature_names)
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
            
    try:
        report = autopsy.generate_autopsy(telemetry, crash_time_step=crash_frame)
    except Exception as e:
        logger.warning(f"Autopsy generation failed (possibly dummy data issues): {e}")
        report = {"error": str(e), "crash_frame": crash_frame}

    # 6. Artifacts Saving & Dashboard
    os.makedirs("output/harness", exist_ok=True)
    report_path = f"output/harness/clinical_autopsy_report_{args.model_type}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, cls=NpEncoder)
    logger.info(f"Saved Autopsy JSON to {report_path}")

    logger.info("Generating 3-panel dashboard")
    
    import csv
    csv_path = f"output/harness/{args.csv_name}"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "baseline_ksm_variance", "crash_frame", "inference_latency_ms_per_frame"])
        writer.writerow([args.model_type, base_ksm_variance, crash_frame, latency_ms])
    logger.info(f"Saved summary metrics to {csv_path}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Pool channels to 128 for plotting
    raw_np = telemetry.squeeze().cpu().numpy() # [seq_len, 1024]
    pool_factor = max(1, raw_np.shape[1] // 128)
    raw_pooled = raw_np[:, ::pool_factor].T # [128, seq_len]

    # Panel 1: Raw Telemetry Heatmap
    axes[0].imshow(raw_pooled, aspect='auto', cmap='viridis', origin='lower')
    axes[0].axvline(x=crash_frame, color='red', linestyle='--', label='Crash Frame')
    axes[0].set_title(f"{args.model_type.upper()} - 1,024-Ch Telemetry (MaxPool 128)")
    axes[0].set_ylabel("Channels")
    axes[0].legend()

    # Panel 2: KSM Over Time
    axes[1].plot(ksm_np, color='orange')
    axes[1].axhline(y=args.ksm_threshold, color='red', linestyle=':', label='Threshold')
    axes[1].axvline(x=crash_frame, color='red', linestyle='--')
    axes[1].set_title("Koopman Stability Metric (KSM)")
    axes[1].set_ylabel("KSM")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()

    # Panel 3: MambaLRP Causal Attribution
    logger.info("Extracting full attribution sequence for Panel 3 plotting...")
    try:
        relevance_tensor = model.compute_attribution(telemetry, target_time_step=crash_frame)
        rel_np = relevance_tensor.squeeze().cpu().numpy() # [seq_len, 1024]
        rel_pooled = rel_np[:, ::pool_factor].T # [128, seq_len]
        
        axes[2].imshow(rel_pooled, aspect='auto', cmap='magma', origin='lower')
        axes[2].axvline(x=crash_frame, color='white', linestyle='--')
        axes[2].set_title("MambaLRP Causal Attribution Heatmap")
    except Exception as e:
        logger.warning(f"Could not compute full sequence attribution for plot: {e}")
        axes[2].text(0.5, 0.5, f"Attribution Plot Failed: {e}", ha='center', va='center', transform=axes[2].transAxes)
        
    axes[2].set_xlabel("Time Step (Frames)")
    axes[2].set_ylabel("Channels")

    plt.tight_layout()
    png_path = f"output/harness/{args.png_name}"
    plt.savefig(png_path, dpi=300)
    logger.info(f"Dashboard saved to {png_path}")

if __name__ == "__main__":
    main()
