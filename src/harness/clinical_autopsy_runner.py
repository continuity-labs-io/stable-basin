import os
import time
import json
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import ray
from ray import tune, train
import wandb

from src.data.ephys.pharma_shock_dataset import PharmacologicalShockDataset
from src.models.attention.baseline_transformer import BaselineTransformer
from src.metrics.autopsy_engine import ThermodynamicAutopsyEngine
from src.models.ssm.baseline_ssm import BaselineSSM
from src.models.ssm.mask_aware_ssm import MaskAwareSSM
from src.models.ssm.mask_aware_mamba import MaskAwareMamba
from src.metrics.metrics import ThermodynamicMetrics
from src.metrics.mamba_lrp import MambaLRPEpsilon
from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("pydmd").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class ModelAdapter(nn.Module):
    """
    Adapter to give BaselineSSM, BaselineTransformer, and MaskAwareSSM
    the same interface as MaskAwareMamba (input projection, forward forecasting head, etc).
    """
    def __init__(self, core_model, input_dim=1024, d_model=256, is_mask_aware=False):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.core = core_model
        self.forward_head = nn.Linear(d_model, input_dim)
        self.is_mask_aware = is_mask_aware
        if self.is_mask_aware:
            self.gate_proj = nn.Linear(input_dim, d_model)

    def forward(self, x, mask=None, return_hidden=False):
        h = self.input_proj(x)
        if self.is_mask_aware:
            if mask is None:
                mask = torch.ones_like(x)
            latent_gate = torch.sigmoid(self.gate_proj(mask))
            hidden_states = self.core(h, latent_gate)
        else:
            hidden_states = self.core(h)
        pred_t_plus_1 = F.softplus(self.forward_head(hidden_states))
        
        reconstructed_t = pred_t_plus_1 
        
        if return_hidden:
            return pred_t_plus_1, reconstructed_t, hidden_states
        return pred_t_plus_1, reconstructed_t

    def get_hidden_states(self, x):
        _, _, hidden_states = self.forward(x, return_hidden=True)
        return hidden_states


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def evaluate_model(trial_config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    config = trial_config["base_config"]
    model_type = trial_config["model_type"]
    device = get_optimal_device(verbose=False)
    
    wandb.init(
        project="stable-basin",
        name=f"autopsy_{model_type}",
        config={"model_type": model_type, **config},
        reinit=True
    )
    
    logger.info(f"--- Running Autopsy for Model: {model_type} ---")
    
    input_dim = config["data"]["input_dim"]
    d_model = config["training"]["d_model"]
    seq_len = config["data"]["seq_len"]
    epochs = config["training"]["epochs"]
    burn_in_frames = config["training"]["burn_in_frames"]
    ksm_threshold = config["evaluation"]["ksm_threshold"]
    png_prefix = config["evaluation"]["png_prefix"]
    csv_prefix = config["evaluation"]["csv_prefix"]
    condition = config["data"]["condition"]
    
    logger.info(f"Loading PharmacologicalShockDataset (seq_len={seq_len}) on worker")
    try:
        dataset = PharmacologicalShockDataset(condition=condition, seq_len=seq_len)
        telemetry = dataset[0].unsqueeze(0).to(device)  # shape: [1, seq_len, 1024]
        mask = torch.ones_like(telemetry)
    except FileNotFoundError:
        logger.warning("Dataset file not found. Generating dummy telemetry for testing.")
        t = torch.linspace(0, 10 * np.pi, seq_len, device=device).unsqueeze(1)
        freqs = torch.linspace(0.5, 3.0, 1024, device=device)
        healthy = torch.sin(t * freqs) + (torch.randn(seq_len, 1024, device=device) * 0.1)
        telemetry = healthy.unsqueeze(0)
        
        crash_frame_true = seq_len // 2
        telemetry[:, crash_frame_true:, :] = torch.randn_like(telemetry[:, crash_frame_true:, :]) * 0.5
        telemetry[:, crash_frame_true-50:crash_frame_true, 120:130] = telemetry[:, crash_frame_true-50:crash_frame_true, 120:130] + 5.0
        
        mask = (torch.rand_like(telemetry) > 0.90).float()
        telemetry = telemetry * mask

    logger.info(f"Initializing {model_type} model")
    if model_type == "meld":
        model = MaskAwareMamba(input_dim=input_dim, d_model=d_model, mask_aware=False).to(device)
    elif model_type == "baseline":
        core = BaselineSSM(d_model=d_model)
        model = ModelAdapter(core, input_dim=input_dim, d_model=d_model).to(device)
    elif model_type == "transformer":
        core = BaselineTransformer(d_model=d_model)
        model = ModelAdapter(core, input_dim=input_dim, d_model=d_model).to(device)
    elif model_type == "mask_aware":
        core = MaskAwareSSM(d_model=d_model)
        model = ModelAdapter(core, input_dim=input_dim, d_model=d_model, is_mask_aware=True).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    lrp_engine = MambaLRPEpsilon(model)
    from src.metrics.attribution_engine import AttributionEngine
    AttributionEngine.get_instance().set_strategy(lambda m, x, t: lrp_engine.attribute(x, t))

    burn_in_len = min(burn_in_frames, seq_len // 4)
    x_train = telemetry[:, :burn_in_len, :]
    mask_train = mask[:, :burn_in_len, :]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    from src.harness.trainer import StableBasinTrainer
    trainer = StableBasinTrainer(model, optimizer, device, loss_type="residual_mse")
    
    logger.info(f"Starting Burn-In Training for {epochs} epochs on first {burn_in_len} frames")
    dataloader = [{"x_raw": x_train, "mask": mask_train}]
    trainer.fit(dataloader, epochs, use_wandb=True)

    model.eval()
    with torch.no_grad():
        _, base_hidden = model(x_train, mask=mask_train)
        base_ksm = ThermodynamicMetrics(alpha=500.0).calculate_ksm(base_hidden[0])
        base_ksm_variance = torch.var(torch.tensor(base_ksm, dtype=torch.float32)).item()
        logger.info(f"Baseline Thermodynamic Stability (KSM Variance): {base_ksm_variance:.6e}")

    logger.info("Running dynamic crash detection over full sequence")
    start_time = time.time()
    with torch.no_grad():
        pred_full, full_hidden = model(telemetry, mask=mask)
    inference_time = time.time() - start_time
    latency_ms = (inference_time / seq_len) * 1000
    logger.info(f"Inference Latency: {latency_ms:.2f} ms/frame")

    ksm_trajectory = ThermodynamicMetrics(alpha=500.0).calculate_ksm(full_hidden[0])
    ksm_np = np.array(ksm_trajectory)
    
    crash_frame = -1
    for t in range(len(ksm_np)):
        if ksm_np[t] < ksm_threshold:
            crash_frame = t
            break

    if crash_frame == -1:
        logger.warning(f"Crash frame not found! KSM never dropped below {ksm_threshold}.")
        crash_frame = len(ksm_np) - 1 
    else:
        logger.info(f"Detected crash at frame {crash_frame} (KSM dropped below {ksm_threshold})")

    logger.info("Executing Thermodynamic Autopsy Engine")
    feature_names = [f"Ch_{i}" for i in range(input_dim)]
    autopsy = ThermodynamicAutopsyEngine(model, feature_names=feature_names)
            
    try:
        report = autopsy.generate_autopsy(telemetry, crash_time_step=crash_frame)
    except Exception as e:
        logger.warning(f"Autopsy generation failed: {e}")
        report = {"error": str(e), "crash_frame": crash_frame}

    os.makedirs("output/harness", exist_ok=True)
    report_path = f"output/harness/clinical_autopsy_report_{model_type}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, cls=NpEncoder)
    logger.info(f"Saved Autopsy JSON to {report_path}")

    logger.info("Generating 3-panel dashboard")
    
    import csv
    csv_path = f"output/harness/{csv_prefix}_{model_type}.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "baseline_ksm_variance", "crash_frame", "inference_latency_ms_per_frame"])
        writer.writerow([model_type, base_ksm_variance, crash_frame, latency_ms])
    logger.info(f"Saved summary metrics to {csv_path}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    raw_np = telemetry.squeeze().cpu().numpy() # [seq_len, input_dim]
    pool_factor = max(1, raw_np.shape[1] // 128)
    raw_pooled = raw_np[:, ::pool_factor].T # [128, seq_len]

    axes[0].imshow(raw_pooled, aspect='auto', cmap='viridis', origin='lower')
    axes[0].axvline(x=crash_frame, color='red', linestyle='--', label='Crash Frame')
    axes[0].set_title(f"{model_type.upper()} - {input_dim}-Ch Telemetry (MaxPool 128)")
    axes[0].set_ylabel("Channels")
    axes[0].legend()

    axes[1].plot(ksm_np, color='orange')
    axes[1].axhline(y=ksm_threshold, color='red', linestyle=':', label='Threshold')
    axes[1].axvline(x=crash_frame, color='red', linestyle='--')
    axes[1].set_title("Koopman Stability Metric (KSM)")
    axes[1].set_ylabel("KSM")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()

    logger.info("Extracting full attribution sequence for Panel 3 plotting...")
    try:
        relevance_tensor = AttributionEngine.get_instance().compute_attribution(model, telemetry, target_time_step=crash_frame)
        rel_np = relevance_tensor.squeeze().cpu().numpy() 
        rel_pooled = rel_np[:, ::pool_factor].T 
        
        axes[2].imshow(rel_pooled, aspect='auto', cmap='magma', origin='lower')
        axes[2].axvline(x=crash_frame, color='white', linestyle='--')
        axes[2].set_title("MambaLRP Causal Attribution Heatmap")
    except Exception as e:
        logger.warning(f"Could not compute full sequence attribution for plot: {e}")
        axes[2].text(0.5, 0.5, f"Attribution Plot Failed: {e}", ha='center', va='center', transform=axes[2].transAxes)
        
    axes[2].set_xlabel("Time Step (Frames)")
    axes[2].set_ylabel("Channels")

    plt.tight_layout()
    png_path = f"output/harness/{png_prefix}_{model_type}.png"
    plt.savefig(png_path, dpi=300)
    logger.info(f"Dashboard saved to {png_path}")
    plt.close(fig)
    
    if wandb.run is not None:
        wandb.log({
            "inference_latency_ms": latency_ms,
            "baseline_ksm_variance": base_ksm_variance,
            "crash_frame": crash_frame,
            "dashboard": wandb.Image(png_path)
        })

        artifact = wandb.Artifact(f"autopsy_{model_type}", type="report")
        artifact.add_file(report_path)
        artifact.add_file(csv_path)
        wandb.log_artifact(artifact)
        
    tune.report({
        "latency_ms": latency_ms, 
        "crash_frame": crash_frame, 
        "baseline_ksm_variance": base_ksm_variance
    })
    
    wandb.finish()


def main():
    import yaml
    parser = argparse.ArgumentParser(description="Clinical Autopsy Runner (Distributed)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    search_space = {
        "base_config": config,
        "model_type": tune.grid_search(config["models"])
    }
    
    ray.init(ignore_reinit_error=True)
    
    tuner = tune.Tuner(
        tune.with_resources(
            evaluate_model, 
            resources={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0}
        ),
        param_space=search_space,
        run_config=tune.RunConfig(name="clinical_autopsy_sweep")
    )
    
    results = tuner.fit()
    logger.info("Ray Tune execution complete.")

if __name__ == "__main__":
    main()
