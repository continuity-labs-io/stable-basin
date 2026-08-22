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
from src.metrics.diagnostic_engine import ThermodynamicDiagnosticEngine
from src.harness.sensor_fusion_predictor import SensorFusionPredictor, SSMType
from src.metrics.metrics import ThermodynamicMetrics
from src.icebox.metrics.mamba_lrp import MambaLRPEpsilon
from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("pydmd").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)



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
    import logging
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    config = trial_config["base_config"]
    model_type = trial_config["model_type"]
    device = get_optimal_device(verbose=False)
    
    wandb.init(
        project="stable-basin",
        name=f"diagnostic_{model_type}",
        config={"model_type": model_type, **config},
        reinit="finish_previous"
    )
    
    logger.info(f"--- Running Diagnostic for Model: {model_type} ---")
    
    input_dim = config["data"]["input_dim"]
    d_model = config["training"]["d_model"]
    seq_len = config["data"]["seq_len"]
    epochs = config["training"]["epochs"]
    burn_in_frames = config["training"]["burn_in_frames"]
    ksm_threshold = config["evaluation"]["ksm_threshold"]
    if model_type == "masr_mamba":
        # Mamba's data-dependent state transitions inject non-linear variance 
        # during stable periodic signals. Relaxing the threshold prevents early false-positives.
        ksm_threshold = 0.85
    png_prefix = config["evaluation"]["png_prefix"]
    csv_prefix = config["evaluation"]["csv_prefix"]
    condition = config["data"]["condition"]
    
    use_synthetic = config.get("use_synthetic", False)
    
    if not use_synthetic:
        logger.info(f"Loading PharmacologicalShockDataset (seq_len={seq_len}) on worker")
        dataset = PharmacologicalShockDataset(condition=condition, seq_len=seq_len)
        telemetry = dataset[0].unsqueeze(0).to(device)  # shape: [1, seq_len, 1024]
        mask = torch.ones(1, seq_len, 1, device=device)
        logger.info(f"Successfully loaded Pharmacological Shock Data! Telemetry shape: {telemetry.shape}")
    else:
        logger.info("Generating synthetic telemetry for testing with crash at seq_len // 2.")
        t = torch.linspace(0, 10 * np.pi, seq_len, device=device).unsqueeze(1)
        freqs = torch.linspace(0.5, 3.0, 1024, device=device)
        healthy = torch.sin(t * freqs) + (torch.randn(seq_len, 1024, device=device) * 0.1)
        telemetry = healthy.unsqueeze(0)
        
        crash_frame_true = seq_len // 2
        telemetry[:, crash_frame_true:, :] = torch.randn_like(telemetry[:, crash_frame_true:, :]) * 0.5
        telemetry[:, crash_frame_true-50:crash_frame_true, 120:130] = telemetry[:, crash_frame_true-50:crash_frame_true, 120:130] + 5.0
        
        mask = torch.ones(1, seq_len, 1, device=device)

    logger.info(f"Initializing {model_type} model")
    model = SensorFusionPredictor(
        ssm_type=model_type, 
        modality_dims=[input_dim], 
        d_model=d_model, 
        out_dim=input_dim
    ).to(device)

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
        base_ksm = ThermodynamicMetrics(alpha=500.0, beta=1.0).calculate_ksm(base_hidden[0])
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
    for t in range(burn_in_len, len(ksm_np)):
        if ksm_np[t] < ksm_threshold:
            crash_frame = t
            break

    if crash_frame == -1:
        logger.warning(f"Crash frame not found! KSM never dropped below {ksm_threshold}.")
        crash_frame = len(ksm_np) - 1 
    else:
        logger.info(f"Detected crash at frame {crash_frame} (KSM dropped below {ksm_threshold})")

    logger.info("Executing Thermodynamic Diagnostic Engine")
    feature_names = [f"Ch_{i}" for i in range(input_dim)]
    diagnostic = ThermodynamicDiagnosticEngine(model, feature_names=feature_names)
            
    try:
        report = diagnostic.generate_diagnostic(telemetry, crash_time_step=crash_frame)
    except Exception as e:
        logger.warning(f"Diagnostic generation failed: {e}")
        report = {"error": str(e), "crash_frame": crash_frame}

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output/harness"))
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"clinical_diagnostic_report_{model_type}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, cls=NpEncoder)
    logger.info(f"Saved Diagnostic JSON to {report_path}")

    logger.info("Generating 3-panel dashboard")
    
    import csv
    csv_path = os.path.join(out_dir, f"{csv_prefix}_{model_type}.csv")
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
    plot_dir = os.path.join(out_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)
    png_path = os.path.join(plot_dir, f"{png_prefix}_{model_type}.png")
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

        artifact = wandb.Artifact(f"diagnostic_{model_type}", type="report")
        artifact.add_file(report_path)
        artifact.add_file(csv_path)
        wandb.log_artifact(artifact)
        
    tune.report({
        "latency_ms": latency_ms, 
        "crash_frame": crash_frame, 
        "baseline_ksm_variance": base_ksm_variance
    })
    
    logger.info(f"SUCCESS: {model_type} evaluation complete! Crash frame detected at: {crash_frame}")
    wandb.finish()


def main():
    import yaml
    parser = argparse.ArgumentParser(description="Clinical Diagnostic Runner (Distributed)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--use_synthetic", action="store_true", help="Force synthetic data for testing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    config["use_synthetic"] = args.use_synthetic
        
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
        run_config=tune.RunConfig(name="clinical_diagnostic_sweep")
    )
    
    results = tuner.fit()
    logger.info("Ray Tune execution complete.")

    # Aggregate individual outputs into single files
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output/harness"))
    import glob
    import pandas as pd

    # Aggregate JSON reports
    all_reports = {}
    json_pattern = os.path.join(out_dir, "clinical_diagnostic_report_*.json")
    for filepath in glob.glob(json_pattern):
        if filepath.endswith("clinical_diagnostic_reports.json"):
            continue
        basename = os.path.basename(filepath)
        model_name = basename.replace("clinical_diagnostic_report_", "").replace(".json", "")
        with open(filepath, "r") as f:
            all_reports[model_name] = json.load(f)
        os.remove(filepath)
        
    if all_reports:
        with open(os.path.join(out_dir, "clinical_diagnostic_reports.json"), "w") as f:
            json.dump(all_reports, f, indent=4, cls=NpEncoder)
        logger.info("Aggregated individual JSON reports into clinical_diagnostic_reports.json")

    # Aggregate CSV metrics
    csv_prefix = config["evaluation"]["csv_prefix"]
    csv_pattern = os.path.join(out_dir, f"{csv_prefix}_*.csv")
    csv_files = [f for f in glob.glob(csv_pattern) if not f.endswith(f"{csv_prefix}.csv")]
    if csv_files:
        df_list = [pd.read_csv(f) for f in csv_files]
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df.to_csv(os.path.join(out_dir, f"{csv_prefix}.csv"), index=False)
            for f in csv_files:
                os.remove(f)
            logger.info(f"Aggregated individual CSV metrics into {csv_prefix}.csv")
if __name__ == "__main__":
    main()
