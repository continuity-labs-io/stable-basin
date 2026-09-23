import json
import wandb
import torch
import numpy as np
import logging
import os
import warnings

# Suppress warnings from scipy/pytorch during evaluation for clean console output
warnings.filterwarnings("ignore")

from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset
from src.metrics.time_domain import ThermodynamicMetrics
from src.metrics.spectral import SpectralMetrics
from src.metrics.entropy_production import entropy_production_mou

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_baseline_metrics():
    logger.info("   Baseline Biological Metrics Evaluation     ")
    
    logger.info("Loading C. elegans biological datasets...")
    try:
        # Load the test dataset for both young and old populations
        ds_young = RealEigenwormDataset("data/worm/EigenWorms_TEST.ts", seq_len=1000, is_aged=False)
        ds_old = RealEigenwormDataset("data/worm/EigenWorms_TEST.ts", seq_len=1000, is_aged=True)
    except FileNotFoundError:
        logger.error("Biological data not found. Please ensure data is present in 'data/worm/'.")
        return

    # Use the first full trajectory from the dataset for evaluation
    traj_young = ds_young.data[0] # shape [seq_len, dim]
    traj_old = ds_old.data[0]

    results = {
        "time_domain": {},
        "spectral": {},
        "entropy": {}
    }

    logger.info("\n--- 1. TIME DOMAIN METRICS ---")
    logger.info("Evaluating Critical Slowing Down (Variance / Autocorrelation over time)")
    time_metrics = ThermodynamicMetrics(alpha=1.0, beta=1.0)
    
    # Calculate CSD over a sliding window
    csd_young = time_metrics.calculate_csd(traj_young, window_size=50)
    csd_old = time_metrics.calculate_csd(traj_old, window_size=50)
    
    mean_csd_young = float(np.mean(csd_young))
    mean_csd_old = float(np.mean(csd_old))
    
    results["time_domain"]["mean_csd_young"] = mean_csd_young
    results["time_domain"]["mean_csd_old"] = mean_csd_old
    
    logger.info(f"Young CSD Mean: {mean_csd_young:.4f}")
    logger.info(f"Old CSD Mean:   {mean_csd_old:.4f}")

    logger.info("\n--- 2. SPECTRAL METRICS ---")
    logger.info("Evaluating Power Spectral Density (Frequency structure)")
    spec_metrics = SpectralMetrics()
    
    freq_y, power_y = spec_metrics.calculate_psd(traj_young, sampling_rate=16.0)
    freq_o, power_o = spec_metrics.calculate_psd(traj_old, sampling_rate=16.0)
    
    # Compute peak frequency for across all dimensions
    peak_idx_y = torch.argmax(power_y.mean(dim=1)) if len(power_y.shape) > 1 else 0
    peak_idx_o = torch.argmax(power_o.mean(dim=1)) if len(power_o.shape) > 1 else 0
    
    peak_freq_y = float(freq_y[peak_idx_y])
    peak_freq_o = float(freq_o[peak_idx_o])
    
    results["spectral"]["peak_freq_young"] = peak_freq_y
    results["spectral"]["peak_freq_old"] = peak_freq_o
    
    logger.info(f"Young Peak Frequency: {peak_freq_y:.4f} Hz")
    logger.info(f"Old Peak Frequency:   {peak_freq_o:.4f} Hz")

    logger.info("\n--- 3. ENTROPY METRICS ---")
    logger.info("Evaluating Thermodynamic Irreversibility (MOU Process Entropy Production)")
    
    traj_young_np = traj_young.numpy()
    traj_old_np = traj_old.numpy()
    
    mou_y = entropy_production_mou(traj_young_np, fs=16.0, lag_samples=1)
    mou_o = entropy_production_mou(traj_old_np, fs=16.0, lag_samples=1)
    
    results["entropy"]["entropy_prod_young"] = float(mou_y.phi)
    results["entropy"]["entropy_prod_old"] = float(mou_o.phi)
    
    logger.info(f"Young Entropy Production: {mou_y.phi:.4f}")
    logger.info(f"Old Entropy Production:   {mou_o.phi:.4f}")

    # Save results
    out_dir = "output/benchmarks/worm_gait"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "01_worm_gait_baseline_metrics.json")
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    wandb.init(project="worm_gait", name="01_baseline_metrics", config=results)
    
    # Flatten results for logging
    flat_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_results[f"{k}/{sub_k}"] = sub_v
        else:
            flat_results[k] = v
    wandb.log(flat_results)
    
    # Save the json as an artifact
    artifact = wandb.Artifact("baseline_metrics_json", type="metrics")
    artifact.add_file(out_file)
    wandb.log_artifact(artifact)
    wandb.finish()
        
    logger.info(f"\n[DONE] Successfully serialized all baseline metrics to: {out_file} and logged to wandb")

if __name__ == "__main__":
    run_baseline_metrics()
