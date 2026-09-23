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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_baseline_metrics():
    logger.info("   Baseline Biological Metrics Evaluation     ")
    
    logger.info("Loading C. elegans biological datasets...")
    try:
        # Load the test dataset for both baseline and degraded populations
        ds_young = RealEigenwormDataset("data/worm/EigenWorms_TEST.ts", seq_len=1000, inject_synthetic_degradation=False)
        ds_old = RealEigenwormDataset("data/worm/EigenWorms_TEST.ts", seq_len=1000, inject_synthetic_degradation=True)
    except FileNotFoundError:
        logger.error("Biological data not found. Please ensure data is present in 'data/worm/'.")
        return

    results = {
        "spectral": {},
        "time_domain": {}
    }
    
    time_metrics = ThermodynamicMetrics(alpha=1.0, beta=1.0)
    spec_metrics = SpectralMetrics()

    def evaluate_cohort(dataset):
        ksm_list = []
        peak_freq_list = []
        for traj in dataset.data:
            # Calculate KSM
            ksm_scores = time_metrics.calculate_ksm(traj, window_size=50)
            ksm_list.append(np.mean(ksm_scores))
            
            # Calculate Peak Frequency (25.0 Hz biological framerate)
            freq, power = spec_metrics.calculate_psd(traj, sampling_rate=25.0)
            peak_idx = torch.argmax(power.mean(dim=1)) if len(power.shape) > 1 else 0
            peak_freq_list.append(float(freq[peak_idx]))
            
        return ksm_list, peak_freq_list

    logger.info("Evaluating Clean Baseline Cohort...")
    ksm_y, freq_y = evaluate_cohort(ds_young)
    
    logger.info("Evaluating Synthetically Degraded Cohort...")
    ksm_o, freq_o = evaluate_cohort(ds_old)

    logger.info("\n--- 1. TIME DOMAIN METRICS (KSM) ---")
    
    mean_ksm_young = float(np.mean(ksm_y))
    std_ksm_young = float(np.std(ksm_y))
    mean_ksm_old = float(np.mean(ksm_o))
    std_ksm_old = float(np.std(ksm_o))
    
    results["time_domain"]["mean_ksm_baseline"] = mean_ksm_young
    results["time_domain"]["std_ksm_baseline"] = std_ksm_young
    results["time_domain"]["mean_ksm_degraded"] = mean_ksm_old
    results["time_domain"]["std_ksm_degraded"] = std_ksm_old
    
    logger.info(f"Clean Baseline KSM: {mean_ksm_young:.4f} ± {std_ksm_young:.4f}")
    logger.info(f"Synthetically Degraded KSM: {mean_ksm_old:.4f} ± {std_ksm_old:.4f}")

    logger.info("\n--- 2. SPECTRAL METRICS (Peak Frequency) ---")
    
    mean_freq_young = float(np.mean(freq_y))
    std_freq_young = float(np.std(freq_y))
    mean_freq_old = float(np.mean(freq_o))
    std_freq_old = float(np.std(freq_o))
    
    results["spectral"]["mean_peak_freq_baseline"] = mean_freq_young
    results["spectral"]["std_peak_freq_baseline"] = std_freq_young
    results["spectral"]["mean_peak_freq_degraded"] = mean_freq_old
    results["spectral"]["std_peak_freq_degraded"] = std_freq_old
    
    logger.info(f"Clean Baseline Peak Frequency: {mean_freq_young:.4f} ± {std_freq_young:.4f} Hz")
    logger.info(f"Synthetically Degraded Peak Frequency: {mean_freq_old:.4f} ± {std_freq_old:.4f} Hz")

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
