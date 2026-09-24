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
from src.metrics.spectral import SpectralMetrics
from src.data.behavior.synthetic_aging import amplitude_residual_stats
from src.metrics.baseline_statistics import compute_stats

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
    
    time_metrics = None # no longer using ThermodynamicMetrics
    spec_metrics = SpectralMetrics()

    def evaluate_cohort(dataset):
        ar1_list = []
        var_list = []
        peak_freq_list = []
        for traj in dataset.data:
            # Calculate Amplitude Residual Stats (CSD proxy)
            stats = amplitude_residual_stats(traj.numpy(), pair=(0, 1))
            ar1_list.append(stats["amp_ar1"])
            var_list.append(stats["amp_var"])
            
            # Calculate Peak Frequency (25.0 Hz biological framerate)
            freq, power = spec_metrics.calculate_psd(traj, sampling_rate=25.0)
            peak_idx = torch.argmax(power.mean(dim=1)) if len(power.shape) > 1 else 0
            peak_freq_list.append(float(freq[peak_idx]))
            
        return ar1_list, var_list, peak_freq_list

    logger.info("Evaluating Clean Baseline Cohort...")
    ar1_y, var_y, freq_y = evaluate_cohort(ds_young)
    
    logger.info("Evaluating Synthetically Degraded Cohort...")
    ar1_o, var_o, freq_o = evaluate_cohort(ds_old)

    logger.info("\n--- 1. TIME DOMAIN METRICS (Amplitude CSD) ---")
    
    mean_ar1_young = float(np.mean(ar1_y))
    std_ar1_young = float(np.std(ar1_y))
    mean_ar1_old = float(np.mean(ar1_o))
    std_ar1_old = float(np.std(ar1_o))
    
    mean_var_young = float(np.mean(var_y))
    std_var_young = float(np.std(var_y))
    mean_var_old = float(np.mean(var_o))
    std_var_old = float(np.std(var_o))
    
    results["time_domain"]["mean_ar1_baseline"] = mean_ar1_young
    results["time_domain"]["std_ar1_baseline"] = std_ar1_young
    results["time_domain"]["mean_ar1_degraded"] = mean_ar1_old
    results["time_domain"]["std_ar1_degraded"] = std_ar1_old
    
    results["time_domain"]["mean_var_baseline"] = mean_var_young
    results["time_domain"]["std_var_baseline"] = std_var_young
    results["time_domain"]["mean_var_degraded"] = mean_var_old
    results["time_domain"]["std_var_degraded"] = std_var_old
    
    ar1_stats = compute_stats(ar1_y, ar1_o)
    for k, v in ar1_stats.items():
        results["time_domain"][f"ar1_{k}"] = v

    var_stats = compute_stats(var_y, var_o)
    for k, v in var_stats.items():
        results["time_domain"][f"var_{k}"] = v
    
    logger.info(f"Clean Baseline AR(1): {mean_ar1_young:.4f} ± {std_ar1_young:.4f}")
    logger.info(f"Synthetically Degraded AR(1): {mean_ar1_old:.4f} ± {std_ar1_old:.4f}")
    logger.info(f"Clean Baseline Variance: {mean_var_young:.4f} ± {std_var_young:.4f}")
    logger.info(f"Synthetically Degraded Variance: {mean_var_old:.4f} ± {std_var_old:.4f}")

    logger.info("\n--- 2. SPECTRAL METRICS (Peak Frequency) ---")
    
    mean_freq_young = float(np.mean(freq_y))
    std_freq_young = float(np.std(freq_y))
    mean_freq_old = float(np.mean(freq_o))
    std_freq_old = float(np.std(freq_o))
    
    results["spectral"]["mean_peak_freq_baseline"] = mean_freq_young
    results["spectral"]["std_peak_freq_baseline"] = std_freq_young
    results["spectral"]["mean_peak_freq_degraded"] = mean_freq_old
    results["spectral"]["std_peak_freq_degraded"] = std_freq_old
    
    freq_stats = compute_stats(freq_y, freq_o)
    for k, v in freq_stats.items():
        results["spectral"][f"peak_freq_{k}"] = v

    
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
