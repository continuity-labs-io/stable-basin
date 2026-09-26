import json
import wandb
import torch
import numpy as np
import logging
import os
import warnings
import yaml

# Suppress warnings from scipy/pytorch during evaluation for clean console output
warnings.filterwarnings("ignore")

from src.metrics.baseline_statistics import compute_stats
from src.benchmarks.aging_resilience.task_registry import get_benchmark_task

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

    
def run_baseline_metrics():
    logger.info("   Baseline Biological Metrics Evaluation     ")
    
    config_path = "configs/worm_gait_experiments.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    task = get_benchmark_task(config)
    
    logger.info("Loading C. elegans biological datasets via task...")
    _, ds_young, ds_old = task.get_raw_datasets(config)

    def evaluate_cohort(dataset):
        aggregated = {}
        for traj in dataset:
            metrics = task.compute_domain_metrics(traj.numpy() if hasattr(traj, "numpy") else traj)
            for k, v in metrics.items():
                if k not in aggregated:
                    aggregated[k] = []
                aggregated[k].append(v)
        return aggregated

    logger.info("Evaluating Clean Baseline Cohort...")
    metrics_young = evaluate_cohort(ds_young)
    
    logger.info("Evaluating Synthetically Degraded Cohort...")
    metrics_old = evaluate_cohort(ds_old)

    logger.info("\n--- METRICS COMPARISON ---")
    results = {}
    
    for key in metrics_young.keys():
        val_y = metrics_young[key]
        val_o = metrics_old.get(key, [])
        if not val_o:
            continue
            
        mean_y = float(np.mean(val_y))
        std_y = float(np.std(val_y))
        mean_o = float(np.mean(val_o))
        std_o = float(np.std(val_o))
        
        results[key] = {
            "mean_baseline": mean_y,
            "std_baseline": std_y,
            "mean_degraded": mean_o,
            "std_degraded": std_o
        }
        
        stats = compute_stats(val_y, val_o)
        for k, v in stats.items():
            results[key][k] = v
            
        logger.info(f"Clean Baseline {key}: {mean_y:.4f} ± {std_y:.4f}")
        logger.info(f"Synthetically Degraded {key}: {mean_o:.4f} ± {std_o:.4f}")

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
