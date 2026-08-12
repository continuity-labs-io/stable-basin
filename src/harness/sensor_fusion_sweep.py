import os
import time
import argparse
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ray
from ray import tune, train
import wandb

from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.harness.sensor_fusion_predictor import SensorFusionPredictor, SSMType
from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DatasetWrapper(Dataset):
    def __init__(self, size: int, seq_len: int = 500, density: float = 0.1, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.dataset = SyntheticWaddingtonDataset(size=size, seq_len=seq_len, density=density)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def evaluate_model(trial_config):
    import logging
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)
    import matplotlib
    matplotlib.use('Agg')
    
    config = trial_config["base_config"]
    model_type = trial_config["model_type"]
    density = trial_config["density"]
    seed = trial_config["seed"]
    
    epochs = config["epochs"]
    train_seq_len = config["train_seq_len"]
    test_seq_len = config["test_seq_len"]
    task_name = trial_config["task_name"]
    
    device = get_optimal_device(verbose=False)
    
    # Initialize WandB
    wandb.init(
        project="stable-basin",
        name=f"{task_name}_{model_type}_d{density}_s{seed}",
        config={"model_type": model_type, "density": density, "seed": seed, **config},
        reinit="finish_previous"
    )
    
    logger.info(f"--- Running {task_name} | Model: {model_type} | Density: {density} | Seed: {seed} ---")
    
    # Seed globally for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Data Preparation
    train_dataset = DatasetWrapper(size=100, seq_len=train_seq_len, density=density, seed=seed)
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    model = SensorFusionPredictor(model_type).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for b_idx, batch in enumerate(dataloader):
            x_raw = batch["x_raw"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["y_true"].to(device)
            
            optimizer.zero_grad()
            preds, _ = model(x_raw, mask)
            loss = criterion(preds, y_true)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            wandb.log({"epoch": epoch, "batch": b_idx, "train_loss": loss.item()})
            
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        logger.info(f"Epoch {epoch:02d}/{epochs} - {model_type} - Loss: {avg_loss:.4f}")

    # Eval loop (Out-of-Distribution / Extrapolation)
    model.eval()
    
    # Use standard SyntheticWaddingtonDataset for eval
    torch.manual_seed(seed + 1)
    np.random.seed(seed + 1)
    ood_dataset = SyntheticWaddingtonDataset(size=50, seq_len=test_seq_len, density=density)
    test_loader = DataLoader(ood_dataset, batch_size=8, shuffle=False)
    
    total_mse = 0.0
    count = 0
    first_preds = None
    first_y_true = None
    
    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            x_raw = batch["x_raw"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["y_true"].to(device)
            
            preds, _ = model(x_raw, mask)
            
            if test_seq_len > train_seq_len:
                mse = ((preds[:, train_seq_len:] - y_true[:, train_seq_len:]) ** 2).mean().item()
            else:
                mse = criterion(preds, y_true).item()
                
            total_mse += mse * x_raw.size(0)
            count += x_raw.size(0)
            
            if b_idx == 0:
                first_preds = preds[0].cpu().numpy().tolist()
                first_y_true = y_true[0].cpu().numpy().tolist()
                
    final_mse = total_mse / count
    logger.info(f"✅ SUCCESS: {model_type} evaluation complete! Final MSE: {final_mse:.4f}")
    
    wandb.log({"final_mse": final_mse})
    
    tune.report({
        "mse": final_mse,
        "loss_history": loss_history,
        "preds": first_preds,
        "y_true": first_y_true
    })
    
    wandb.finish()

def save_benchmark_plot(df_results, config, task_name, png_name):
    import matplotlib.pyplot as plt
    train_seq_len = config["train_seq_len"]
    test_seq_len = config["test_seq_len"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    colors = {
        "baseline": "r--", "forward_fill": "m-.", "mask_concat": "y-",
        "transformer": "g:", "mask_aware": "b-", "gru_d": "c-", "ode_rnn": "m-"
    }
    labels = {
        "baseline": "Zero-Padded SSM", "forward_fill": "Forward-Fill SSM",
        "mask_concat": "Mask-Concat SSM", "transformer": "Causal Transformer",
        "mask_aware": "MASR (Ours)", "gru_d": "GRU-D", "ode_rnn": "ODE-RNN"
    }

    y_true_plotted = False
    for _, row in df_results.iterrows():
        name = row["config/model_type"]
        loss_hist = row["loss_history"]
        preds = row["preds"]
        y_true = row["y_true"]
        
        ax1.plot(loss_hist, colors.get(name, "k-"), linewidth=2, label=labels.get(name, name))
        
        if not y_true_plotted:
            ax2.plot(y_true, "k-", linewidth=4, label="True Phase")
            y_true_plotted = True
            
        linewidth = 2.5 if name == "mask_aware" else 1.5
        alpha = 1.0 if name == "mask_aware" else 0.8
        ax2.plot(preds, colors.get(name, "k-"), linewidth=linewidth, alpha=alpha, label=labels.get(name, name))

    ax1.set_title(f"MSE Loss Convergence (Training on seq_len={train_seq_len})")
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.axvline(x=train_seq_len, color="grey", linestyle="--", linewidth=2)
    ax2.text(train_seq_len + 10, ax2.get_ylim()[1] * 0.9, "Training Horizon", color="grey", fontsize=10)
    ax2.set_title(f"Waddington Phase Tracking: {task_name.capitalize()} (seq_len={test_seq_len})")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Phase State")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs("output/harness", exist_ok=True)
    out_path = f"output/harness/{png_name}"
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved plot to {out_path}")
    plt.close(fig)

def save_density_plot(df_results, config, png_name):
    import matplotlib.pyplot as plt
    densities = sorted(df_results["config/density"].unique().tolist(), reverse=True)
    models = df_results["config/model_type"].unique().tolist()
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {"baseline": "red", "forward_fill": "magenta", "mask_concat": "yellow", 
              "gru_d": "cyan", "ode_rnn": "orange", "mask_aware": "lime"}
              
    for m in models:
        m_data = df_results[df_results["config/model_type"] == m]
        means = []
        stds = []
        for d in densities:
            d_scores = m_data[m_data["config/density"] == d]["mse"].values
            means.append(np.mean(d_scores))
            stds.append(np.std(d_scores))
            
        ax.plot(densities, means, marker='o', color=colors.get(m, "white"), linewidth=2.5, label=m.upper())
        ax.fill_between(densities, np.array(means) - np.array(stds), np.array(means) + np.array(stds), 
                        color=colors.get(m, "white"), alpha=0.15)
                        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_title("OOD Generalization vs. Sensor Density (5-Seed Variance)", color='white', fontweight='bold')
    ax.set_xlabel("Sensor Density (Log Scale -> Lower is Sparser)", color='white')
    ax.set_ylabel("Out-of-Distribution MSE (Log Scale)", color='white')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    os.makedirs("output/harness", exist_ok=True)
    out_path = f"output/harness/{png_name}"
    plt.savefig(out_path, dpi=300)
    logger.info(f"Dashboard saved to {out_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)
        
    if args.task not in full_config["tasks"]:
        raise ValueError(f"Task {args.task} not found in {args.config}")
        
    task_config = full_config["tasks"][args.task]
    
    ray.init(ignore_reinit_error=True)
    
    search_space = {
        "base_config": task_config,
        "task_name": args.task,
        "model_type": tune.grid_search(task_config["models"]),
        "density": tune.grid_search(task_config["densities"]),
        "seed": tune.grid_search(task_config["seeds"])
    }
    
    tuner = tune.Tuner(
        tune.with_resources(
            evaluate_model, 
            resources={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0}
        ),
        param_space=search_space,
        run_config=tune.RunConfig(name=f"sensor_fusion_sweep_{args.task}")
    )
    
    results = tuner.fit()
    logger.info(f"Ray Tune execution complete for task {args.task}.")
    
    # Aggregation
    df = results.get_dataframe()
    
    # Save CSV
    csv_path = f"output/harness/{task_config['csv_name']}"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved raw results to {csv_path}")
    
    if args.task in ["baseline", "extrapolation"]:
        save_benchmark_plot(df, task_config, args.task, task_config["png_name"])
    elif args.task == "density_sweep":
        save_density_plot(df, task_config, task_config["png_name"])

if __name__ == "__main__":
    main()
