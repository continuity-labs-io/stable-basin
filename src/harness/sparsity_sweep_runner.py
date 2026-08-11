import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.models.simulators.sensor_fusion_predictor import SensorFusionPredictor
from src.utils.device import get_optimal_device
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_experiment(device, model_name, sparsity, seed, epochs=10, train_seq_len=500, test_seq_len=2000):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Dataset generation takes time inside __getitem__, so we keep the training size small (size=100)
    train_dataset = SyntheticWaddingtonDataset(size=100, seq_len=train_seq_len, sparsity=sparsity)
    test_dataset = SyntheticWaddingtonDataset(size=50, seq_len=test_seq_len, sparsity=sparsity)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    model = SensorFusionPredictor(model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    # Train loop
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            x_raw = batch["x_raw"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["y_true"].to(device)
            
            optimizer.zero_grad()
            preds = model(x_raw, mask)
            loss = criterion(preds, y_true)
            loss.backward()
            optimizer.step()
            
    # Eval loop (Out-of-Distribution / Extrapolation)
    model.eval()
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            x_raw = batch["x_raw"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["y_true"].to(device)
            
            preds = model(x_raw, mask)
            mse = criterion(preds, y_true).item()
            total_mse += mse * x_raw.size(0)
            count += x_raw.size(0)
            
    return total_mse / count

def main():
    parser = argparse.ArgumentParser(description="Sensor Fusion Sparsity Sweep Runner")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train-seq-len", type=int, default=500, help="Training sequence length")
    parser.add_argument("--test-seq-len", type=int, default=2000, help="Testing sequence length")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.1, 0.05, 0.02, 0.01, 0.005, 0.001], help="Sparsity levels to sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 256, 512, 1024], help="Random seeds to evaluate")
    parser.add_argument("--models", type=str, nargs="+", default=["baseline", "forward_fill", "mask_concat", "gru_d", "ode_rnn", "mask_aware"], help="Models to evaluate")
    parser.add_argument("--png-name", type=str, default="05_sparsity_sweep.png", help="Filename for the output PNG")
    parser.add_argument("--csv-name", type=str, default="05_sparsity_sweep.csv", help="Filename for the output CSV")
    args = parser.parse_args()

    device = get_optimal_device(verbose=True)
    sparsities = args.sparsities
    seeds = args.seeds
    models = args.models
    
    results = {m: {s: [] for s in sparsities} for m in models}
    
    logger.info("Commencing 5-Seed Sparsity Sweep (FitzHugh-Nagumo Oscillator)...")
    logger.info(f"Sweep parameters: {len(sparsities)} sparsities | {len(seeds)} seeds | {len(models)} models = {len(sparsities)*len(seeds)*len(models)} training runs")
    
    for sparsity in sparsities:
        for seed in seeds:
            for m in models:
                mse = run_experiment(device, m, sparsity, seed, epochs=args.epochs, train_seq_len=args.train_seq_len, test_seq_len=args.test_seq_len)
                results[m][sparsity].append(mse)
                logger.info(f"Sparsity: {sparsity*100:0.1f}% | Seed: {seed:<4} | Model: {m:<15} | OOD-MSE: {mse:.4f}")
                
    # Plotting Statistical Rigor
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {"baseline": "red", "forward_fill": "magenta", "mask_concat": "yellow", 
              "gru_d": "cyan", "ode_rnn": "orange", "mask_aware": "lime"}
              
    for m in models:
        means = [np.mean(results[m][s]) for s in sparsities]
        stds = [np.std(results[m][s]) for s in sparsities]
        
        ax.plot(sparsities, means, marker='o', color=colors[m], linewidth=2.5, label=m.upper())
        ax.fill_between(sparsities, np.array(means) - np.array(stds), np.array(means) + np.array(stds), 
                        color=colors[m], alpha=0.15)
                        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()  # 10% on left down to 0.1% on right
    
    ax.set_title("OOD Generalization vs. Sensor Sparsity (5-Seed Variance)", color='white', fontweight='bold')
    ax.set_xlabel("Sensor Sparsity (Log Scale -> Lower is Sparser)", color='white')
    ax.set_ylabel("Out-of-Distribution MSE (Log Scale)", color='white')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    os.makedirs("output/harness", exist_ok=True)
    out_path = f"output/harness/{args.png_name}"
    plt.savefig(out_path, dpi=300)
    logger.info(f"Dashboard saved to {out_path}")
    
    # Save CSV
    csv_data = {"Sparsity": sparsities}
    for m in models:
        csv_data[f"{m}_mean"] = [np.mean(results[m][s]) for s in sparsities]
        csv_data[f"{m}_std"] = [np.std(results[m][s]) for s in sparsities]
        
    df = pd.DataFrame(csv_data)
    csv_out_path = f"output/harness/{args.csv_name}"
    df.to_csv(csv_out_path, index=False)
    logger.info(f"Saved CSV to {csv_out_path}")

if __name__ == "__main__":
    main()
