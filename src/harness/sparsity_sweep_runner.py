import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
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
    device = get_optimal_device(verbose=True)
    sparsities = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    seeds = [42, 100, 256, 512, 1024]
    
    models = ["baseline", "forward_fill", "mask_concat", "gru_d", "ode_rnn", "mask_aware"]
    results = {m: {s: [] for s in sparsities} for m in models}
    
    logger.info("Commencing 5-Seed Sparsity Sweep (FitzHugh-Nagumo Oscillator)...")
    logger.info(f"Sweep parameters: {len(sparsities)} sparsities | {len(seeds)} seeds | {len(models)} models = {len(sparsities)*len(seeds)*len(models)} training runs")
    
    for sparsity in sparsities:
        for seed in seeds:
            for m in models:
                mse = run_experiment(device, m, sparsity, seed)
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
    
    os.makedirs("output/data", exist_ok=True)
    plt.savefig("output/data/05_sparsity_sweep.png", dpi=300)
    logger.info("Dashboard saved to output/data/05_sparsity_sweep.png")

if __name__ == "__main__":
    main()
