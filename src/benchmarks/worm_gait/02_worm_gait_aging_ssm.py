import os
import json
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

from src.harness.sensor_fusion_predictor import SensorFusionPredictor
from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset

def main():
    # 1. Architecture
    model = SensorFusionPredictor(
        ssm_type="zero_padded_ssm",
        modality_dims=[6],
        d_model=64,
        out_dim=6
    )
    
    # 2. Data
    train_dataset = RealEigenwormDataset("data/worm/EigenWorms_TRAIN.ts", seq_len=100, inject_synthetic_degradation=False)
    young_eval_dataset = RealEigenwormDataset("data/worm/EigenWorms_TEST.ts", seq_len=100, inject_synthetic_degradation=False)
    old_eval_dataset = RealEigenwormDataset("data/worm/EigenWorms_TEST.ts", seq_len=100, inject_synthetic_degradation=True)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    young_eval_loader = DataLoader(young_eval_dataset, batch_size=8, shuffle=False)
    old_eval_loader = DataLoader(old_eval_dataset, batch_size=8, shuffle=False)
    
    # 3. Objective (Next-Step Forecasting)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    
    model.train()
    epochs = 50
    print("Training BaselineSSM...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            # batch is [batch_size, seq_len, 6]
            x_raw = batch
            mask = torch.ones_like(x_raw[:, :, :1])
            
            y_true = x_raw[:, 1:, :]
            preds, _, _ = model(x_raw, mask=mask)
            pred = preds[:, :-1, :]
            
            loss = mse_loss(pred, y_true)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            
    # 4. Evaluation (The Shift Test)
    model.eval()
    
    def evaluate_mse(loader):
        mses = []
        with torch.no_grad():
            for batch in loader:
                x_raw = batch
                mask = torch.ones_like(x_raw[:, :, :1])
                y_true = x_raw[:, 1:, :]
                preds, _, _ = model(x_raw, mask=mask)
                pred = preds[:, :-1, :]
                
                # Sequence-wise MSE
                # y_true shape: [batch_size, seq_len-1, 6]
                sq_diff = (pred - y_true)**2
                # average over time and feature dim
                seq_mse = sq_diff.mean(dim=(1, 2))
                mses.extend(seq_mse.tolist())
        return np.array(mses)
        
    print("Evaluating Young and Old datasets...")
    young_mses = evaluate_mse(young_eval_loader)
    old_mses = evaluate_mse(old_eval_loader)
    
    # 5. Statistical Output
    ks_stat, p_val = scipy.stats.ks_2samp(young_mses, old_mses)
    cohens_d = pg.compute_effsize(young_mses, old_mses, eftype='cohen')
    
    print(f"KS-statistic: {ks_stat:.4f}")
    print(f"p-value: {p_val:.4e}")
    print(f"Cohen's d: {cohens_d:.4f}")
    
    # 6. Serialization & Visualization
    out_dir = "output/benchmarks/worm_gait"
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "young_mse_mean": float(np.mean(young_mses)),
        "old_mse_mean": float(np.mean(old_mses)),
        "young_mse_std": float(np.std(young_mses)),
        "old_mse_std": float(np.std(old_mses))
    }
    
    wandb.init(project="worm_gait", name="02_aging_ssm", config=metrics)
    wandb.log(metrics)
    
    with open(os.path.join(out_dir, "02_baseline_ssm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    plt.figure(figsize=(8, 6))
    sns.kdeplot(young_mses, label="Young Eval", fill=True, alpha=0.5)
    sns.kdeplot(old_mses, label="Old Eval", fill=True, alpha=0.5)
    plt.title("BaselineSSM MSE Distributions (Young vs Old)")
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_baseline_ssm_mse.png"))
    wandb.log({"02_baseline_ssm_mse": wandb.Image(os.path.join(out_dir, "02_baseline_ssm_mse.png"))})
    plt.close()
    
    artifact = wandb.Artifact("02_baseline_ssm_metrics", type="metrics")
    artifact.add_file(os.path.join(out_dir, "02_baseline_ssm_metrics.json"))
    wandb.log_artifact(artifact)
    wandb.finish()
    
    print("Saved metrics and plot, and logged to wandb.")

if __name__ == "__main__":
    main()
