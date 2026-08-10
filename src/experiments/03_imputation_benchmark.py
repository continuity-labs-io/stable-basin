import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.models.simulators.waddington_predictor import WaddingtonPredictor, SSMType
from src.utils.device import get_optimal_device
from torch.utils.data import DataLoader, Dataset

class DatasetWrapper(Dataset):
    def __init__(self, size: int, seq_len: int = 500):
        self.dataset = SyntheticWaddingtonDataset(size=size, seq_len=seq_len)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]

def save_benchmark_plot(model_names, loss_history, preds_dict, test_y_true, train_seq_len, test_seq_len):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Top Subplot: Training Loss
    colors = {
        SSMType.BASELINE.value: "r--", 
        SSMType.FORWARD_FILL.value: "m-.", 
        SSMType.MASK_CONCAT.value: "y-", 
        SSMType.TRANSFORMER.value: "g:", 
        SSMType.MASK_AWARE.value: "b-"
    }
    
    labels = {
        SSMType.BASELINE.value: "Zero-Padded SSM", 
        SSMType.FORWARD_FILL.value: "Forward-Fill SSM", 
        SSMType.MASK_CONCAT.value: "Mask-Concat SSM", 
        SSMType.TRANSFORMER.value: "Causal Transformer", 
        SSMType.MASK_AWARE.value: "MASR (Ours)"
    }
    
    for name in model_names:
        ax1.plot(loss_history[name], colors[name], linewidth=2, label=labels[name])
    
    ax1.set_title(f"MSE Loss Convergence (Training on seq_len={train_seq_len})")
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Bottom Subplot: Extrapolation
    ax2.plot(test_y_true, "k-", linewidth=4, label="True Phase")
    for name in model_names:
        linewidth = 2.5 if name == SSMType.MASK_AWARE.value else 1.5
        alpha = 1.0 if name == SSMType.MASK_AWARE.value else 0.8
        ax2.plot(preds_dict[name], colors[name], linewidth=linewidth, alpha=alpha, label=labels[name])

    ax2.axvline(x=train_seq_len, color="grey", linestyle="--", linewidth=2)
    TEXT_OFFSET_X = 10
    TEXT_RELATIVE_Y = 0.9
    TEXT_FONT_SIZE = 10
    ax2.text(train_seq_len + TEXT_OFFSET_X, ax2.get_ylim()[1] * TEXT_RELATIVE_Y, "Training Horizon\n(Length Extrapolation)", color="grey", fontsize=TEXT_FONT_SIZE)
    
    ax2.set_title(f"Waddington Phase Tracking: Out-Of-Distribution Stress Test (seq_len={test_seq_len})")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Phase State")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs("output/data", exist_ok=True)
    plt.savefig("output/data/04_arxiv_money_chart.png", dpi=300)
    print("Saved plot to output/data/04_arxiv_money_chart.png")


def main():
    # Sequence length used for training the models.
    TRAIN_SEQ_LEN = 500
    # Extended sequence length used to test out-of-distribution capabilities.
    TEST_SEQ_LEN = 2000

    device = get_optimal_device(verbose=True)

    # Data Preparation
    train_dataset = DatasetWrapper(size=100, seq_len=TRAIN_SEQ_LEN)
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Initialize all 5 models
    model_names = [m.value for m in SSMType]
    models = {name: WaddingtonPredictor(name).to(device) for name in model_names}

    optimizers = {name: optim.AdamW(models[name].parameters(), lr=0.005) for name in model_names}
    criterion = nn.MSELoss()

    loss_history = {name: [] for name in model_names}

    # Training Loop
    epochs = 40
    print(f"Training {len(models)} architectures on seq_len={TRAIN_SEQ_LEN}...")
    for epoch in range(1, epochs + 1):
        for m in models.values():
            m.train()

        running_losses = {name: 0.0 for name in model_names}

        for batch in dataloader:
            x_raw = batch["x_raw"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["y_true"].to(device)

            for name, model in models.items():
                optimizers[name].zero_grad()
                preds = model(x_raw, mask)
                loss = criterion(preds, y_true)
                loss.backward()
                optimizers[name].step()
                running_losses[name] += loss.item()

        log_str = f"Epoch {epoch:02d}/{epochs} | "
        for name in model_names:
            avg_loss = running_losses[name] / len(dataloader)
            loss_history[name].append(avg_loss)
            log_str += f"{name}: {avg_loss:.3f} | "
        
        if epoch % 5 == 0 or epoch == 1:
            print(log_str)

    print(f"\nRunning Length Extrapolation Stress Test (seq_len={TEST_SEQ_LEN})...")
    for m in models.values():
        m.eval()

    ood_dataset = SyntheticWaddingtonDataset(size=1, seq_len=TEST_SEQ_LEN)
    test_batch = ood_dataset[0]

    test_x_raw = test_batch["x_raw"].unsqueeze(0).to(device)
    test_mask = test_batch["mask"].unsqueeze(0).to(device)
    test_y_true = test_batch["y_true"].cpu().numpy()

    preds_dict = {}
    with torch.no_grad():
        for name, model in models.items():
            preds_dict[name] = model(test_x_raw, test_mask)[0].cpu().numpy()
            
            # Calculate OOD MSE (T > TRAIN_SEQ_LEN)
            ood_mse = ((preds_dict[name][TRAIN_SEQ_LEN:] - test_y_true[TRAIN_SEQ_LEN:])**2).mean()
            print(f"OOD-MSE [{name}]: {ood_mse:.4f}")

    
    results = {"True Phase": test_y_true.flatten()}
    for k, v in preds_dict.items():
        results[k] = v.flatten()

    df = pd.DataFrame(results)
    df.to_csv("output/data/04_arxiv_results.csv", index=False)
    print("Saved CSV to output/data/04_arxiv_results.csv")

    save_benchmark_plot(model_names, loss_history, preds_dict, test_y_true, TRAIN_SEQ_LEN, TEST_SEQ_LEN)


if __name__ == "__main__":
    main()
