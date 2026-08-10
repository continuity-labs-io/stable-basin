import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.models.simulators.sensor_fusion_predictor import SensorFusionPredictor, SSMType
from src.utils.device import get_optimal_device
from torch.utils.data import DataLoader, Dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DatasetWrapper(Dataset):
    def __init__(self, size: int, seq_len: int = 500):
        self.dataset = SyntheticWaddingtonDataset(size=size, seq_len=seq_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def save_benchmark_plot(
    model_names, loss_history, preds_dict, test_y_true, train_seq_len, test_seq_len, task_name, png_name
):
    # Plot configuration constants
    TEXT_OFFSET_X = 10
    TEXT_RELATIVE_Y = 0.9
    TEXT_FONT_SIZE = 10

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Top Subplot: Training Loss
    colors = {
        SSMType.BASELINE.value: "r--",
        SSMType.FORWARD_FILL.value: "m-.",
        SSMType.MASK_CONCAT.value: "y-",
        SSMType.TRANSFORMER.value: "g:",
        SSMType.MASK_AWARE.value: "b-",
        SSMType.GRU_D.value: "c-",
        SSMType.ODE_RNN.value: "m-",
    }

    labels = {
        SSMType.BASELINE.value: "Zero-Padded SSM",
        SSMType.FORWARD_FILL.value: "Forward-Fill SSM",
        SSMType.MASK_CONCAT.value: "Mask-Concat SSM",
        SSMType.TRANSFORMER.value: "Causal Transformer",
        SSMType.MASK_AWARE.value: "MASR (Ours)",
        SSMType.GRU_D.value: "GRU-D",
        SSMType.ODE_RNN.value: "ODE-RNN",
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
        ax2.plot(
            preds_dict[name],
            colors[name],
            linewidth=linewidth,
            alpha=alpha,
            label=labels[name],
        )

    ax2.axvline(x=train_seq_len, color="grey", linestyle="--", linewidth=2)
    ax2.text(
        train_seq_len + TEXT_OFFSET_X,
        ax2.get_ylim()[1] * TEXT_RELATIVE_Y,
        "Training Horizon\n(Length Extrapolation)",
        color="grey",
        fontsize=TEXT_FONT_SIZE,
    )

    ax2.set_title(
        f"Waddington Phase Tracking: {task_name.capitalize()} (seq_len={test_seq_len})"
    )
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Phase State")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs("output/data", exist_ok=True)
    out_path = f"output/data/{png_name}"
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved plot to {out_path}")


def run_benchmark(task_name, epochs, train_seq_len, test_seq_len, model_names, csv_name, png_name):
    device = get_optimal_device(verbose=True)

    # Data Preparation
    train_dataset = DatasetWrapper(size=100, seq_len=train_seq_len)
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Initialize all models
    models = {name: SensorFusionPredictor(name).to(device) for name in model_names}

    optimizers = {
        name: optim.AdamW(models[name].parameters(), lr=0.005) for name in model_names
    }
    criterion = nn.MSELoss()

    loss_history = {name: [] for name in model_names}

    # Training Loop
    logger.info(f"--- Starting Task: {task_name} ---")
    logger.info(f"Training {len(models)} architectures on seq_len={train_seq_len} for {epochs} epochs...")
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
            logger.info(log_str)

    logger.info(f"Running Length Extrapolation Stress Test (seq_len={test_seq_len})...")
    for m in models.values():
        m.eval()

    ood_dataset = SyntheticWaddingtonDataset(size=1, seq_len=test_seq_len)
    test_batch = ood_dataset[0]

    test_x_raw = test_batch["x_raw"].unsqueeze(0).to(device)
    test_mask = test_batch["mask"].unsqueeze(0).to(device)
    test_y_true = test_batch["y_true"].cpu().numpy()

    preds_dict = {}
    with torch.no_grad():
        for name, model in models.items():
            preds_dict[name] = model(test_x_raw, test_mask)[0].cpu().numpy()

            # Calculate MSE
            if test_seq_len > train_seq_len:
                ood_mse = ((preds_dict[name][train_seq_len:] - test_y_true[train_seq_len:]) ** 2).mean()
                logger.info(f"OOD-MSE [{name}]: {ood_mse:.4f}")
            else:
                mse = ((preds_dict[name] - test_y_true) ** 2).mean()
                logger.info(f"MSE [{name}]: {mse:.4f}")

    results = {"True Phase": test_y_true.flatten()}
    for k, v in preds_dict.items():
        results[k] = v.flatten()

    df = pd.DataFrame(results)
    csv_out_path = f"output/data/{csv_name}"
    df.to_csv(csv_out_path, index=False)
    logger.info(f"Saved CSV to {csv_out_path}")

    save_benchmark_plot(
        model_names, loss_history, preds_dict, test_y_true, train_seq_len, test_seq_len, task_name, png_name
    )


def main():
    parser = argparse.ArgumentParser(description="Sensor Fusion Benchmark Runner")
    parser.add_argument("--task-name", type=str, required=True, help="Name of the task/benchmark")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--train-seq-len", type=int, default=500, help="Training sequence length")
    parser.add_argument("--test-seq-len", type=int, default=2000, help="Testing sequence length")
    parser.add_argument("--models", type=str, nargs="+", default=[m.value for m in SSMType], help="List of models to benchmark")
    parser.add_argument("--csv-name", type=str, required=True, help="Filename for the output CSV")
    parser.add_argument("--png-name", type=str, required=True, help="Filename for the output PNG")
    
    args = parser.parse_args()
    
    run_benchmark(
        task_name=args.task_name,
        epochs=args.epochs,
        train_seq_len=args.train_seq_len,
        test_seq_len=args.test_seq_len,
        model_names=args.models,
        csv_name=args.csv_name,
        png_name=args.png_name
    )

if __name__ == "__main__":
    main()
