# Role Instruction

You are an expert PyTorch ML Engineer. We need to demonstrate the "Length
Extrapolation" failure mode of standard Transformers compared to continuous-time
State-Space Models.

# Task 1: Update the Dataset Simulator

In `src/data/waddington_dataset.py`, modify the `__getitem__` logic so the
discrete phase jumps scale proportionally to `self.seq_len`. (This ensures the
phase transitions happen out-of-distribution during our 2000-step test). Replace
the hardcoded `jump1` and `jump2` variable declarations with:

```python
# Proportional jumps based on seq_len
jump1 = torch.randint(int(self.seq_len * 0.2), int(self.seq_len * 0.4), (1,)).item()
jump2 = torch.randint(int(self.seq_len * 0.6), int(self.seq_len * 0.8), (1,)).item()
```

# Task 2: Create the Extrapolation Script

Create `src/experiments/02_extrapolation_benchmark.py`. Copy the entire contents
of `src/experiments/01_train_synthetic_benchmark.py` into this new file. Keep
the training loop exactly as it is (training on length 500 for 30 epochs). We
are ONLY replacing the "Evaluation & Plotting" section at the very bottom with
the following code:

```python
# ==========================================
# EVALUATION: LENGTH EXTRAPOLATION STRESS TEST
# ==========================================
print("Running Length Extrapolation Stress Test (seq_len=2000)...")
model_baseline.eval()
model_mask_aware.eval()
model_transformer.eval()

# Generate an Out-Of-Distribution test sequence (4x longer than training)
ood_dataset = SyntheticWaddingtonDataset(size=1, seq_len=2000)
test_batch = ood_dataset[0]

# Add batch dimension and move to device
test_x_raw = test_batch["x_raw"].unsqueeze(0).to(device)
test_mask = test_batch["mask"].unsqueeze(0).to(device)
test_y_true = test_batch["y_true"].cpu().numpy()

with torch.no_grad():
    test_pred_baseline = model_baseline(test_x_raw, test_mask)[0].cpu().numpy()
    test_pred_mask_aware = model_mask_aware(test_x_raw, test_mask)[0].cpu().numpy()
    test_pred_transformer = model_transformer(test_x_raw, test_mask)[0].cpu().numpy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top Subplot: Training Loss
ax1.plot(baseline_loss_history, "r--", linewidth=2, label="Baseline SSM")
ax1.plot(mask_aware_loss_history, "b-", linewidth=2, label="Mask-Aware SSM")
ax1.plot(transformer_loss_history, "g:", linewidth=2, label="Transformer")
ax1.set_title("MSE Loss Convergence (Training on seq_len=500)")
ax1.set_ylabel("MSE")
ax1.set_xlabel("Epoch")
ax1.legend()

# Bottom Subplot: Length Extrapolation Tracking
ax2.plot(test_y_true, "k-", linewidth=3, label="True Phase")
ax2.plot(test_pred_baseline, "r--", linewidth=1.5, alpha=0.8, label="Zero-Padded Baseline")
ax2.plot(test_pred_transformer, "g:", linewidth=2, label="Causal Transformer")
ax2.plot(test_pred_mask_aware, "b-", linewidth=2, label="Mask-Aware Routing")

# Mark the training horizon
ax2.axvline(x=500, color="grey", linestyle="--", linewidth=2)
ax2.text(
    510,
    ax2.get_ylim()[1] * 0.9,
    "Training Horizon\n(Length Extrapolation)",
    color="grey",
    fontsize=10,
)

ax2.set_title("Waddington Phase Tracking: Out-Of-Distribution Stress Test (seq_len=2000)")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Phase State")
ax2.legend(loc="upper left")

plt.tight_layout()
os.makedirs("output/data", exist_ok=True)
plt.savefig("output/data/03_extrapolation_results.png")
print("Saved plot to output/data/03_extrapolation_results.png")
```
