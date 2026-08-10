# Sparsity Sweep (5-Seed Variance)

We need to mathematically prove the robustness of the `mask_aware` architecture against the baselines under extreme sensory deprivation. We will iterate across 5 random seeds, testing the temporal bounds from 10% active sensors down to a microscopic 0.1% active sensors.

## 1. Dataset Parameterization
Update `SyntheticWaddingtonDataset` in `src/data/waddington_dataset.py` to accept a dynamic `sparsity` parameter in `__init__` and `__getitem__`. The current hardcoded logic `(torch.rand(...) > 0.95)` must use the passed sparsity variable instead.

## 2. Dedicated Sweep Orchestrator
Instead of polluting `sensor_fusion_runner.py`, create a dedicated orchestrator: `src/harness/sparsity_sweep_runner.py`.

This script will encapsulate a double for-loop over sparsities and seeds, train the models programmatically, and output the final graph.

**Sweep Parameters:**
- `sparsities = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]`
- `seeds = [42, 100, 256, 512, 1024]`
- `models = ["baseline", "forward_fill", "mask_concat", "gru_d", "ode_rnn", "mask_aware"]`
- `epochs = 10` (Reduced to keep the 180-model sweep computationally feasible)

## 3. Makefile Target
Add a new target to the `Makefile`:
```makefile
sensor-fusion-sparsity-sweep:
	python -m src.harness.sparsity_sweep_runner
```

## 4. Plotting
The script should generate a plot showing OOD Generalization (MSE) vs. Sensor Sparsity with 5-Seed Variance shading.

Example Plotting Logic:
```python
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
    print("\n[+] Dashboard saved to output/data/05_sparsity_sweep.png")
```
