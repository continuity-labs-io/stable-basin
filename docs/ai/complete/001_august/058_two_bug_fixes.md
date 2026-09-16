# Role Instruction

You are an expert PyTorch ML Engineer. We are fixing two critical bugs in the
benchmark: a "Symmetry Lock" destroying the Subspace Router, and a data-leakage
bug in the dataset generator.

# Task 1: Fix the Data Leakage (Deterministic W_1)

Overwrite `src/data/waddington_dataset.py`. Update the `__init__` method of
`SyntheticWaddingtonDataset` to ensure `self.W_1` is deterministic so the test
set speaks the same biological language as the training set:

```python
    def __init__(self, size=100, seq_len=500):
        self.size = size
        self.seq_len = seq_len
        # Ensure consistent biological mapping across datasets
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.W_1 = torch.randn(1, 10)
        torch.set_rng_state(rng_state)
```

Task 2: True Orthogonal Subspace Routing Overwrite
src/models/encoders/fusion.py. Replace the entire BiologicalCartridgeFusion
class with this code to explicitly partition the 64-D latent space and
mathematically guarantee stasis:

```python
import torch
import torch.nn as nn

class BiologicalCartridgeFusion(nn.Module):
    def __init__(self, d_cartridge: int, n_modalities: int, d_model: int):
        super().__init__()
        self.W_cart = nn.Linear(d_cartridge, d_model, bias=False)
        self.W_gate = nn.Linear(n_modalities, d_model, bias=True)

        # TRUE ORTHOGONAL SUBSPACE ROUTING PRIOR
        half = d_model // 2
        with torch.no_grad():
            # 1. Absolute Stasis Default: sigmoid(-10) = 4.5e-5.
            # Guarantees memory survives 1000+ step voids.
            self.W_gate.bias.fill_(-10.0)
            self.W_gate.weight.fill_(0.0)

            # 2. Orthogonal Routing
            # Modality 0 (Voltage) strictly controls latent dimensions 0 to 31
            self.W_gate.weight[:half, 0] = 20.0

            # Modality 1 (Epigenetics) strictly controls latent dimensions 32 to 63
            self.W_gate.weight[half:, 1] = 20.0

    def forward(self, x_raw: torch.Tensor, mask: torch.Tensor):
        latent_x = self.W_cart(x_raw)
        latent_gate = torch.sigmoid(self.W_gate(mask))
        return latent_x, latent_gate
```

Task 3: Increase Training Epochs In
src/experiments/01_train_synthetic_benchmark.py:

Change epochs = 30 to epochs = 50 to allow the optimizer to fully leverage the
newly protected subspace.
