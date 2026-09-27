I want to build a PyTorch `Dataset` that mathematically mimics the 'Compositional Objects With Stickers' (COWS) benchmark failure conditions, completely bypassing the need for a 3D physics simulator. Please write `src/data/behavior/cows_proxy_dataset.py`.

1. Create a `SyntheticCOWSDataset(Dataset)` that generates `size` sequences of `seq_len` (e.g., 100).
2. We have 5 'Macro' classes (e.g., Mug, Cube, etc.) and 7 'Micro' classes (e.g., TBP Logo, Numenta Logo, etc.). This creates 35 possible Compositional Classes (matching COWS Small/Large concepts).
3. **Macro Modality (The 3D Object):** Generate `x_macro` (shape: `[seq_len, 20]`). Create a unique slow-varying continuous base manifold for each of the 5 Macro classes (e.g., using different frequencies of sinusoids or an Ornstein-Uhlenbeck process).
4. **Micro Modality (The 2D Sticker):** Generate `x_micro` (shape: `[seq_len, 10]`). Create a unique fast-varying signal for each of the 7 Micro classes.
5. **The Adversarial Condition (Rotation & Noise):** To simulate the `randrot_noise` condition that breaks standard models, add a boolean `adversarial_mode` parameter. If True, apply a random orthogonal rotation matrix to `x_micro` (simulating spatial misalignment) and add heavy Gaussian noise (`std=0.5`) to both tensors.
6. `__getitem__` should return: `{"x_macro": tensor, "x_micro": tensor, "label": int (0 to 34)}`.

*Note: Please ensure all code uses subdued, professional logging without emojis, following our repository standards.*
