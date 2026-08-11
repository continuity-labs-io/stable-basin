import torch
import numpy as np
from src.data.waddington_dataset import SyntheticWaddingtonDataset


def test_waddington_data_shapes_and_masks():
    dataset = SyntheticWaddingtonDataset(size=5, seq_len=500)
    assert len(dataset) == 5

    batch = dataset[0]
    x_raw = batch["x_raw"]
    mask = batch["mask"]
    y_true = batch["y_true"]

    assert x_raw.shape == (500, 30)
    assert mask.shape == (500, 2)
    assert y_true.shape == (500, 1)

    # Modality 0 (first 20 dims) mask should be all 1s
    assert torch.all(mask[:, 0] == 1.0)

    # Modality 1 (last 10 dims) mask should be sparse (~5%)
    density = mask[:, 1].mean().item()
    assert 0.01 < density < 0.15  # generous bounds for random generation

    # Check zero padding on modality 1 when mask is 0
    mod1_unmasked = x_raw[:, 20:][mask[:, 1] == 0]
    assert torch.all(mod1_unmasked == 0.0)


def test_waddington_dynamics_leakage():
    """
    Ensure Modality 0 (Continuous slow variable) provides information about w,
    but does NOT provide a direct shortcut to v (y_true).
    Modality 1 (Sparse fast variable) SHOULD correlate strongly with v when unmasked.
    """
    dataset = SyntheticWaddingtonDataset(size=1, seq_len=2000)
    batch = dataset[0]

    x_raw = batch["x_raw"]
    y_true = batch["y_true"]
    mask = batch["mask"]

    mod0 = x_raw[:, :20]
    mod1 = x_raw[:, 20:]

    y_true_np = y_true.squeeze().numpy()

    # Calculate Pearson correlation between each dimension of Mod0 and y_true
    max_mod0_corr = 0.0
    for dim in range(20):
        mod0_dim = mod0[:, dim].numpy()
        # They will be somewhat correlated because w and v are coupled,
        # but shouldn't be perfectly correlated (no shortcut).
        if np.std(mod0_dim) > 1e-5 and np.std(y_true_np) > 1e-5:
            corr = np.abs(np.corrcoef(mod0_dim, y_true_np)[0, 1])
            if corr > max_mod0_corr:
                max_mod0_corr = corr

    # Modality 0 is the slow variable, which lags the fast variable. 
    # Correlation is usually moderate but never > 0.9.
    assert max_mod0_corr < 0.90, (
        f"Data leakage detected! Modality 0 has suspiciously high correlation {max_mod0_corr} with target v."
    )

    # Now check Modality 1 (when it is NOT masked out)
    mod1_active_idx = mask[:, 1] == 1.0
    active_y = y_true[mod1_active_idx].squeeze().numpy()
    active_mod1 = mod1[mod1_active_idx]

    # At least one dimension in Modality 1 MUST be highly correlated because it's the projection of v
    max_mod1_corr = 0.0
    for dim in range(10):
        mod1_dim = active_mod1[:, dim].numpy()
        if np.std(mod1_dim) > 1e-5 and np.std(active_y) > 1e-5:
            corr = np.abs(np.corrcoef(mod1_dim, active_y)[0, 1])
            if corr > max_mod1_corr:
                max_mod1_corr = corr

    assert max_mod1_corr > 0.5, (
        f"Causal link missing! Modality 1 has max correlation {max_mod1_corr} with target v."
    )
