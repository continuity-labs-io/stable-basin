import numpy as np
import pytest
from src.data.sim2real.rna_dataloader import TranscriptomicLoader


def test_psi_dataloader_event_tensor():
    loader = TranscriptomicLoader(crash_minute=10)
    event_tensor = loader.build_continuous_event_tensor(total_minutes=15)

    # Test 1: Shape Validation
    assert event_tensor.ndim == 2, "Tensor must be 2D"
    assert event_tensor.shape[1] == 3, (
        "Tensor must have exactly 3 features: [Time, Gene_Idx, Intensity]"
    )

    # Test 2: Chronological Ordering
    times = event_tensor[:, 0].numpy()
    assert np.all(np.diff(times) >= 0), "Events are not strictly chronological!"

    # Test 3: Waddington Crash Compounding Logic
    # Verify panic gene (e.g., TP53, index 1) fires much more frequently after minute 10
    tp53_idx = 1.0
    pre_crash_tp53 = event_tensor[
        (event_tensor[:, 0] < 10 * 60000) & (event_tensor[:, 1] == tp53_idx)
    ]
    post_crash_tp53 = event_tensor[
        (event_tensor[:, 0] >= 10 * 60000) & (event_tensor[:, 1] == tp53_idx)
    ]

    pre_crash_rate = len(pre_crash_tp53) / 10.0  # events per min
    post_crash_rate = len(post_crash_tp53) / 5.0  # events per min

    assert post_crash_rate > pre_crash_rate * 5, (
        "Panic gene hazard rate did not explode after crash!"
    )
