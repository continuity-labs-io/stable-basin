"""
STABLE BASIN: Master Ephys Execution Dashboard (8_ephys_demo.py)
Validating continuous-time Mamba-2 engine on raw 1,024-channel HD-MEA data.
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import psutil
import logging
import argparse

matplotlib.use("Agg")


def setup_diagnostic_logger():
    logger = logging.getLogger("DiagnosticLogger")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    os.makedirs("output", exist_ok=True)
    fh = logging.FileHandler("output/ephys_diagnostic.log")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = setup_diagnostic_logger()

from src.data.ephys.hdmea_dataset import HDMEADataset
from src.models.ssm.spike_forecaster import SpikeForecaster
from src.models.losses.meld_loss import MeldLoss
from src.metrics.metrics import ThermodynamicMetrics
from src.metrics.mamba_lrp import MambaLRPEpsilon
from src.metrics.hardware_monitor import HardwareMonitor
from src.utils.device import get_optimal_device


def plot_ephys_dashboard(
    raw_ephys,
    vram_history,
    ksm_scores,
    relevance,
    event_frame,
    crash_ms,
    filename="8_ephys_demo.png",
):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=False)

    # Decimate time axis for faster plotting (10,000 frames is massive for imshow)
    downsample = 10
    raw_sub = raw_ephys[::downsample, :64]
    rel_sub = relevance[::downsample, :64]

    time_axis = np.arange(raw_sub.shape[0]) * (downsample / 20000.0)
    event_time = event_frame / 20000.0

    # --- Panel 1: Raw Telemetry ---
    ax1 = axes[0]

    event_idx = int(event_frame / downsample)
    if event_idx > 0:
        vmax_raw = np.percentile(np.abs(raw_sub[:event_idx]), 95)
    else:
        vmax_raw = np.max(np.abs(raw_sub))

    ax1.imshow(
        raw_sub.T,
        aspect="auto",
        cmap="magma",
        vmin=-vmax_raw,
        vmax=vmax_raw,
        extent=[time_axis[0], time_axis[-1], 64, 0],
    )
    ax1.axvline(
        x=event_time,
        color="white",
        linestyle="--",
        linewidth=2,
        label=f"Waddington Crash (T={crash_ms}ms)",
    )
    ax1.set_title(
        "Panel 1: Raw HD-MEA 20kHz Telemetry (Subsampled to 64 Ch)",
        color="white",
        fontweight="bold",
    )
    ax1.set_ylabel("Electrode Array")
    ax1.legend(loc="upper right")

    # --- Panel 2: Thermodynamic Extraction (KSM) ---
    ax2 = axes[1]
    ksm_time = np.arange(len(ksm_scores)) / 20000.0
    ax2.plot(ksm_time, ksm_scores, color="springgreen", linewidth=2)
    ax2.axvline(x=event_time, color="white", linestyle="--", linewidth=2)
    ax2.axhline(
        y=0.9, color="crimson", linestyle=":", linewidth=2, label="Stability Collapse Threshold"
    )
    ax2.set_title("Panel 2: PyDMD Koopman Stability Metric (KSM)", color="white", fontweight="bold")
    ax2.set_ylabel("Stable Eigenvalue Bound")
    ax2.set_xlabel("Time (Seconds)")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.2)

    # --- Panel 3: MambaLRP Attribution ---
    ax3 = axes[2]
    vmax = np.max(np.abs(rel_sub)) * 0.5
    im3 = ax3.imshow(
        rel_sub.T,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[time_axis[0], time_axis[-1], 64, 0],
    )
    ax3.axvline(x=event_time, color="white", linestyle="--", linewidth=2)
    ax3.set_title(
        "Panel 3: MambaLRPEpsilon Root Cause Attribution (Targeted at Crash Frame)",
        color="white",
        fontweight="bold",
    )
    ax3.set_xlabel("Continuous Time Context (Seconds)")
    ax3.set_ylabel("Electrode Array")

    cbar = fig.colorbar(im3, ax=ax3, orientation="vertical", pad=0.01)
    cbar.set_label("Predictive Relevance", color="white")

    # --- Panel 4: VRAM Hardware Monitor ---
    ax4 = axes[3]
    ax4.plot(range(1, len(vram_history) + 1), vram_history, color="cyan", marker="o", linewidth=2)
    ax4.set_title(
        "Panel 4: Hardware Monitor - Constant O(1) Memory Footprint",
        color="white",
        fontweight="bold",
    )
    ax4.set_ylabel("Peak VRAM (MB)")
    ax4.set_xlabel("Training Iterations")
    ax4.set_ylim(0, max(vram_history) * 1.5 if vram_history else 100)
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300)
    print(f"[*] Dashboard saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Master Ephys Execution Dashboard")
    parser.add_argument(
        "--seq_ms", type=int, default=500, help="Sequence length in milliseconds (e.g. 500)"
    )
    args = parser.parse_args()

    device = get_optimal_device(allow_mps=False, verbose=True)

    print("\n" + "=" * 80)
    print(" STABLE BASIN: MASTER EPHYS DASHBOARD (8_ephys_demo)")
    print("=" * 80)

    # =========================================================================
    # DEMO KNOBS
    # =========================================================================
    BURN_IN_ITERATIONS = 10
    SEQUENCE_LENGTH_MS = args.seq_ms
    CRASH_INJECTION_MS = 250
    SAMPLING_RATE_HZ = 20000

    SEQ_LEN = int((SEQUENCE_LENGTH_MS / 1000.0) * SAMPLING_RATE_HZ)
    EVENT_FRAME = int((CRASH_INJECTION_MS / 1000.0) * SAMPLING_RATE_HZ)
    TARGET_CHANNELS = 1024

    file_path = "data/ephys/hdmea_neuropulse.brw"

    print("[*] 1. Initializing HDMEADataset...")
    try:
        dataset = HDMEADataset(data_path=file_path, seq_len=SEQ_LEN)
        # Slicing the first 1024 channels for the demo
        batch = dataset[0][:, :TARGET_CHANNELS].unsqueeze(0).to(device)
    except Exception as e:
        print(
            f"[!] Warning: Native BRW dataloader failed ({e}). Falling back to synthetic HD-MEA Tensor."
        )
        batch = torch.randn(1, SEQ_LEN, TARGET_CHANNELS, device=device).abs() * 0.5

    print(f"    -> Extracted sequence shape: {batch.shape}")
    logger.info(
        f"Data Ingestion Audit: shape={batch.shape}, dtype={batch.dtype}, "
        f"min={batch.min().item():.4f}, max={batch.max().item():.4f}, "
        f"mean={batch.mean().item():.4f}, std={batch.std().item():.4f}"
    )

    # 2. Burn-in Training
    print("\n[*] 2. Initializing SpikeForecaster (Mamba-2)...")
    model = SpikeForecaster(input_dim=TARGET_CHANNELS, d_model=256, d_state=64).to(device)

    criterion = MeldLoss(alpha=1.0, beta=0.1, gamma=0.0, L=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("[*] Commencing 10-Iteration Burn-In Loop...")
    model.train()

    vram_history = []
    process = psutil.Process()

    state_t = batch[:, :-1, :]
    target_t_plus_1 = batch[:, 1:, :]
    delta_x = torch.full((1, 1), 1.0 / SAMPLING_RATE_HZ, device=device)

    for iteration in range(1, BURN_IN_ITERATIONS + 1):
        optimizer.zero_grad()

        pred_t_plus_1 = model(state_t)

        # Mock reconstructed_t to bypass Time-Reversal projection requirement
        loss, _ = criterion(state_t, target_t_plus_1, pred_t_plus_1, state_t, delta_x)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        logger.debug(
            f"Burn-in Audit [Iter {iteration}]: loss={loss.item():.4f}, grad_norm={grad_norm.item():.4f}"
        )

        # Manual VRAM tracking
        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated() / (1024**2)
        elif device.type == "mps":
            mem = torch.mps.current_allocated_memory() / (1024**2)
        else:
            mem = process.memory_info().rss / (1024**2)
        vram_history.append(mem)

        print(
            f"    [Iteration {iteration:02d}/{BURN_IN_ITERATIONS}] Loss: {loss.item():.4f} (Stable via Z-Score) | VRAM: {mem:.2f} MB"
        )

    # 3. The Waddington Crash
    print(
        f"\n[*] 3. Simulating Waddington Crash (Biological Flatline at T={EVENT_FRAME} / {CRASH_INJECTION_MS}ms)..."
    )
    model.eval()
    val_seq = batch.clone()

    # Simulate true biological flatline: drive voltage to near 0 to force DMD eigenvalues to collapse
    val_seq[:, EVENT_FRAME:, :] = 0.0

    before_crash = val_seq[:, EVENT_FRAME - 1000 : EVENT_FRAME, :]
    after_crash = val_seq[:, EVENT_FRAME : EVENT_FRAME + 1000, :]
    logger.info(
        f"Crash Audit: Before crash mean={before_crash.mean().item():.6f}, std={before_crash.std().item():.6f}"
    )
    logger.info(
        f"Crash Audit: After crash mean={after_crash.mean().item():.6f}, std={after_crash.std().item():.6f}"
    )

    # 4. Thermodynamic Extraction
    print("\n[*] 4. Extracting Thermodynamic Manifold (Koopman Stability Metric)...")
    with torch.no_grad():
        _, hidden_states = model(val_seq, return_hidden=True)

    print("    -> Passing to PyDMD...")
    metrics = ThermodynamicMetrics(alpha=500.0, beta=1.0)

    # Decimate by 50 to compute fast (10,000 fits takes minutes otherwise)
    decimation_factor = 50
    z_seq_decimated = hidden_states[0, ::decimation_factor, :]

    # Pass debug_crash_frame as the decimated frame index of the crash
    ksm_scores_decimated = metrics.calculate_ksm(
        z_seq_decimated, window_size=5, debug_crash_frame=(EVENT_FRAME // decimation_factor)
    )

    ksm_scores = np.interp(
        np.arange(SEQ_LEN),
        np.arange(len(ksm_scores_decimated)) * decimation_factor,
        ksm_scores_decimated,
    )

    # 5. Attribution
    print("\n[*] 5. Executing MambaLRPEpsilon Root Cause Attribution...")
    lrp = MambaLRPEpsilon(model)
    relevance_tensor = lrp.attribute(val_seq, target_time_step=EVENT_FRAME)

    logger.info(f"LRP Audit: Total sum of relevance_tensor={relevance_tensor.sum().item():.4f}")

    raw_numpy = val_seq[0].cpu().numpy()
    rel_numpy = relevance_tensor[0].cpu().numpy()

    # 6. Dashboard
    print("\n[*] 6. Rendering 4-Panel Publication-Ready Dashboard...")
    plot_ephys_dashboard(
        raw_numpy, vram_history, ksm_scores, rel_numpy, EVENT_FRAME, CRASH_INJECTION_MS
    )
    print("\n[+] EPHYS PIPELINE DEMO COMPLETE.")


if __name__ == "__main__":
    main()
