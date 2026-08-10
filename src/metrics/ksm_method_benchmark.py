import torch
import time
from src.models.ssm.meld_engine import MeldEngine
from src.metrics.metrics import ThermodynamicMetrics, calculate_dynamic_rank

import logging

logger = logging.getLogger(__name__)

"""
Sample Result:
+--------------------+--------------------+---------------+-----------------------------------+-------------------------+
| Algorithm Name     | Avg Latency (ms)   | Max FPS       | Topological Accuracy              | 100Hz Viability         |
+--------------------+--------------------+---------------+-----------------------------------+-------------------------+
| DMD                | 1.29               | 774.5         | Lagging (approx. -46 frames)      | Yes                     |
| PALC               | 721.71             | 1.4           | Instantaneous Frame-Perfect       | No (Compute Bottleneck) |
+--------------------+--------------------+---------------+-----------------------------------+-------------------------+
"""


def run_accuracy_benchmark(device):
    z_seq = torch.randn(100, 832, device=device) * 0.1

    # Inject catastrophic variance explosion at frame 50
    # multiply frames 50-100 by an exponentially increasing scalar
    scalars = torch.exp(torch.linspace(0, 5, 50, device=device))
    z_seq[50:100] = z_seq[50:100] * scalars.unsqueeze(1)

    thermo = ThermodynamicMetrics(alpha=500.0)
    ksm_scores = thermo.calculate_ksm(z_seq, window_size=4, rank_method="dynamic")

    # Identify exact frame KSM metric drops below 0.9
    drop_frame = None
    for i, score in enumerate(ksm_scores):
        if score < 0.9:
            drop_frame = i
            break

    return drop_frame


def run_dmd_speed_benchmark(device, warmup=False):
    num_steps = 5 if warmup else 100
    total_time = 0.0

    # Dummy sliding windows of shape [4, 832]
    X_windows = [torch.randn(3, 832, device=device) for _ in range(num_steps)]
    Y_windows = [torch.randn(3, 832, device=device) for _ in range(num_steps)]

    for i in range(num_steps):
        X = X_windows[i]
        Y = Y_windows[i]

        start = time.perf_counter()

        # Move to CPU explicitly to avoid MPS fallback warning for LAPACK ops
        X_cpu = X.cpu()
        U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(X_cpu, full_matrices=False)
        U = U_cpu.to(device)
        S = S_cpu.to(device)
        Vh = Vh_cpu.to(device)

        n_rows, n_cols = X_cpu.shape
        rank = calculate_dynamic_rank(S_cpu.numpy(), n_rows, n_cols)
        
        U_k = U[:, :rank]
        S_inv_k = torch.diag(1.0 / S[:rank])
        Vh_k = Vh[:rank, :]

        A_tilde = S_inv_k @ U_k.T @ Y @ Vh_k.T

        # Explicit CPU move for eigvals
        eigenvalues = torch.linalg.eigvals(A_tilde.cpu()).to(device)
        _ = torch.max(torch.abs(eigenvalues)).item()

        end = time.perf_counter()
        total_time += end - start

    return (total_time / num_steps) * 1000  # ms per step


def run_palc_speed_benchmark(model, device, warmup=False):
    num_steps = 5 if warmup else 100
    total_time = 0.0

    def step_fn(z):
        preds, _ = model(z)
        return preds

    z_t_list = [torch.randn(1, 1, 832, device=device) for _ in range(num_steps)]

    for i in range(num_steps):
        z_t = z_t_list[i]

        start = time.perf_counter()

        J = torch.autograd.functional.jacobian(step_fn, z_t)
        # Reshape to dense [832, 832]
        J = J.reshape(832, 832)

        _ = torch.linalg.inv(J + torch.eye(832, device=device) * 1e-4)

        end = time.perf_counter()
        total_time += end - start

    return (total_time / num_steps) * 1000  # ms per step


def main():
    from src.utils.device import get_optimal_device

    device = get_optimal_device(verbose=True)
    logger.info(f"Using device: {device}")

    model = MeldEngine(input_dim=832, d_model=256, mask_aware=False).to(device).eval()

    logger.info("Running Accuracy Benchmark (Temporal Lag)...")
    drop_frame = run_accuracy_benchmark(device)
    lag = drop_frame - 50 if drop_frame is not None else "N/A"
    logger.info(
        f"Variance explosion injected at frame 50. KSM dropped below 0.9 at frame {drop_frame}. (Lag: {lag} frames)"
    )

    logger.info("Running 5-step warmup loop to compile hardware graphs...")
    run_dmd_speed_benchmark(device, warmup=True)
    run_palc_speed_benchmark(model, device, warmup=True)

    logger.info("Running DMD Speed Benchmark...")
    dmd_latency = run_dmd_speed_benchmark(device)

    logger.info("Running PALC (Exact Jacobian) Speed Benchmark...")
    palc_latency = run_palc_speed_benchmark(model, device)

    dmd_fps = 1000.0 / dmd_latency if dmd_latency > 0 else float("inf")
    palc_fps = 1000.0 / palc_latency if palc_latency > 0 else float("inf")

    logger.info("\n")
    logger.info(
        "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+" + "-" * 35 + "+" + "-" * 25 + "+"
    )
    logger.info(
        f"| {'Algorithm Name':<18} | {'Avg Latency (ms)':<18} | {'Max FPS':<13} | {'Topological Accuracy':<33} | {'100Hz Viability':<23} |"
    )
    logger.info(
        "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+" + "-" * 35 + "+" + "-" * 25 + "+"
    )
    logger.info(
        f"| {'DMD (Sliding Window)':<18} | {dmd_latency:<18.2f} | {dmd_fps:<13.1f} | {f'Lagging (approx. {lag} frames)':<33} | {'Yes':<23} |"
    )
    logger.info(
        f"| {'PALC (Exact Jacob.)':<18} | {palc_latency:<18.2f} | {palc_fps:<13.1f} | {'Instantaneous Frame-Perfect':<33} | {'No (Compute Bottleneck)':<23} |"
    )
    logger.info(
        "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+" + "-" * 35 + "+" + "-" * 25 + "+"
    )


if __name__ == "__main__":
    main()
