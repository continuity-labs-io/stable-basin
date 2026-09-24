"""
Executive Summary: Waddington Collapse Benchmark

This benchmark evaluates a hierarchical observer operating within a thermodynamic framework.
At its core, this experiment is designed to test a Predictive Coding Graph against a
pharmacological perturbation (Diazepam) that interrupts electrical communication.

By analyzing the "Waddington collapse," the benchmark measures whether the thermodynamic
collapse (the flattening of the macro-state energy basin, quantified by the Hessian trace)
can anticipate the actual physical electrical crash of the system.

This updated version trains the graph on healthy baseline sequences to form the Waddington
basin, before executing the true drug-shock sequence.
"""

import os
import jax
import jax.numpy as jnp
import equinox as eqx
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.data.ephys.pharma_shock_dataset import PharmacologicalShockDataset
from src.data.datasets import JAXDictDataset
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.predictive_coding_graph import PredictiveCodingGraph
from src.echo.metrics.energy_landscape import curvature_over_states, ScalarEnergy
from src.echo.harness.echo_trainer import EchoTrainer
from src.echo.primitives.ebm import PrecisionWeightedEBM


def build_waddington_graph(key, input_dim):
    k1, k2, k3 = jax.random.split(key, 3)

    # Micro Observer (d_sensory must match input_dim)
    d_internal_micro = 64
    d_sensory_micro = input_dim
    d_active_micro = 64
    d_external_micro = 64
    d_micro = d_internal_micro + d_sensory_micro + d_active_micro + d_external_micro

    micro = MarkovBlanketObserver(
        d_internal_micro,
        d_sensory_micro,
        d_active_micro,
        d_external_micro,
        ebm_hidden_size=64,
        ebm_depth=2,
        n_steps=1,
        temperature=1.0,
        key=k1,
    )

    # Macro Observer (condensed latent space)
    d_internal_macro = 16
    d_sensory_macro = 8
    d_active_macro = 4
    d_external_macro = 4

    macro = MarkovBlanketObserver(
        d_internal_macro,
        d_sensory_macro,
        d_active_macro,
        d_external_macro,
        ebm_hidden_size=32,
        ebm_depth=2,
        n_steps=1,
        temperature=1.0,
        key=k2,
    )

    # Use PrecisionWeightedEBM
    micro = eqx.tree_at(
        lambda m: m.ebm,
        micro,
        PrecisionWeightedEBM(d_state=d_micro, hidden_size=64, depth=2, key=k3),
    )
    macro = eqx.tree_at(
        lambda m: m.ebm,
        macro,
        PrecisionWeightedEBM(d_state=macro.hull.d_state, hidden_size=32, depth=2, key=k3),
    )

    graph = PredictiveCodingGraph(micro, macro, n_steps=1, key=k3)
    return graph, d_micro + macro.hull.d_state


def run_waddington_collapse_benchmark(
    output_plot: str = "output/benchmarks/basic/01_waddington_collapse_plot.png",
) -> int:
    key = jax.random.PRNGKey(42)
    k_graph, k_train, k_eval = jax.random.split(key, 3)

    # 1. Load control (healthy) data for burn-in training
    print("Loading control dataset for burn-in...")
    try:
        control_dataset_raw = PharmacologicalShockDataset(condition="control", seq_len=128)
    except FileNotFoundError as e:
        print(f"Dataset not found, skipping full execution. {e}")
        return 0

    # Get input dimension from the first sample
    sample_tensor = control_dataset_raw[0]
    input_dim = sample_tensor.shape[1]

    # 2. Build the graph
    graph, d_state = build_waddington_graph(k_graph, input_dim)

    control_dataset = JAXDictDataset(control_dataset_raw, d_state)
    control_loader = DataLoader(control_dataset, batch_size=4, shuffle=True)

    # 3. Burn-in Training
    print("Starting burn-in training...")
    trainer = EchoTrainer(graph, learning_rate=1e-3, max_grad_norm=1.0)

    epochs = 2
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, batch in enumerate(control_loader):
            k_step = jax.random.fold_in(k_train, epoch * 1000 + i)
            # Convert PyTorch DataLoader tensors to JAX arrays
            jax_batch = {k: jnp.array(v.numpy()) for k, v in batch.items()}
            graph, trainer, loss = trainer.step(graph, jax_batch, k_step, dt=0.01)
            epoch_loss += loss
            if i >= 20:  # Train for a few steps to form basin
                break
        print(f"Burn-in Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / 20:.4f}")

    print("Burn-in complete.")

    # 4. Load the pharmacological shock sequence
    print("Loading 50uM shock dataset...")
    shock_dataset_raw = PharmacologicalShockDataset(condition="50uM", seq_len=2000)
    data_tensor = shock_dataset_raw[0]  # Grab a 2000-frame sequence

    # Preprocess the data tensor
    data_np = data_tensor.numpy()
    data_np = (data_np - np.mean(data_np)) / (np.std(data_np) + 1e-5)
    data_seq = jnp.array(data_np)

    # Initialize random starting states for unrolling
    x_micro_init = jax.random.normal(k_eval, (graph.d_micro,)) * 0.1
    x_macro_init = jax.random.normal(k_eval, (graph.d_macro,)) * 0.1
    x_init = jnp.concatenate([x_micro_init, x_macro_init])

    print("Unrolling sequence over the shocked data...")
    trajectory = graph.forced_unroll(k_eval, x_init, dt=0.01, seq=data_seq)

    # Extract Macro-State trajectory
    macro_traj = trajectory[:, graph.d_micro :]

    # 5. Metric Extraction & EBET Calculation
    print("Calculating macro-state curvature...")
    energy_fn = ScalarEnergy(graph.flow_factor.macro_ebm)
    res = curvature_over_states(energy_fn, macro_traj, chunk_size=500, nonfinite="keep")
    trace_np = np.array(res["hessian_trace"])

    # Compute rolling variance to find the physical crash
    rolling_var = np.array([np.var(data_np[max(0, i - 50) : i + 1]) for i in range(len(data_np))])

    baseline_var = np.mean(rolling_var[50:200]) if len(rolling_var) >= 200 else np.mean(rolling_var)
    var_threshold = 0.5 * baseline_var

    electrical_crash_frame = -1
    for i in range(200, len(rolling_var)):
        if rolling_var[i] < var_threshold:
            electrical_crash_frame = i
            break

    if electrical_crash_frame == -1:
        electrical_crash_frame = len(rolling_var) - 1

    # Find thermodynamic collapse
    baseline_trace = np.nanmean(trace_np[:200]) if len(trace_np) >= 200 else np.nanmean(trace_np)
    collapse_threshold = 0.5 * baseline_trace

    thermodynamic_collapse_frame = -1
    for i in range(200, len(trace_np)):
        if trace_np[i] < collapse_threshold:
            thermodynamic_collapse_frame = i
            break

    if thermodynamic_collapse_frame == -1:
        thermodynamic_collapse_frame = len(trace_np) - 1

    ebet = electrical_crash_frame - thermodynamic_collapse_frame

    print(f"Thermodynamic Collapse Frame: {thermodynamic_collapse_frame}")
    print(f"Electrical Crash Frame: {electrical_crash_frame}")
    print(f"Energy Basin Escape Time (EBET): {ebet}")

    if ebet > 0:
        print(
            "MVM PROOF SUCCESS: Curvature flattened BEFORE physical signal collapse. Top-down "
            "failure confirmed."
        )
    else:
        print("MVM PROOF FAILED: Curvature did not anticipate physical signal collapse.")

    # 6. Output Visualization
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    ax1.plot(rolling_var, color="blue", label="HD-MEA Rolling Variance")
    ax1.axvline(x=electrical_crash_frame, color="red", linestyle="--", label="Electrical Crash")
    ax1.set_ylabel("Variance")
    ax1.set_title("Physical Crash (Ground Truth)")
    ax1.legend()

    ax2.plot(trace_np, color="green", label="Macro Hessian Trace")
    ax2.axvline(
        x=thermodynamic_collapse_frame,
        color="orange",
        linestyle="--",
        label="Thermodynamic Collapse",
    )
    ax2.set_xlabel("Time (Frames)")
    ax2.set_ylabel("Hessian Trace")
    ax2.set_title("Waddington Basin Geometry (Trained Basin)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    print(f"Saved plot to {output_plot}")
    return ebet


if __name__ == "__main__":
    run_waddington_collapse_benchmark()
