import os
import argparse
import yaml
import tempfile
import logging
import jax
import jax.numpy as jnp
import torch
import json
import pingouin as pg
from scipy.stats import ks_2samp, wasserstein_distance
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx

from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset, SyntheticWormMockDataset
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.hierarchy import PredictiveCodingGraph
from src.echo.architecture.datasets import JAXDictDataset
from src.echo.primitives.ebm import GaussianEBM, PrecisionWeightedEBM
from src.echo.harness.echo_runner import EchoRunner
from src.echo.harness.echo_trainer import EchoTrainer
from src.echo.metrics.thermal_interpretability import HessianCurvatureTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_graph(ebm_class, key, config):
    """
    Constructs the hierarchical Predictive Coding Graph for the worm gait benchmark.
    
    Args:
        ebm_class: The Energy-Based Model class to use (e.g., GaussianEBM or PrecisionWeightedEBM).
        key: JAX PRNG key for random number generation.
        config: Dictionary containing the configuration values for the observers.
        
    Returns:
        A tuple containing:
            - graph (PredictiveCodingGraph): The initialized PredictiveCodingGraph.
            - d_state (int): The total internal state dimensionality of the combined micro and macro observers.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Micro observer config
    d_internal_micro = config['observer']['micro']['d_internal']
    d_sensory_micro = config['observer']['micro']['d_sensory']
    d_active_micro = config['observer']['micro']['d_active']
    d_external_micro = config['observer']['micro']['d_external']
    d_micro = d_internal_micro + d_sensory_micro + d_active_micro + d_external_micro
    
    # Macro observer config
    d_internal_macro = config['observer']['macro']['d_internal']
    d_sensory_macro = config['observer']['macro']['d_sensory']
    d_active_macro = config['observer']['macro']['d_active']
    d_external_macro = config['observer']['macro']['d_external']
    
    micro_cfg = config['observer']['micro']
    macro_cfg = config['observer']['macro']
    
    micro = MarkovBlanketObserver(
        d_internal_micro, d_sensory_micro, d_active_micro, d_external_micro, 
        ebm_hidden_size=micro_cfg['ebm_hidden_size'], 
        ebm_depth=micro_cfg['ebm_depth'], 
        n_steps=micro_cfg['n_steps'], 
        temperature=micro_cfg['temperature'], 
        key=k1
    )
                                  
    macro = MarkovBlanketObserver(
        d_internal_macro, d_sensory_macro, d_active_macro, d_external_macro, 
        ebm_hidden_size=macro_cfg['ebm_hidden_size'], 
        ebm_depth=macro_cfg['ebm_depth'], 
        n_steps=macro_cfg['n_steps'], 
        temperature=macro_cfg['temperature'], 
        key=k2
    )
    
    # Overwrite the ebm with the desired one
    micro = eqx.tree_at(
        lambda m: m.ebm, 
        micro, 
        ebm_class(d_state=d_micro, hidden_size=micro_cfg['ebm_hidden_size'], depth=micro_cfg['ebm_depth'], key=k3)
    )
    macro = eqx.tree_at(
        lambda m: m.ebm, 
        macro, 
        ebm_class(d_state=macro.hull.d_state, hidden_size=macro_cfg['ebm_hidden_size'], depth=macro_cfg['ebm_depth'], key=k3)
    )
        
    graph = PredictiveCodingGraph(micro, macro, n_steps=config['graph']['n_steps'], key=k3)
    return graph, d_micro + macro.hull.d_state


def get_full_states(graph, loader):
    full_traj_list = []
    for batch in loader:
        s_true = batch['s_true'].numpy()
        x_init = batch['x_init'].numpy()
        for i in range(len(s_true)):
            # Note: seq is s_true[i]
            traj = graph.forced_unroll(
                jax.random.PRNGKey(0), jnp.array(x_init[i]), 0.01, jnp.array(s_true[i])
            )
            full_traj = traj
            full_traj_list.append(full_traj)
    return jnp.concatenate(full_traj_list, axis=0)


def compute_full_trace(tracker, states, batch_size=1000):
    num_states = states.shape[0]
    traces = []
    for i in range(0, num_states, batch_size):
        batch = states[i:i+batch_size]
        res = tracker.batch_calculate_curvature(batch)
        traces.append(res["hessian_trace"])
    return jnp.concatenate(traces, axis=0)


def plot_ablation_results(
    eval_young_dataset_raw,
    eval_old_dataset_raw,
    trace_young_A,
    trace_old_A,
    trace_young_B,
    trace_old_B,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    traj_young = eval_young_dataset_raw.data[0].numpy()
    traj_old = eval_old_dataset_raw.data[0].numpy()
    axes[0].plot(traj_young[:500, 0], traj_young[:500, 1], label="Young (Day 1-3)")
    axes[0].plot(traj_old[:500, 0], traj_old[:500, 1], label="Old (Day 9+)", alpha=0.7)
    axes[0].set_title("Panel A: The Limit Cycle")
    axes[0].set_xlabel("Sensor Dimension 0")
    axes[0].set_ylabel("Sensor Dimension 1")
    axes[0].legend()
    trace_young_A_np = np.nan_to_num(np.array(trace_young_A), nan=1.0)
    trace_old_A_np = np.nan_to_num(np.array(trace_old_A), nan=1.0)
    trace_young_B_np = np.nan_to_num(np.array(trace_young_B), nan=1.0)
    trace_old_B_np = np.nan_to_num(np.array(trace_old_B), nan=1.0)

    # Panel B: Laplace Flatline We use a thick line for Young and a dashed line
    # for Old because the GaussianEBM's Hessian is mathematically constant
    # across the state space, causing perfect overlap.
    axes[1].plot(trace_young_A_np, label="Young", color="blue", linewidth=4)
    axes[1].plot(
        trace_old_A_np, label="Old", color="orange", linestyle="--", linewidth=2
    )
    axes[1].set_title("Panel B: Laplace Flatline")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Hessian Trace (Curvature)")
    axes[1].legend()
    
    # Panel C: Waddington Basin Flattening
    axes[2].hist(
        trace_young_B_np, bins=20, alpha=0.5, label="Young", color="blue", density=True
    )
    axes[2].hist(
        trace_old_B_np, bins=20, alpha=0.7, label="Old", color="orange", 
        density=True, histtype="step", linewidth=2
    )
    axes[2].set_title("Panel C: Waddington Basin Flattening")
    axes[2].set_xlabel("Hessian Trace (Curvature)")
    axes[2].set_ylabel("Density")
    axes[2].legend()
    
    plt.tight_layout()
    os.makedirs("output/echo/benchmarks", exist_ok=True)
    plt.savefig("output/echo/benchmarks/06_worm_gait_decline_ablation.png")
    plt.close()
    
    logger.info(
        "Benchmark complete. Plot saved to "
        "output/echo/benchmarks/06_worm_gait_decline_ablation.png"
    )


def main():
    parser = argparse.ArgumentParser(description="Worm Gait Aging EBM Benchmark")
    parser.add_argument("--config", type=str, default="configs/worm_gait_ebm.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Initializing Young (Train) and Old (Eval) datasets.")
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    
    try:
        seq_len = config['dataset']['seq_len']
        train_young_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TRAIN.ts", seq_len=seq_len, is_aged=False
        )
        eval_young_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TEST.ts", seq_len=seq_len, is_aged=False
        )
        eval_old_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TEST.ts", seq_len=seq_len, is_aged=True
        )
    except FileNotFoundError:
        logger.warning("Local biological data not found. Falling back to SyntheticWormMockDataset.")
        train_young_dataset_raw = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
        eval_young_dataset_raw = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
        eval_old_dataset_raw = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
    
    # Determine d_state
    _, d_state = build_graph(GaussianEBM, key, config)
    
    train_young_dataset = JAXDictDataset(train_young_dataset_raw, d_state)
    eval_young_dataset = JAXDictDataset(eval_young_dataset_raw, d_state)
    eval_old_dataset = JAXDictDataset(eval_old_dataset_raw, d_state)
    
    batch_size = config.get('dataset', {}).get('batch_size', 2)
    train_young_loader = DataLoader(train_young_dataset, batch_size=batch_size, shuffle=True)
    eval_young_loader = DataLoader(eval_young_dataset, batch_size=batch_size, shuffle=False)
    eval_old_loader = DataLoader(eval_old_dataset, batch_size=batch_size, shuffle=False)
    
    opt_cfg = config.get('optimization', {})
    lr = opt_cfg.get('learning_rate', 0.0001)
    max_grad_norm = opt_cfg.get('max_grad_norm', 0.1)
    dt = config.get('experiment', {}).get('dt', 0.01)
    
    logger.info("Training Run A: GaussianEBM (The Laplace Baseline)")
    key, kA = jax.random.split(key)
    graph_A, _ = build_graph(GaussianEBM, kA, config)
    trainer_A = EchoTrainer(graph_A, learning_rate=lr, max_grad_norm=max_grad_norm)
    runner_A = EchoRunner(args.config)
    runner_A.setup(trainer_A)
    graph_A = runner_A.run(graph_A, train_young_loader, train_young_loader, key, dt=dt)
    
    logger.info("Training Run B: PrecisionWeightedEBM (Multimodal MLP)")
    key, kB = jax.random.split(key)
    graph_B, _ = build_graph(PrecisionWeightedEBM, kB, config)
    trainer_B = EchoTrainer(graph_B, learning_rate=lr, max_grad_norm=max_grad_norm)
    runner_B = EchoRunner(args.config)
    runner_B.setup(trainer_B)
    graph_B = runner_B.run(graph_B, train_young_loader, train_young_loader, key, dt=dt)
    
    logger.info("Serializing trained Young Worm engine to disk.")
    os.makedirs("output/echo/benchmarks", exist_ok=True)
    eqx.tree_serialise_leaves(
        "output/echo/benchmarks/06_worm_gait_decline_trained_engine.eqx", graph_B
    )
    
    logger.info("Evaluating frozen EBM models on Day 9+ biological population.")

    # Core experimental conditions: 
    # - Population Age: Young (1-3 days) vs. Old (9+ days)
    # - EBM Architecture: A. Gaussian (Laplace baseline) vs. B. Precision Weighted (multimodal MLP).
    full_states_young_A = get_full_states(graph_A, eval_young_loader)
    full_states_old_A = get_full_states(graph_A, eval_old_loader)
    
    full_states_young_B = get_full_states(graph_B, eval_young_loader)
    full_states_old_B = get_full_states(graph_B, eval_old_loader)
    
    logger.info("Computing Hessian Traces (Full Evaluation Dataset).")
    tracker_A = HessianCurvatureTracker(graph_A.ebm)
    tracker_B = HessianCurvatureTracker(graph_B.ebm)
    
    trace_young_A = compute_full_trace(tracker_A, full_states_young_A)
    trace_old_A = compute_full_trace(tracker_A, full_states_old_A)
    
    trace_young_B = compute_full_trace(tracker_B, full_states_young_B)
    trace_old_B = compute_full_trace(tracker_B, full_states_old_B)
    
    def compute_metrics(name, t_young, t_old):
        ty = np.nan_to_num(np.array(t_young), nan=1.0)
        to = np.nan_to_num(np.array(t_old), nan=1.0)
        ks_stat, ks_pval = ks_2samp(ty, to)
        wd = wasserstein_distance(ty, to)
        d = pg.compute_effsize(ty, to, eftype='cohen')
        metrics = {
            "mean_young": float(np.mean(ty)),
            "std_young": float(np.std(ty)),
            "mean_old": float(np.mean(to)),
            "std_old": float(np.std(to)),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_pval),
            "wasserstein_distance": float(wd),
            "cohens_d": float(d)
        }
        logger.info(f"--- Metrics for {name} ---")
        logger.info(f"Young: mean={metrics['mean_young']:.4f}, std={metrics['std_young']:.4f}")
        logger.info(f"Old:   mean={metrics['mean_old']:.4f}, std={metrics['std_old']:.4f}")
        logger.info(f"KS Stat: {metrics['ks_statistic']:.4f} (p={metrics['ks_p_value']:.4e})")
        logger.info(f"Wasserstein Dist: {metrics['wasserstein_distance']:.4f}")
        logger.info(f"Cohen's d: {metrics['cohens_d']:.4f}")
        return metrics

    logger.info("Calculating Full-Series Thermodynamic Statistics.")
    metrics_A = compute_metrics("Run A (GaussianEBM)", trace_young_A, trace_old_A)
    metrics_B = compute_metrics("Run B (PrecisionWeightedEBM)", trace_young_B, trace_old_B)

    all_metrics = {
        "GaussianEBM": metrics_A,
        "PrecisionWeightedEBM": metrics_B
    }

    metrics_path = "output/echo/benchmarks/06_worm_gait_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Serialized full statistical metrics to {metrics_path}")
    
    plot_ablation_results(
        eval_young_dataset_raw,
        eval_old_dataset_raw,
        trace_young_A,
        trace_old_A,
        trace_young_B,
        trace_old_B,
    )

if __name__ == "__main__":
    main()
