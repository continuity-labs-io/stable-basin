import os
import yaml
import tempfile
import logging
import jax
import jax.numpy as jnp
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx

from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset, SyntheticWormMockDataset
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.hierarchy import PredictiveCodingGraph
from src.echo.primitives.ebm import GaussianEBM, PrecisionWeightedEBM
from src.echo.harness.echo_runner import EchoRunner
from src.echo.harness.echo_trainer import EchoTrainer
from src.echo.metrics.thermal_interpretability import HessianCurvatureTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class JAXDictDataset(Dataset):
    def __init__(self, base_dataset, d_state):
        self.base = base_dataset
        self.d_state = d_state
        
    def __len__(self):
        return len(self.base)
        
    def __getitem__(self, idx):
        s_true = self.base[idx]
        x_init = torch.randn(self.d_state) * 0.01
        return {'s_true': s_true, 'x_init': x_init}

def build_graph(ebm_class, key):
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Micro observer config
    d_internal_micro = 8
    d_sensory_micro = 6
    d_active_micro = 8
    d_external_micro = 8
    d_micro = d_internal_micro + d_sensory_micro + d_active_micro + d_external_micro
    
    # Macro observer config
    d_internal_macro = 4
    d_sensory_macro = 4
    d_active_macro = 4
    d_external_macro = 4
    
    micro = MarkovBlanketObserver(
        d_internal_micro, d_sensory_micro, d_active_micro, d_external_micro, 
        ebm_hidden_size=32, ebm_depth=2, n_steps=1, temperature=1.0, key=k1
    )
                                  
    macro = MarkovBlanketObserver(
        d_internal_macro, d_sensory_macro, d_active_macro, d_external_macro, 
        ebm_hidden_size=16, ebm_depth=2, n_steps=1, temperature=1.0, key=k2
    )
    
    # Overwrite the ebm with the desired one
    micro = eqx.tree_at(
        lambda m: m.ebm, 
        micro, 
        ebm_class(d_state=d_micro, hidden_size=32, depth=2, key=k3)
    )
    macro = eqx.tree_at(
        lambda m: m.ebm, 
        macro, 
        ebm_class(d_state=macro.hull.d_state, hidden_size=16, depth=2, key=k3)
    )
        
    graph = PredictiveCodingGraph(micro, macro, n_steps=1, key=k3)
    return graph, d_micro + macro.hull.d_state


def get_macro_states(graph, loader):
    macro_traj_list = []
    for batch in loader:
        s_true = batch['s_true'].numpy()
        x_init = batch['x_init'].numpy()
        for i in range(len(s_true)):
            # Note: seq is s_true[i]
            traj = graph.forced_unroll(
                jax.random.PRNGKey(0), jnp.array(x_init[i]), 0.01, jnp.array(s_true[i])
            )
            macro_traj = traj[:, graph.d_micro:]
            macro_traj_list.append(macro_traj)
    return jnp.concatenate(macro_traj_list, axis=0)


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
    logger.info("Initializing Young (Train) and Old (Eval) datasets.")
    torch.manual_seed(42)
    key = jax.random.PRNGKey(42)
    
    try:
        train_young_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TRAIN.ts", seq_len=100, is_aged=False
        )
        eval_young_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TEST.ts", seq_len=100, is_aged=False
        )
        eval_old_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TEST.ts", seq_len=100, is_aged=True
        )
    except FileNotFoundError:
        logger.warning("Local biological data not found. Falling back to SyntheticWormMockDataset.")
        train_young_dataset_raw = SyntheticWormMockDataset(seq_len=100, num_samples=50)
        eval_young_dataset_raw = SyntheticWormMockDataset(seq_len=100, num_samples=50)
        eval_old_dataset_raw = SyntheticWormMockDataset(seq_len=100, num_samples=50)
    
    # Determine d_state
    _, d_state = build_graph(GaussianEBM, key)
    
    train_young_dataset = JAXDictDataset(train_young_dataset_raw, d_state)
    eval_young_dataset = JAXDictDataset(eval_young_dataset_raw, d_state)
    eval_old_dataset = JAXDictDataset(eval_old_dataset_raw, d_state)
    
    train_young_loader = DataLoader(train_young_dataset, batch_size=2, shuffle=True)
    eval_young_loader = DataLoader(eval_young_dataset, batch_size=2, shuffle=False)
    eval_old_loader = DataLoader(eval_old_dataset, batch_size=2, shuffle=False)
    
    config_path = "configs/echo_training.yaml"
    logger.info("Training Run A: GaussianEBM (The Laplace Baseline)")
    key, kA = jax.random.split(key)
    graph_A, _ = build_graph(GaussianEBM, kA)
    trainer_A = EchoTrainer(graph_A, learning_rate=0.0001, max_grad_norm=0.1)
    runner_A = EchoRunner(config_path)
    runner_A.setup(trainer_A)
    graph_A = runner_A.run(graph_A, train_young_loader, train_young_loader, key, dt=0.01)
    
    logger.info("Training Run B: PrecisionWeightedEBM (Multimodal MLP)")
    key, kB = jax.random.split(key)
    graph_B, _ = build_graph(PrecisionWeightedEBM, kB)
    trainer_B = EchoTrainer(graph_B, learning_rate=0.0001, max_grad_norm=0.1)
    runner_B = EchoRunner(config_path)
    runner_B.setup(trainer_B)
    graph_B = runner_B.run(graph_B, train_young_loader, train_young_loader, key, dt=0.01)
    
    logger.info("Serializing trained Young Worm engine to disk.")
    os.makedirs("output/echo/benchmarks", exist_ok=True)
    eqx.tree_serialise_leaves(
        "output/echo/benchmarks/06_worm_gait_decline_trained_engine.eqx", graph_B
    )
    
    logger.info("Evaluating frozen EBM models on Day 9+ biological population.")

    # Core experimental conditions: 
    # - Population Age: Young (1-3 days) vs. Old (9+ days)
    # - EBM Architecture: A. Gaussian (Laplace baseline) vs. B. Precision Weighted (multimodal MLP).
    macro_states_young_A = get_macro_states(graph_A, eval_young_loader)
    macro_states_old_A = get_macro_states(graph_A, eval_old_loader)
    
    macro_states_young_B = get_macro_states(graph_B, eval_young_loader)
    macro_states_old_B = get_macro_states(graph_B, eval_old_loader)
    
    logger.info("Computing Hessian Traces.")
    tracker_A = HessianCurvatureTracker(graph_A.flow_factor.macro_ebm)
    tracker_B = HessianCurvatureTracker(graph_B.flow_factor.macro_ebm)
    
    eval_states_young_A = macro_states_young_A[::10][:1000]
    trace_young_A = tracker_A.batch_calculate_curvature(eval_states_young_A)["hessian_trace"]
    
    eval_states_old_A = macro_states_old_A[::10][:1000]
    trace_old_A = tracker_A.batch_calculate_curvature(eval_states_old_A)["hessian_trace"]
    
    eval_states_young_B = macro_states_young_B[::10][:1000]
    trace_young_B = tracker_B.batch_calculate_curvature(eval_states_young_B)["hessian_trace"]
    
    eval_states_old_B = macro_states_old_B[::10][:1000]
    trace_old_B = tracker_B.batch_calculate_curvature(eval_states_old_B)["hessian_trace"]
    
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
