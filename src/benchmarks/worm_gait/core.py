import pingouin as pg
from scipy.stats import ks_2samp, wasserstein_distance
from src.echo.harness.echo_runner import EchoRunner
from src.echo.harness.echo_trainer import EchoTrainer
from src.echo.metrics.energy_landscape import curvature_over_states, ScalarEnergy
import logging
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from src.benchmarks.aging_resilience.task_registry import get_benchmark_task
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.predictive_coding_graph import PredictiveCodingGraph
from src.echo.primitives.ebm import PrecisionWeightedEBM
from jaxtyping import PRNGKeyArray

logger = logging.getLogger(__name__)

def build_graph(ebm_class, key, config):
    k1, k2, k3 = jax.random.split(key, 3)

    d_internal_micro = config["observer"]["micro"]["d_internal"]
    d_sensory_micro = config["observer"]["micro"]["d_sensory"]
    d_active_micro = config["observer"]["micro"]["d_active"]
    d_external_micro = config["observer"]["micro"]["d_external"]
    d_micro = d_internal_micro + d_sensory_micro + d_active_micro + d_external_micro

    d_internal_macro = config["observer"]["macro"]["d_internal"]
    d_sensory_macro = config["observer"]["macro"]["d_sensory"]
    d_active_macro = config["observer"]["macro"]["d_active"]
    d_external_macro = config["observer"]["macro"]["d_external"]

    micro_cfg = config["observer"]["micro"]
    macro_cfg = config["observer"]["macro"]

    micro = MarkovBlanketObserver(
        d_internal_micro, d_sensory_micro, d_active_micro, d_external_micro,
        ebm_hidden_size=micro_cfg["ebm_hidden_size"],
        ebm_depth=micro_cfg["ebm_depth"],
        n_steps=micro_cfg["n_steps"],
        temperature=micro_cfg["temperature"],
        key=k1,
    )

    macro = MarkovBlanketObserver(
        d_internal_macro, d_sensory_macro, d_active_macro, d_external_macro,
        ebm_hidden_size=macro_cfg["ebm_hidden_size"],
        ebm_depth=macro_cfg["ebm_depth"],
        n_steps=macro_cfg["n_steps"],
        temperature=macro_cfg["temperature"],
        key=k2,
    )

    micro = eqx.tree_at(
        lambda m: m.ebm, micro,
        ebm_class(d_state=d_micro, hidden_size=micro_cfg["ebm_hidden_size"], depth=micro_cfg["ebm_depth"], key=k3),
    )
    macro = eqx.tree_at(
        lambda m: m.ebm, macro,
        ebm_class(d_state=macro.hull.d_state, hidden_size=macro_cfg["ebm_hidden_size"], depth=macro_cfg["ebm_depth"], key=k3),
    )

    graph = PredictiveCodingGraph(micro, macro, n_steps=config["graph"]["n_steps"], key=k3)
    return graph, d_micro + macro.hull.d_state


@eqx.filter_jit
def simulate_sde(
    graph: PredictiveCodingGraph,
    x0: jax.Array,
    lambda_gain: float,
    N: int,
    dt: float,
    key: PRNGKeyArray,
) -> jax.Array:
    d_micro = graph.d_micro
    d_macro = graph.d_macro
    d_full = d_micro + d_macro

    ff = graph.flow_factor

    Q_micro = ff.micro_solenoidal.Q
    L_micro = jnp.tril(ff.micro_dissipative.W)
    Gamma_micro = L_micro @ L_micro.T
    if ff.use_micro_blanket:
        M_micro = ff.micro_hull.get_topology_mask()
        Q_micro = Q_micro * M_micro
        Gamma_micro = Gamma_micro * M_micro

    Q_macro = ff.macro_solenoidal.Q
    L_macro = jnp.tril(ff.macro_dissipative.W)
    Gamma_macro = L_macro @ L_macro.T
    if ff.use_macro_blanket:
        M_macro = ff.macro_hull.get_topology_mask()
        Q_macro = Q_macro * M_macro
        Gamma_macro = Gamma_macro * M_macro

    Q_full = jax.scipy.linalg.block_diag(Q_micro, Q_macro)
    Gamma_full = jax.scipy.linalg.block_diag(Gamma_micro, Gamma_macro)

    evals, evecs = jnp.linalg.eigh(Gamma_full + ff.epsilon * jnp.eye(d_full))
    evals = jnp.maximum(evals, 0.0)
    S_full = evecs @ jnp.diag(jnp.sqrt(evals))

    def energy_fn(x):
        x_u = x[:d_micro]
        x_m = x[d_micro:]
        return lambda_gain * ff.joint_energy_fn(x_u, x_m)

    T_micro = 0.05

    def scan_step(x, key_step):
        grad_E = jax.grad(energy_fn)(x)
        drift = -(Q_full + Gamma_full) @ grad_E
        drift = jnp.clip(drift, -100.0, 100.0)  # prevent gradient explosion
        dW = jax.random.normal(key_step, (d_full,))
        diffusion = jnp.sqrt(2.0 * T_micro * dt) * (S_full @ dW)
        x_next = x + drift * dt + diffusion
        return x_next, x_next

    keys = jax.random.split(key, N)
    _, trajectory = jax.lax.scan(scan_step, x0, keys)
    return jnp.vstack([x0, trajectory])


def setup_experiment(config):
    logger.info("Initializing the physics engine (PredictiveCodingGraph).")
    seed = config["experiment"]["seed"]
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    c_micro = config["observer"]["micro"]
    c_macro = config["observer"]["macro"]

    micro = MarkovBlanketObserver(
        c_micro["d_internal"], c_micro["d_sensory"], c_micro["d_active"], c_micro["d_external"],
        ebm_hidden_size=c_micro["ebm_hidden_size"], ebm_depth=c_micro["ebm_depth"], n_steps=1, temperature=c_micro["temperature"], key=k1,
    )
    macro = MarkovBlanketObserver(
        c_macro["d_internal"], c_macro["d_sensory"], c_macro["d_active"], c_macro["d_external"],
        ebm_hidden_size=c_macro["ebm_hidden_size"], ebm_depth=c_macro["ebm_depth"], n_steps=1, temperature=c_macro["temperature"], key=k2,
    )
    d_micro_full = c_micro["d_internal"] + c_micro["d_sensory"] + c_micro["d_active"] + c_micro["d_external"]
    d_macro_full = c_macro["d_internal"] + c_macro["d_sensory"] + c_macro["d_active"] + c_macro["d_external"]

    micro = eqx.tree_at(
        lambda m: m.ebm, micro,
        PrecisionWeightedEBM(d_state=d_micro_full, hidden_size=c_micro["ebm_hidden_size"], depth=c_micro["ebm_depth"], key=k3),
    )
    macro = eqx.tree_at(
        lambda m: m.ebm, macro,
        PrecisionWeightedEBM(d_state=d_macro_full, hidden_size=c_macro["ebm_hidden_size"], depth=c_macro["ebm_depth"], key=k3),
    )

    graph = PredictiveCodingGraph(micro, macro, n_steps=1, key=k3)
    model_path = config["paths"]["model_weights"]
    try:
        graph = eqx.tree_deserialise_leaves(model_path, graph)
        logger.info("Successfully loaded trained engine.")
    except Exception as e:
        logger.warning(f"Trained model not found at {model_path}! Proceeding with random initialization.")

    d_full = graph.d_micro + graph.d_macro
    task = get_benchmark_task(config)
    _, _, eval_old_dataset_raw = task.get_raw_datasets(config)
    
    sample = eval_old_dataset_raw[0]
    if isinstance(sample, (tuple, list)):
        bio_frame = sample[0].numpy()
    else:
        bio_frame = sample.numpy()
        
    logger.info("Extracting pathological initial state (x0) from 'Old' fallback.")
    x0_noise = jax.random.normal(k4, (d_full,)) * 2.0
    x0_np = np.array(x0_noise)
    idx_s = micro.hull.d_internal
    idx_e = micro.hull.d_internal + micro.hull.d_sensory
    x0_np[idx_s:idx_e] = bio_frame
    x0 = jnp.array(x0_np)
    return graph, x0, k5


def get_full_states(graph, loader):
    full_traj_list = []
    for batch in loader:
        s_true = batch["s_true"].numpy()
        x_init = batch["x_init"].numpy()
        for i in range(len(s_true)):
            # Note: seq is s_true[i]
            traj = graph.forced_unroll(
                jax.random.PRNGKey(0), jnp.array(x_init[i]), 0.01, jnp.array(s_true[i])
            )
            full_traj = traj
            full_traj_list.append(full_traj)
    return jnp.concatenate(full_traj_list, axis=0)


def compute_full_trace(energy_fn, states, batch_size=1000):
    res = curvature_over_states(energy_fn, states, chunk_size=batch_size, nonfinite="drop")
    logger.info(f"Dropped {res['n_nonfinite']} non-finite traces out of {states.shape[0]}.")
    return res["hessian_trace"]


def compute_metrics(name, t_young, t_old):
    ty = np.array(t_young)
    to = np.array(t_old)
    ks_stat, ks_pval = ks_2samp(ty, to)
    wd = wasserstein_distance(ty, to)
    d = pg.compute_effsize(ty, to, eftype="cohen")
    metrics = {
        "mean_young": float(np.mean(ty)),
        "std_young": float(np.std(ty)),
        "mean_old": float(np.mean(to)),
        "std_old": float(np.std(to)),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_pval),
        "wasserstein_distance": float(wd),
        "cohens_d": float(d),
    }
    logger.info(f"--- Metrics for {name} ---")
    logger.info(f"Young: mean={metrics['mean_young']:.4f}, std={metrics['std_young']:.4f}")
    logger.info(f"Old:   mean={metrics['mean_old']:.4f}, std={metrics['std_old']:.4f}")
    logger.info(f"KS Stat: {metrics['ks_statistic']:.4f} (p={metrics['ks_p_value']:.4e})")
    logger.info(f"Wasserstein Dist: {metrics['wasserstein_distance']:.4f}")
    logger.info(f"Cohen's d: {metrics['cohens_d']:.4f}")
    return metrics


def run_aging_experiment(
    config, ebm_class, key, train_young_loader, eval_young_loader, eval_old_loader, config_path
):
    """
        Executes a complete training and evaluation pipeline for a given Energy-Based Model class
        on the aging benchmark dataset.

        Args:
            config (dict): The configuration dictionary.
            ebm_class (type): The class of the Energy-Based Model to instantiate.
            key (jax.Array): A JAX PRNG key for random initialization.
            train_young_loader (DataLoader): DataLoader for the training set (Young population).
            eval_young_loader (DataLoader): DataLoader for evaluating the Young population.
            eval_old_loader (DataLoader): DataLoader for evaluating the Old population.
            config_path (str): The file path to the YAML configuration to be read by the EchoRunner.

        Returns:
            tuple: A 4-tuple containing:
    - metrics (dict): A dictionary of statistics including Cohen's d and Wasserstein
                distance.
                - trace_young (jnp.ndarray): The calculated Hessian traces for the Young population.
                - trace_old (jnp.ndarray): The calculated Hessian traces for the Old population.
                - graph (PredictiveCodingGraph): The fully trained predictive coding graph.
    """
    opt_cfg = config.get("optimization", {})
    lr = opt_cfg.get("learning_rate", 0.0001)
    max_grad_norm = opt_cfg.get("max_grad_norm", 0.1)
    dt = config.get("experiment", {}).get("dt", 0.01)

    logger.info(f"Training Run: {ebm_class.__name__}")
    graph, _ = build_graph(ebm_class, key, config)
    trainer = EchoTrainer(graph, learning_rate=lr, max_grad_norm=max_grad_norm)

    runner = EchoRunner(config_path)
    runner.setup(trainer)
    graph = runner.run(graph, train_young_loader, eval_young_loader, key, dt=dt)

    logger.info(f"Evaluating {ebm_class.__name__} on biological population.")
    full_states_young = get_full_states(graph, eval_young_loader)
    full_states_old = get_full_states(graph, eval_old_loader)

    logger.info(f"Computing Hessian Traces for {ebm_class.__name__}.")
    energy_fn = ScalarEnergy(graph.ebm)
    trace_young = compute_full_trace(energy_fn, full_states_young)
    trace_old = compute_full_trace(energy_fn, full_states_old)

    metrics = compute_metrics(ebm_class.__name__, trace_young, trace_old)
    return metrics, trace_young, trace_old, graph

