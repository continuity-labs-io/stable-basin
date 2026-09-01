import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import logging

from src.echo.architecture.markov_hull import MarkovHull
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.hierarchy import PredictiveCodingGraph
from src.echo.clinic.interventions import DigitalTwinAnnealer, DigitalTwinInterrogator
from src.echo.metrics.thermal_interpretability import HessianCurvatureTracker
from src.echo.physics.dissipative import DissipativeFriction
from src.echo.primitives.ebm import PrecisionWeightedEBM

# Configure simple, readable logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calc_micro_drift(graph: PredictiveCodingGraph, x_micro: jax.Array, x_macro: jax.Array) -> jax.Array:
    factor = graph.thermalizer.graph.sites[0].factor.base
    grad_micro, grad_macro = jax.grad(factor.joint_energy_fn, argnums=(0, 1))(x_micro, x_macro)
    
    M_micro = factor.micro_hull.get_topology_mask()
    Q_micro = factor.micro_solenoidal.Q * M_micro
    L_micro = jnp.tril(factor.micro_dissipative.W)
    Gamma_micro = (L_micro @ L_micro.T) * M_micro
    
    drift_micro = -(Q_micro + Gamma_micro) @ grad_micro
    return drift_micro

def main():
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    d_internal = 4
    d_sensory = 4
    d_active = 4
    d_external = 4
    d_state = d_internal + d_sensory + d_active + d_external
    
    # Create a JAX PredictiveCodingGraph containing two nested MarkovBlanketObservers: 
    # 1) A fast, local Micro-Observer (cells) 
    # 2) A slow, global Macro-Observer (tissue/organ)
    #
    # We hack the weights using equinox.tree_at to simulate the biological ravages of aging 
    # via three specific thermodynamic failures:
    # 1) Flattened Prior (Loss of Identity)
    # 2) Induced Blindness (Silent Drift)
    # 3) Lowered Friction (Mechanical Decay)
    logging.info("=== Step 1: Initialize Degraded Alice ===")
    micro_obs = MarkovBlanketObserver(
        d_internal=d_internal, d_sensory=d_sensory, d_active=d_active, d_external=d_external,
        ebm_hidden_size=16, ebm_depth=2, n_steps=1, temperature=0.0, key=k1
    )
    macro_obs = MarkovBlanketObserver(
        d_internal=d_internal, d_sensory=d_sensory, d_active=d_active, d_external=d_external,
        ebm_hidden_size=16, ebm_depth=2, n_steps=1, temperature=0.0, key=k2
    )
    
    alice = PredictiveCodingGraph(micro_observer=micro_obs, macro_observer=macro_obs, n_steps=1, key=k3)
    
    # Degrade Alice to simulate aging:
    # 1. Flatten the macro attractor basin (degrade macro EBM)
    # 2. Induce Silent Drift / Blindness (sever W_down so micro prediction errors don't reach macro)
    # 3. Lower physical friction
    factor = alice.thermalizer.graph.sites[0].factor.base
    
    factor_degraded = eqx.tree_at(lambda f: f.macro_ebm.energy_head.weight, factor, factor.macro_ebm.energy_head.weight * 0.01)
    factor_degraded = eqx.tree_at(lambda f: f.macro_ebm.precision_head.weight, factor_degraded, factor_degraded.macro_ebm.precision_head.weight * 0.01)
    factor_degraded = eqx.tree_at(lambda f: f.W_down.weight, factor_degraded, factor_degraded.W_down.weight * 0.0)
    factor_degraded = eqx.tree_at(lambda f: f.micro_dissipative.W, factor_degraded, factor_degraded.micro_dissipative.W * 0.01)
    factor_degraded = eqx.tree_at(lambda f: f.macro_dissipative.W, factor_degraded, factor_degraded.macro_dissipative.W * 0.01)
    
    alice_degraded = eqx.tree_at(lambda g: g.thermalizer.graph.sites[0].factor.base, alice, factor_degraded)
    
    x_micro = jax.random.normal(k4, (d_state,)) * 0.1
    x_macro = jax.random.normal(k5, (d_state,)) * 0.1
    
    logging.info("Patient 'Alice' initialized and synthetically degraded.")
    
    # Attach the HessianCurvatureTracker to Alice's Macro-Observer.
    # Drop a mathematical plumb bob into her neural network by computing the 
    # trace of the 2nd derivative of her energy landscape.
    logging.info("\n=== Step 2: Measure Geometry (The Hessian) ===")
    tracker = HessianCurvatureTracker(alice_degraded.thermalizer.graph.sites[0].factor.base.macro_ebm)
    hessian_res = tracker.calculate_curvature(x_macro)
    hessian_trace = float(hessian_res["hessian_trace"])
    
    logging.info(f"STEP 2: Measuring Waddington Geometry... Trace = {hessian_trace:.4f}. Attractor basin is flattened.")
    
    # Is Alice resting comfortably, or has she lost the capacity to detect damage?
    # We "ping" her with a strong virtual jolt of energy directly into her micro-level sensors.
    # We then use the Joint Free Energy equation to measure the resulting Prediction Error.
    logging.info("\n=== Step 3: The Hardware Ping (Detecting Silent Drift) ===")
    interrogator = DigitalTwinInterrogator()
    q_ext_pulse = jax.random.normal(k1, (d_state,)) * 5.0  # Strong ping
    
    res = interrogator.ping_and_measure(alice_degraded, x_micro, x_macro, q_ext_pulse)
    micro_surp = res["micro_surprisal"]
    macro_surp = res["macro_surprisal"]
    discordance = res["discordance"]
    
    logging.info(f"Micro Surprisal: {micro_surp:.4f} | Macro Surprisal: {macro_surp:.4f} | Discordance: {discordance:.4f}")
    if discordance < 0.1:
        logging.info("DIAGNOSIS: Silent Drift Detected! Macro-level is blind to physical micro-damage.")
    else:
        logging.info("DIAGNOSIS: Concordant.")
        
    # How do we fix Alice? We create her Counterfactual Twin (Twin B). 
    # Using the DigitalTwinAnnealer, we mathematically multiply her 
    # Friction (Γ) and Precision (Π) weights by 100.0.
    logging.info("\n=== Step 4: Compute Counterfactual (The Reference Twin) ===")
    annealer = DigitalTwinAnnealer()
    alice_optimal = annealer.anneal_twin(alice_degraded, gamma_boost=100.0, pi_boost=100.0)
    
    # Also boost the energy head so the basin steepness explicitly changes in the plot
    alice_optimal = jax.tree_util.tree_map(
        lambda x: eqx.tree_at(lambda p: p.energy_head.weight, x, x.energy_head.weight * 100.0) 
        if isinstance(x, PrecisionWeightedEBM) else x,
        alice_optimal,
        is_leaf=lambda x: isinstance(x, PrecisionWeightedEBM)
    )
    logging.info("Twin B generated. Friction (Γ) and Precision (Π) have been restored in silico without a population database.")
    
    # We have the Sick Twin (A), and we have the Healthy Twin (B). 
    # We want the physical patient to behave like the Healthy Twin.
    # We ask the physics engine to calculate the deterministic drift vector 
    # (the restorative force) for both twins at Alice's exact current coordinates.
    #
    # drift_A: What Alice's frail body is currently doing to heal itself (very little).
    # drift_B: What Alice's optimal body should be doing (pulling aggressively back to center).
    #
    # The Result: We subtract them: Q_actuation = drift_B - drift_A. 
    # This vector is the Thermodynamic Cure. It is the exact, continuous-time exogenous 
    # bioelectric frequency the hardware must inject into Alice's physical sensors. 
    # By applying this frequency, the hardware acts as a synthetic Markov Blanket, 
    # artificially providing the top-down precision and friction that her aged cells 
    # can no longer generate themselves.
    logging.info("\n=== Step 5: Actuate (The Hardware Delta) ===")
    drift_A = calc_micro_drift(alice_degraded, x_micro, x_macro)
    drift_B = calc_micro_drift(alice_optimal, x_micro, x_macro)
    
    Q_actuation = drift_B - drift_A
    q_norm = float(jnp.linalg.norm(Q_actuation))
    
    logging.info(f"Required Actuation Norm (L2): {q_norm:.4f}")
    logging.info("This is the exact exogenous energy the hardware must inject to force the physical tissue back into its youthful limit cycle.")
    
    # OUTPUT: Matplotlib dashboard
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Subplot A
    ax1.bar(["Micro Surprisal", "Macro Surprisal"], [micro_surp, macro_surp], color=['red', 'blue'])
    ax1.set_title("Step 3: Ping Phase (Silent Drift)")
    ax1.set_ylabel("Gradient Norm (L2)")
    
    # Subplot B
    tracker_B = HessianCurvatureTracker(alice_optimal.thermalizer.graph.sites[0].factor.base.macro_ebm)
    hessian_res_B = tracker_B.calculate_curvature(x_macro)
    hessian_trace_B = float(hessian_res_B["hessian_trace"])
    
    ax2.bar(["Twin A (Degraded)", "Twin B (Optimal)"], [hessian_trace, hessian_trace_B], color=['orange', 'green'])
    ax2.set_title("Step 2 & 4: Waddington Basin Curvature")
    ax2.set_ylabel("Curvature / Steepness")
    
    plt.suptitle("Active Inference Clinical Workflow")
    plt.tight_layout()
    plt.savefig("output/echo/active_inference_clinical_workflow.png")
    plt.close()

if __name__ == "__main__":
    main()
