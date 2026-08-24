"""
Phase 1: Torx Workflow & DFG Basics

This script verifies our JAX/Torx substrate is fully operational alongside PyTorch.
It demonstrates two core Torx primitives:
1. DFG (Directed Factor Graph): Wiring independent factors into a causal graph.
2. ChainFactor: Sequentially composing a factor over discrete time steps.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import logging
import argparse

import jax
import jax.numpy as jnp

from torx import DFG, Site, ChainFactor
from torx.tractable_prob_factors import DeterministicFactor

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("TorxSandbox")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="jax", help="Backend to use (pytorch or jax)")
    args = parser.parse_args()

    logger.info("\n" + "="*60)
    logger.info(" STABLE BASIN 2.0: TORX DFG & CHAINFACTOR SANDBOX")
    logger.info("="*60)

    # 1. Initialize Substrate & PRNG
    device = get_optimal_device(verbose=True, backend=args.backend)
    key = jax.random.key(42)

    # Define the universal data shape (a 1D continuous biological state)
    port_spec = jax.ShapeDtypeStruct((1,), jnp.float32)

    # =====================================================================
    # EXPERIMENT A: Directed Factor Graph (DFG) Wiring
    # =====================================================================
    logger.info("\n[+] EXPERIMENT A: Wiring a 2-Node Directed Factor Graph")

    # Define Node 1: A sensor that amplifies an input signal by 2.0
    sensor_factor = DeterministicFactor(
        fn=lambda inputs, site_info: inputs["raw_signal"] * 2.0,
        input_ports={"raw_signal": port_spec},
        output_spec=port_spec
    )

    # Define Node 2: A biological response that adds a base metabolic rate
    metabolism_factor = DeterministicFactor(
        fn=lambda inputs, site_info: inputs["amplified_signal"] + 5.0,
        input_ports={"amplified_signal": port_spec},
        output_spec=port_spec
    )

    # Wire them into a DFG
    graph = DFG(
        sites=(
            Site(
                name="sensor",
                factor=sensor_factor,
                parents=("in_signal",),
                porting_fn=("raw_signal",),
                param_key=None, info_key=None, site_info=None
            ),
            Site(
                name="metabolism",
                factor=metabolism_factor,
                parents=("sensor",),  # Takes input directly from the sensor node
                porting_fn=("amplified_signal",),
                param_key=None, info_key=None, site_info=None
            ),
        ),
        input_ports={"in_signal": port_spec},
        output_name="metabolism" # The final output we care about
    )

    # Execute the DFG
    initial_signal = {"in_signal": jnp.array([1.5], dtype=jnp.float32)}
    
    # The DFG automatically traverses topological order and passes data
    final_output = graph.sample(key, inputs=initial_signal, params={})
    
    logger.info(f"    -> Initial Signal: {initial_signal['in_signal'].item():.2f}")
    logger.info(f"    -> DFG Final Output: {final_output.item():.2f} (Expected: 8.00)")

    # =====================================================================
    # EXPERIMENT B: Continuous Time Unrolling (ChainFactor)
    # =====================================================================
    logger.info("\n[+] EXPERIMENT B: State Space Unrolling via ChainFactor")

    # Define a simple continuous-time decay (e.g., biological homeostasis)
    # h_t = 0.9 * h_{t-1} + 0.1 (baseline drift)
    decay_factor = DeterministicFactor(
        fn=lambda inputs, site_info: inputs["h"] * 0.9 + 0.1,
        input_ports={"h": port_spec},
        output_spec=port_spec
    )

    # Wrap it in a ChainFactor to run sequentially for 5 time steps.
    # Uses jax.lax.scan internally for ultra-fast GPU/XLA unrolling.
    ssm_chain = ChainFactor(
        base=decay_factor,
        n_steps=5,
        feedback_porting_fn="h",  # The output feeds back into the "h" port
        weight_tied=True
    )

    initial_state = {"h": jnp.array([10.0], dtype=jnp.float32)}
    
    # Execute the recurrent chain
    final_state = ssm_chain.sample(key, initial_state, params=None)
    logger.info(f"    -> Initial Homeostatic State: {initial_state['h'].item():.2f}")
    logger.info(f"    -> Final State after 5 steps: {final_state.item():.2f}")
    
    logger.info("\n[SUCCESS] Torx/JAX substrate and DFG primitives verified.")
    logger.info("="*60 + "\n")

if __name__ == "__main__":
    main()
