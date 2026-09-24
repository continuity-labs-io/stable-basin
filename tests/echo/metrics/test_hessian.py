import pytest
import numpy as np
import jax
import jax.numpy as jnp
from src.echo.metrics.hessian import HessianTraceEvaluator

class DummyFlowFactor:
    def joint_energy_fn(self, x_micro, x_macro):
        x = jnp.concatenate([x_micro, x_macro])
        return jnp.sum(x**2)

class DummyGraph:
    def __init__(self):
        self.d_micro = 2
        self.d_macro = 2
        self.flow_factor = DummyFlowFactor()

    def forced_unroll(self, key, x_init, dt, seq):
        return jnp.repeat(x_init[None, :], seq.shape[0], axis=0)

def test_hessian_trace_evaluator():
    """Test the Hessian Trace Evaluator.
    Must adhere to ARRANGE, ACT, ASSERT block structure.
    """
    # ARRANGE
    graph = DummyGraph()
    dt = 0.01
    seed = 42
    burn_in = 2
    trace_batch = 16
    evaluator = HessianTraceEvaluator(graph, dt, seed, burn_in, trace_batch)

    seq_len = 10
    traj = np.random.normal(size=(50, 4))
    starts = np.array([0, 10, 20])
    worm_id = 0

    # ACT
    trace, n_nan = evaluator(traj, worm_id, starts, seq_len)

    # ASSERT
    assert len(trace) == 24
    assert np.allclose(trace, 8.0)
    assert n_nan == 0
