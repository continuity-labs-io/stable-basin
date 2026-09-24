import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from src.echo.harness.trace_evaluator import HessianTraceEvaluator

class DummyEBM(eqx.Module):
    def __call__(self, x):
        return jnp.sum(x**2), None

class DummyGraph(eqx.Module):
    d_micro: int = 2
    d_macro: int = 2
    ebm: eqx.Module = DummyEBM()

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
