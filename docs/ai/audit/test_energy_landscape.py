"""
tests/test_energy_landscape.py

Pins the contract of src/echo/metrics/energy_landscape.py:
  - trace-only and full-spectrum paths report the same trace
  - the trace is right on an analytic quadratic
  - new weights do not trigger a recompile (energy passed as a PyTree)
  - chunked evaluation equals unchunked evaluation
  - non-finite traces are dropped / kept / raised, never imputed
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest

from src.echo.metrics.energy_landscape import (
    ScalarEnergy,
    batch_calculate_curvature,
    batch_hessian_trace,
    calculate_curvature,
    curvature_over_states,
    hessian_trace,
)
from src.echo.primitives.ebm import PrecisionWeightedEBM

D = 46


def _states(n, seed=0, d=D):
    return jax.random.normal(jax.random.PRNGKey(seed), (n, d))


def _pwebm(seed, d=D):
    return PrecisionWeightedEBM(d_state=d, hidden_size=32, depth=2, key=jax.random.PRNGKey(seed))


class Quadratic(eqx.Module):
    A: jax.Array

    def __call__(self, x):
        return 0.5 * x @ self.A @ x


_TRACE_EVENTS = [0]


class CountingEnergy(eqx.Module):
    """Python body only runs while JAX traces, so the counter counts (re)compiles."""

    w: jax.Array

    def __call__(self, x):
        _TRACE_EVENTS[0] += 1
        return jnp.sum(jnp.tanh(self.w * x) ** 2)


def test_trace_paths_agree_and_match_eigen_sum():
    energy = ScalarEnergy(_pwebm(1))
    X = _states(64)
    t_fast = batch_hessian_trace(energy, X)
    full = batch_calculate_curvature(energy, X)
    np.testing.assert_allclose(t_fast, full["hessian_trace"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(full["hessian_trace"], full["eigenvalues"].sum(-1), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(hessian_trace(energy, X[0]), calculate_curvature(energy, X[0])["hessian_trace"], rtol=1e-6)


def test_quadratic_is_exact():
    M = jax.random.normal(jax.random.PRNGKey(2), (D, D))
    A = M @ M.T / D + 0.1 * jnp.eye(D)
    res = calculate_curvature(Quadratic(A), _states(1)[0])
    np.testing.assert_allclose(res["hessian_trace"], jnp.trace(A), rtol=1e-5)
    np.testing.assert_allclose(res["min_eigenvalue"], jnp.linalg.eigvalsh(A)[0], rtol=1e-4)
    assert int(res["n_negative"]) == 0


def test_indefinite_hessian_is_flagged():
    A = jnp.diag(jnp.concatenate([jnp.full(D - 1, 1.0), jnp.array([-50.0])]))
    res = calculate_curvature(Quadratic(A), _states(1)[0])
    assert float(res["hessian_trace"]) < 0  # one steep negative direction swamps 45 positive ones
    assert int(res["n_negative"]) == 1


def test_new_weights_do_not_recompile():
    X = _states(16)
    batch_hessian_trace(CountingEnergy(jnp.ones(D)), X).block_until_ready()
    after_first = _TRACE_EVENTS[0]
    assert after_first > 0
    batch_hessian_trace(CountingEnergy(2.0 * jnp.ones(D)), X).block_until_ready()
    assert _TRACE_EVENTS[0] == after_first


def test_new_weights_are_actually_used():
    X = _states(8)
    a = batch_hessian_trace(ScalarEnergy(_pwebm(3)), X)
    b = batch_hessian_trace(ScalarEnergy(_pwebm(4)), X)
    assert not np.allclose(a, b)


def test_chunked_equals_unchunked():
    energy = ScalarEnergy(_pwebm(5))
    X = _states(1000, seed=6)
    whole = np.asarray(batch_hessian_trace(energy, X))
    res = curvature_over_states(energy, X, chunk_size=256)
    assert res["hessian_trace"].shape == (1000,)
    np.testing.assert_allclose(res["hessian_trace"], whole, rtol=1e-6, atol=1e-6)
    full = curvature_over_states(energy, X, chunk_size=300, full_spectrum=True)
    assert full["eigenvalues"].shape == (1000, D)


class NanWhenNegative(eqx.Module):
    def __call__(self, x):
        return jnp.sqrt(x[0]) * jnp.sum(x**2)  # NaN wherever x[0] < 0


def test_nonfinite_policy():
    X = _states(200, seed=7)
    n_bad = int(np.sum(np.asarray(X[:, 0]) < 0))
    assert 0 < n_bad < 200

    dropped = curvature_over_states(NanWhenNegative(), X, chunk_size=64)
    assert dropped["n_nonfinite"] == n_bad
    assert dropped["hessian_trace"].shape == (200 - n_bad,)
    assert np.all(np.isfinite(dropped["hessian_trace"]))

    kept = curvature_over_states(NanWhenNegative(), X, chunk_size=64, nonfinite="keep")
    assert kept["hessian_trace"].shape == (200,)
    assert int((~kept["finite_mask"]).sum()) == n_bad

    with pytest.raises(ValueError):
        curvature_over_states(NanWhenNegative(), X, nonfinite="raise")


def test_scalar_energy_applies_hull():
    from src.echo.architecture.markov_hull import MarkovHull

    hull = MarkovHull(d_internal=2, d_sensory=2, d_active=1, d_external=1, D_s=jnp.zeros((2, 2)))
    ebm = _pwebm(8, d=6)
    x = jnp.arange(6.0)
    expected = ebm(hull.apply_sensory_degradation(x))[0]
    np.testing.assert_allclose(ScalarEnergy(ebm, hull)(x), expected, rtol=1e-6)


def test_joint_ebm_class_is_stable_and_matches_joint_energy():
    """Needs the full repo (torx). Guards the predictive_coding_graph.patch fix."""
    pytest.importorskip("torx")
    from src.echo.architecture.observer import MarkovBlanketObserver
    from src.echo.architecture.predictive_coding_graph import PredictiveCodingGraph

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    kw = dict(ebm_hidden_size=8, ebm_depth=1, n_steps=1, temperature=1.0)
    micro = MarkovBlanketObserver(2, 2, 1, 1, key=k1, **kw)
    macro = MarkovBlanketObserver(2, 1, 1, 1, key=k2, **kw)
    graph = PredictiveCodingGraph(micro, macro, n_steps=1, key=k3)

    assert type(graph.ebm) is type(graph.ebm)
    x = _states(1, d=graph.d_micro + graph.d_macro)[0]
    direct = graph.flow_factor.joint_energy_fn(x[: graph.d_micro], x[graph.d_micro :])
    np.testing.assert_allclose(ScalarEnergy(graph.ebm)(x), direct, rtol=1e-6)
