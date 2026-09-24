"""
src/echo/harness/trace_evaluator.py   (replaces src/echo/metrics/hessian.py)

Harness, not a metric: decides WHICH states to measure (forced unroll over fixed
windows, burn-in, common random numbers) and delegates HOW curvature is measured to
src.echo.metrics.energy_landscape.
"""

from __future__ import annotations

import numpy as np

from src.echo.metrics.energy_landscape import ScalarEnergy, curvature_over_states


class HessianTraceEvaluator:
    """Forced unroll over fixed windows, then the Hessian trace of the joint energy.

    Common random numbers: x_init and PRNG keys depend only on (worm_id, window), so
    clean and degraded versions of the same worm see identical noise.
    """

    def __init__(self, graph, dt: float, seed: int, burn_in: int, trace_batch: int):
        import equinox as eqx
        import jax

        self.jax = jax
        self.graph, self.seed, self.burn_in, self.trace_batch = graph, seed, burn_in, trace_batch
        self.d_state = graph.d_micro + graph.d_macro
        # Requires JointEBM at module scope (see predictive_coding_graph.patch);
        # otherwise every access to graph.ebm is a new class and forces a recompile.
        self.energy = ScalarEnergy(graph.ebm)

        @eqx.filter_jit
        def unroll(g, x_inits, seqs, keys):
            return jax.vmap(lambda xi, s, k: g.forced_unroll(k, xi, dt, seq=s))(x_inits, seqs, keys)

        self._unroll = unroll

    def __call__(self, traj: np.ndarray, worm_id: int, starts: np.ndarray, seq_len: int):
        jax = self.jax
        seqs = np.stack([traj[s : s + seq_len] for s in starts]).astype(np.float32)
        rngs = [np.random.default_rng([self.seed, worm_id, w]) for w in range(len(starts))]
        x_inits = np.stack([0.01 * r.standard_normal(self.d_state) for r in rngs]).astype(np.float32)
        base = jax.random.PRNGKey(self.seed)
        keys = jax.numpy.stack([jax.random.fold_in(base, worm_id * 10_000 + w) for w in range(len(starts))])

        states = self._unroll(self.graph, x_inits, seqs, keys)
        X = states[:, self.burn_in :, :].reshape(-1, self.d_state)
        res = curvature_over_states(self.energy, X, chunk_size=self.trace_batch, nonfinite="drop")
        return res["hessian_trace"], res["n_nonfinite"]
