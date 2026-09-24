import numpy as np

class HessianTraceEvaluator:
    """Forced unroll over fixed windows, then exact Hessian trace of the joint energy.

    Common random numbers: x_init and PRNG keys depend only on (worm_id, window), so
    clean and degraded versions of the same worm see identical noise.
    """

    def __init__(self, graph, dt: float, seed: int, burn_in: int, trace_batch: int):
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        self.graph, self.seed, self.burn_in, self.trace_batch = graph, seed, burn_in, trace_batch
        self.d_state = graph.d_micro + graph.d_macro
        d_micro = graph.d_micro

        @eqx.filter_jit
        def unroll(g, x_inits, seqs, keys):
            return jax.vmap(lambda xi, s, k: g.forced_unroll(k, xi, dt, seq=s))(x_inits, seqs, keys)

        @eqx.filter_jit
        def traces(ff, X):
            def energy(x):
                return ff.joint_energy_fn(x[:d_micro], x[d_micro:])

            return jax.vmap(lambda x: jnp.trace(jax.hessian(energy)(x)))(X)

        self._unroll, self._traces = unroll, traces

    def __call__(self, traj: np.ndarray, worm_id: int, starts: np.ndarray, seq_len: int):
        jax, jnp = self.jax, self.jnp
        seqs = np.stack([traj[s : s + seq_len] for s in starts]).astype(np.float32)
        rngs = [np.random.default_rng([self.seed, worm_id, w]) for w in range(len(starts))]
        x_inits = np.stack([0.01 * r.standard_normal(self.d_state) for r in rngs]).astype(np.float32)
        base = jax.random.PRNGKey(self.seed)
        keys = jnp.stack([jax.random.fold_in(base, worm_id * 10_000 + w) for w in range(len(starts))])

        states = self._unroll(self.graph, jnp.asarray(x_inits), jnp.asarray(seqs), keys)
        X = states[:, self.burn_in :, :].reshape(-1, self.d_state)
        out = []
        for i in range(0, X.shape[0], self.trace_batch):
            chunk = X[i : i + self.trace_batch]
            pad = self.trace_batch - chunk.shape[0]
            if pad:
                chunk = jnp.concatenate([chunk, jnp.repeat(chunk[-1:], pad, axis=0)], axis=0)
            out.append(np.asarray(self._traces(self.graph.flow_factor, chunk))[: self.trace_batch - pad])
        t = np.concatenate(out)
        finite = np.isfinite(t)
        return t[finite], int((~finite).sum())
