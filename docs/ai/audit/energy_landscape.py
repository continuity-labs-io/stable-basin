"""
src/echo/metrics/energy_landscape.py

Single source of truth for energy-landscape curvature. Every Hessian in the repo
should be computed here; harnesses decide WHICH states to evaluate, this module
decides HOW curvature is measured.

Point functions (jitted)
    calculate_curvature(energy_fn, x)        full spectrum: trace, eigenvalues, rank,
                                             nullity, min eigenvalue, # negative
    hessian_trace(energy_fn, x)              trace only, no eigendecomposition
    batch_calculate_curvature(energy_fn, X)  vmapped full spectrum   (API unchanged)
    batch_hessian_trace(energy_fn, X)        vmapped trace only

Driver (host side)
    curvature_over_states(energy_fn, X, ...) fixed-shape chunks (one compile), and an
                                             explicit non-finite policy: drop | keep | raise.
                                             There is deliberately no impute option.

Both paths define the trace as jnp.trace(H), so trace-only and full-spectrum runs
report the same number. At d = 46 the trace-only path skips a 46x46 eigensolve per
state (roughly 4-5x faster on CPU in a toy benchmark).

Passing the energy function
    Under eqx.filter_jit anything that is not an array is static and cached by
    identity. So pass a PyTree:
        ScalarEnergy(ebm)             any EBM whose __call__ returns (E, Pi)
        ScalarEnergy(ebm, hull)       same, with hull.apply_sensory_degradation first
    and not `lambda x: ebm(x)[0]`. A fresh lambda recompiles on every call, and a
    reused lambda keeps serving the weights it was first compiled with even after
    the model it closes over has changed.

Reading the trace
    The trace sums signed eigenvalues. Away from a minimum a learned MLP energy is
    often indefinite, and negative directions cancel positive ones, so "trace = basin
    steepness" only holds where the Hessian is positive semi-definite. Check
    min_eigenvalue / n_negative before interpreting a trace as steepness.
"""

from __future__ import annotations

from typing import Callable, Dict, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ScalarEnergy",
    "calculate_curvature",
    "hessian_trace",
    "batch_calculate_curvature",
    "batch_hessian_trace",
    "curvature_over_states",
]


class ScalarEnergy(eqx.Module):
    """PyTree adapter: turns an (E, Pi)-returning EBM into a scalar energy function.

    Because this is an eqx.Module, its weights are traced (not baked in), so one
    compiled program serves every model with the same architecture.
    """

    ebm: eqx.Module
    hull: eqx.Module | None = None

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.hull is not None:
            x = self.hull.apply_sensory_degradation(x)
        out = self.ebm(x)
        return out[0] if isinstance(out, tuple) else out


def _hessian(energy_fn: Callable, x: jax.Array) -> jax.Array:
    return jax.hessian(energy_fn)(x)  # forward-over-reverse


@eqx.filter_jit
def calculate_curvature(
    energy_fn: Callable, x: jax.Array, rank_tol: float = 1e-4
) -> Dict[str, jax.Array]:
    """
    Full curvature spectrum at a single state.

    Args:
        energy_fn: scalar energy function of a 1D state (prefer a ScalarEnergy).
        x: 1D state vector of shape (d_state,).
        rank_tol: absolute eigenvalue cutoff for rank / nullity. This is scale
            dependent: it depends on the units of E and x.

    Returns:
        hessian_trace    jnp.trace(H); identical definition to hessian_trace()
        eigenvalues      (d_state,) ascending, from eigvalsh
        hessian_rank     count of |lambda| > rank_tol
        hessian_nullity  d_state - rank
        min_eigenvalue   smallest eigenvalue; < 0 means H is indefinite here
        n_negative       count of eigenvalues < -rank_tol
    """
    H = _hessian(energy_fn, x)
    eigenvalues = jnp.linalg.eigvalsh(H)  # H is symmetric for a C^2 scalar field
    rank = jnp.sum(jnp.abs(eigenvalues) > rank_tol)
    return {
        "hessian_trace": jnp.trace(H),
        "eigenvalues": eigenvalues,
        "hessian_rank": rank,
        "hessian_nullity": eigenvalues.shape[0] - rank,
        "min_eigenvalue": eigenvalues[0],
        "n_negative": jnp.sum(eigenvalues < -rank_tol),
    }


@eqx.filter_jit
def hessian_trace(energy_fn: Callable, x: jax.Array) -> jax.Array:
    """Trace of the Hessian at a single state, without an eigendecomposition."""
    return jnp.trace(_hessian(energy_fn, x))


@eqx.filter_jit
def batch_calculate_curvature(
    energy_fn: Callable, x_seq: jax.Array, rank_tol: float = 1e-4
) -> Dict[str, jax.Array]:
    """Full spectrum over states of shape (N, d_state). Returns arrays with leading N."""
    return jax.vmap(lambda x: calculate_curvature(energy_fn, x, rank_tol))(x_seq)


@eqx.filter_jit
def batch_hessian_trace(energy_fn: Callable, x_seq: jax.Array) -> jax.Array:
    """Trace only over states of shape (N, d_state). Returns shape (N,)."""
    return jax.vmap(lambda x: jnp.trace(_hessian(energy_fn, x)))(x_seq)


def curvature_over_states(
    energy_fn: Callable,
    states,
    *,
    chunk_size: int = 2048,
    full_spectrum: bool = False,
    nonfinite: Literal["drop", "keep", "raise"] = "drop",
    rank_tol: float = 1e-4,
) -> Dict[str, np.ndarray | int]:
    """
    Curvature over many states, in fixed-shape chunks, returned as NumPy.

    The last chunk is padded by repeating its final row, so every chunk has the same
    shape and the jitted kernel compiles once. Padded rows are trimmed.

    Non-finite traces (NaN/inf) are never imputed:
        drop   rows with a non-finite trace are removed (default)
        keep   all rows returned; use result["finite_mask"]
        raise  ValueError if any trace is non-finite

    Returns a dict with "hessian_trace" (and the other spectrum keys if
    full_spectrum=True), plus "finite_mask" (for the rows before dropping) and
    "n_nonfinite".
    """
    X = jnp.asarray(states)
    if X.ndim != 2:
        raise ValueError(f"states must be 2D (N, d_state); got shape {X.shape}.")
    n = X.shape[0]
    if n == 0:
        raise ValueError("states is empty.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    chunk_size = min(chunk_size, n)
    pieces: list[Dict[str, np.ndarray]] = []
    for i in range(0, n, chunk_size):
        chunk = X[i : i + chunk_size]
        m = chunk.shape[0]
        if m < chunk_size:
            chunk = jnp.concatenate([chunk, jnp.repeat(chunk[-1:], chunk_size - m, axis=0)], axis=0)
        if full_spectrum:
            res = batch_calculate_curvature(energy_fn, chunk, rank_tol)
        else:
            res = {"hessian_trace": batch_hessian_trace(energy_fn, chunk)}
        pieces.append({k: np.asarray(v)[:m] for k, v in res.items()})

    out: Dict[str, np.ndarray | int] = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    finite = np.isfinite(out["hessian_trace"])
    n_bad = int((~finite).sum())
    if n_bad and nonfinite == "raise":
        raise ValueError(f"{n_bad} of {n} Hessian traces are non-finite.")
    if nonfinite == "drop":
        out = {k: v[finite] for k, v in out.items()}
    elif nonfinite not in ("keep", "raise"):
        raise ValueError(f"Unknown nonfinite policy: {nonfinite!r}")
    out["finite_mask"] = finite
    out["n_nonfinite"] = n_bad
    return out
