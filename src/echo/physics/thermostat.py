import math
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray

class Thermostat(eqx.Module):
    """
    Enforces the Fluctuation-Dissipation Theorem by integrating deterministic
    physics with stochastic environmental noise over a continuous time step (dt)
    using the Euler-Maruyama method.
    """
    temperature: float = eqx.field(static=True)

    def __init__(self, temperature: float = 1.0):
        """
        Initializes the Thermostat module.
        
        Args:
            temperature: Scalar temperature T. Default is 1.0.
        """
        self.temperature = float(temperature)

    def __call__(
        self,
        x: Float[Array, "d_state"],
        grad_E: Float[Array, "d_state"],
        Q: Float[Array, "d_state d_state"],
        L: Float[Array, "d_state d_state"],
        dt: float,
        key: PRNGKeyArray,
        omega_ext: jax.Array | None = None,
        q_ext: jax.Array | None = None
    ) -> Float[Array, "d_state"]:
        """
        Computes the next state using the Euler-Maruyama method.
        
        Args:
            x: The current biological state vector.
            grad_E: The pre-computed gradient vector of the energy landscape at x.
            Q: Skew-symmetric matrix from the SolenoidalFlow module.
            L: Lower-triangular Cholesky factor from the DissipativeFriction module.
            dt: Scalar continuous time step.
            key: PRNG key for generating Wiener process noise.
            omega_ext: Optional external thermodynamic force vector to apply as an explicit perturbation to the deterministic drift.
            q_ext: Optional exogenous actuation signal to apply to the deterministic drift.
            
        Returns:
            The next state vector of shape (d_state,).
        """
        # Ensure floating point type for safety
        dt_jnp = jnp.array(dt, dtype=jnp.float32)
        T = jnp.array(self.temperature, dtype=jnp.float32)

        # 1. Compute Gamma
        Gamma = L @ L.T
        
        # 2. Deterministic drift: -(Q - Gamma) @ grad_E
        drift = -(Q - Gamma) @ grad_E
        
        drift_total = drift + (omega_ext if omega_ext is not None else 0.0) + (q_ext if q_ext is not None else 0.0)
        
        # 3. Stochastic diffusion (noise): sqrt(2 * T * dt) * (L @ dW)
        dW = jax.random.normal(key, shape=x.shape, dtype=jnp.float32)
        diffusion = jnp.sqrt(2.0 * T * dt_jnp) * (L @ dW)
        
        # 4. Final Euler-Maruyama update
        x_next = x + (drift_total * dt_jnp) + diffusion
        
        return x_next
