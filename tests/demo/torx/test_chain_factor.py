import jax
import jax.numpy as jnp
from torx import DFG, Site, ChainFactor, PortSpec
from torx.factor import AbstractReferenceFactor
import equinox as eqx

class DummyFactor(AbstractReferenceFactor):
    input_ports: dict[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)
    
    def __init__(self):
        self.input_ports = {
            "micro": jax.ShapeDtypeStruct((16,), jnp.float32),
            "macro": jax.ShapeDtypeStruct((2,), jnp.float32)
        }
        self.output_spec = jax.ShapeDtypeStruct((2,), jnp.float32)
        
    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        # By default torx passes the full exogenous tensor instead of slicing it
        out = inputs["macro"] + jnp.sum(inputs["micro"])
        return (out, None) if return_aux else out
        
    def init_params(self, key):
        return None

def test_chain_factor_exogenous_broadcasting():
    f = DummyFactor()
    cf = ChainFactor(base=f, n_steps=10, feedback_porting_fn="macro", weight_tied=True)
    g = DFG(
        sites=(
            Site(
                name="chain",
                factor=cf,
                parents=("env_micro", "env_macro"),
                porting_fn=("micro", "macro"),
                param_key=None, info_key=None, site_info=None
            ),
        ),
        input_ports={
            "env_micro": jax.ShapeDtypeStruct((10, 16), jnp.float32),
            "env_macro": jax.ShapeDtypeStruct((2,), jnp.float32)
        },
        output_name="chain"
    )
    
    key = jax.random.key(0)
    inputs = {
        "env_micro": jnp.ones((10, 16)),
        "env_macro": jnp.zeros((2,))
    }
    
    out = g.sample(key, inputs=inputs, params={})
    
    # 10 * 16 * 1 = 160 per step
    # After 10 steps, 160 * 10 = 1600
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([1600., 1600.]))
