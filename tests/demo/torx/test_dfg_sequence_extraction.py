import jax
import jax.numpy as jnp
from torx import DFG, Site, ChainFactor
from torx.factor import AbstractReferenceFactor
import equinox as eqx

class DummyFactor(AbstractReferenceFactor):
    input_ports = {"h": jax.ShapeDtypeStruct((1,), jnp.float32)}
    output_spec = jax.ShapeDtypeStruct((1,), jnp.float32)
    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        out = inputs["h"] + 1.0
        return (out, out) if return_aux else out
    def init_params(self, key): 
        return None

def test_dfg_sequence_extraction():
    cf = ChainFactor(base=DummyFactor(), n_steps=5, feedback_porting_fn="h", weight_tied=True)
    g = DFG(
        sites=(Site("cf", cf, ("env_h",), ("h",), None, None, None),),
        input_ports={"env_h": jax.ShapeDtypeStruct((1,), jnp.float32)},
        output_name="cf"
    )
    key = jax.random.key(0)
    out = g.sample(key, {"env_h": jnp.zeros(1)}, {}, return_aux=True)
    
    assert isinstance(out, tuple)
    assert len(out) == 2
    
    final, aux = out
    assert jnp.allclose(final, jnp.array([5.0]))
    
    assert isinstance(aux, tuple)
    aux_sequence = aux[0]
    expected_aux = jnp.array([[1.], [2.], [3.], [4.], [5.]])
    assert jnp.allclose(aux_sequence, expected_aux)
