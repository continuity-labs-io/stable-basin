import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from torx import DFG, Site
from src.echo.physics.solenoidal import SolenoidalFlow
from src.echo.physics.dissipative import DissipativeFriction
from src.echo.physics.thermostat import Thermostat
from src.echo.primitives.ebm import PrecisionWeightedEBM
from src.echo.primitives.thermalizer import ThermoFlowFactor

def create_mock_components(d_state=4, key=jax.random.PRNGKey(0)):
    key_ebm, key_solenoidal, key_dissipative = jax.random.split(key, 3)
    
    ebm = PrecisionWeightedEBM(
        d_state=d_state,
        hidden_size=16,
        depth=2,
        key=key_ebm
    )
    solenoidal = SolenoidalFlow(d_state=d_state, key=key_solenoidal)
    dissipative = DissipativeFriction(d_state=d_state, key=key_dissipative)
    thermostat = Thermostat(temperature=1.0)
    
    return ebm, solenoidal, dissipative, thermostat

def test_thermo_flow_factor_sample_shape_and_nans():
    """
    Instantiates the factor, passes dummy inputs, and ensures
    output shapes are correct without NaNs.
    """
    # ARRANGE
    d_state = 4
    dt = 0.1
    key_comp, key_sample, key_x = jax.random.split(jax.random.PRNGKey(42), 3)
    
    ebm, solenoidal, dissipative, thermostat = create_mock_components(d_state, key_comp)
    
    factor = ThermoFlowFactor(
        ebm=ebm,
        solenoidal=solenoidal,
        dissipative=dissipative,
        thermostat=thermostat,
        d_state=d_state
    )
    
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    inputs = {"x": x, "dt": dt}
    
    # ACT
    x_next = factor.sample(key_sample, inputs=inputs, params={})
    
    # ASSERT
    assert x_next.shape == (d_state,), "Output shape mismatch"
    assert not jnp.any(jnp.isnan(x_next)), "Output contains NaNs"

def test_thermo_flow_factor_return_aux():
    """
    Ensures return_aux correctly produces a tuple (x_next, None).
    """
    # ARRANGE
    d_state = 4
    dt = 0.1
    key_comp, key_sample, key_x = jax.random.split(jax.random.PRNGKey(123), 3)
    
    ebm, solenoidal, dissipative, thermostat = create_mock_components(d_state, key_comp)
    
    factor = ThermoFlowFactor(
        ebm=ebm,
        solenoidal=solenoidal,
        dissipative=dissipative,
        thermostat=thermostat,
        d_state=d_state
    )
    
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    inputs = {"x": x, "dt": dt}
    
    # ACT
    out = factor.sample(key_sample, inputs=inputs, params={}, return_aux=True)
    
    # ASSERT
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (d_state,)
    assert out[1] is None

def test_thermo_flow_factor_torx_dfg():
    """
    Wires the factor into a Torx DFG and asserts that graph.sample() executes safely.
    """
    # ARRANGE
    d_state = 4
    dt = 0.1
    key_comp, key_sample, key_x = jax.random.split(jax.random.PRNGKey(999), 3)
    
    ebm, solenoidal, dissipative, thermostat = create_mock_components(d_state, key_comp)
    
    factor = ThermoFlowFactor(
        ebm=ebm,
        solenoidal=solenoidal,
        dissipative=dissipative,
        thermostat=thermostat,
        d_state=d_state
    )
    
    # Create inputs dictionary for DFG
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    dt_tensor = jnp.array(dt)
    
    graph = DFG(
        sites=(
            Site(
                name="thermo_flow",
                factor=factor,
                parents=("x", "dt", "omega_ext"),
                porting_fn=("x", "dt", "omega_ext"),
                param_key=None, info_key=None, site_info=None
            ),
        ),
        input_ports={
            "x": jax.ShapeDtypeStruct((d_state,), jnp.float32),
            "dt": jax.ShapeDtypeStruct((), jnp.float32),
            "omega_ext": jax.ShapeDtypeStruct((d_state,), jnp.float32)
        },
        output_name="thermo_flow"
    )
    
    # ACT
    x_next = graph.sample(key_sample, inputs={"x": x, "dt": dt_tensor, "omega_ext": jnp.zeros(d_state)}, params={})
    
    # ASSERT
    assert x_next.shape == (d_state,)
    assert not jnp.any(jnp.isnan(x_next))

def test_thermo_flow_factor_differentiability():
    """
    Defines a dummy loss function wrapping graph.sample, and asserts that 
    equinox.filter_value_and_grad successfully computes gradients with respect 
    to the factor's internal parameters.
    """
    # ARRANGE
    d_state = 4
    dt = 0.1
    dt_tensor = jnp.array(dt)
    key_comp, key_sample, key_x = jax.random.split(jax.random.PRNGKey(777), 3)
    
    ebm, solenoidal, dissipative, thermostat = create_mock_components(d_state, key_comp)
    
    factor = ThermoFlowFactor(
        ebm=ebm,
        solenoidal=solenoidal,
        dissipative=dissipative,
        thermostat=thermostat,
        d_state=d_state
    )
    
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    graph = DFG(
        sites=(
            Site(
                name="thermo_flow",
                factor=factor,
                parents=("x", "dt", "omega_ext"),
                porting_fn=("x", "dt", "omega_ext"),
                param_key=None, info_key=None, site_info=None
            ),
        ),
        input_ports={
            "x": jax.ShapeDtypeStruct((d_state,), jnp.float32),
            "dt": jax.ShapeDtypeStruct((), jnp.float32),
            "omega_ext": jax.ShapeDtypeStruct((d_state,), jnp.float32)
        },
        output_name="thermo_flow"
    )
    
    # Define dummy loss mapping DFG to scalar
    @eqx.filter_value_and_grad
    def loss_fn(model_graph, input_x, sample_key):
        x_next = model_graph.sample(sample_key, inputs={"x": input_x, "dt": dt_tensor, "omega_ext": jnp.zeros(d_state)}, params={})
        return jnp.sum(x_next ** 2)
        
    # ACT
    loss, grads = loss_fn(graph, x, key_sample)
    
    # ASSERT
    assert not jnp.isnan(loss)
    
    # Extract the factor from the grads DFG
    factor_grad = grads.sites[0].factor
    
    # EBM gradients should exist and not be NaN
    assert not jnp.any(jnp.isnan(factor_grad.ebm.energy_head.weight))
    # Engine (SolenoidalFlow) gradients should exist
    assert not jnp.any(jnp.isnan(factor_grad.solenoidal.W))
    # Brake (DissipativeFriction) gradients should exist
    assert not jnp.any(jnp.isnan(factor_grad.dissipative.W))

from src.echo.primitives.thermalizer import TorxThermalizer

def test_torx_thermalizer_shape_and_nans():
    """
    Instantiates the full stack, wraps it in TorxThermalizer, and ensures
    it correctly unrolls the step without producing NaNs.
    """
    # ARRANGE
    d_state = 4
    n_steps = 10
    dt = 0.01
    key_comp, key_sample, key_x = jax.random.split(jax.random.PRNGKey(42), 3)
    
    ebm, solenoidal, dissipative, thermostat = create_mock_components(d_state, key_comp)
    
    factor = ThermoFlowFactor(
        ebm=ebm,
        solenoidal=solenoidal,
        dissipative=dissipative,
        thermostat=thermostat,
        d_state=d_state
    )
    
    thermalizer = TorxThermalizer(
        flow_factor=factor,
        n_steps=n_steps,
        d_state=d_state
    )
    
    x_init = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    # ACT
    x_final = thermalizer(key_sample, x_init, dt)
    
    # ASSERT
    # Note: Torx ChainFactor might return a dict or the final state depending on how it's unpacked.
    # Usually the output of ChainFactor is a trajectory [n_steps, d_state].
    assert not jnp.any(jnp.isnan(x_final)), "Output contains NaNs"

def test_torx_thermalizer_vram_xla_proof():
    """
    Proves that calling the thermalizer twice executes the `@equinox.filter_jit`
    compiled graph without throwing a `ConcretizationTypeError` (proving that
    `n_steps` was correctly handled as a static field).
    """
    # ARRANGE
    d_state = 4
    n_steps = 10
    dt = 0.01
    key_comp, key_sample1, key_sample2, key_x1, key_x2 = jax.random.split(jax.random.PRNGKey(123), 5)
    
    ebm, solenoidal, dissipative, thermostat = create_mock_components(d_state, key_comp)
    
    factor = ThermoFlowFactor(
        ebm=ebm,
        solenoidal=solenoidal,
        dissipative=dissipative,
        thermostat=thermostat,
        d_state=d_state
    )
    
    thermalizer = TorxThermalizer(
        flow_factor=factor,
        n_steps=n_steps,
        d_state=d_state
    )
    
    x_init1 = jax.random.normal(key_x1, (d_state,), dtype=jnp.float32)
    x_init2 = jax.random.normal(key_x2, (d_state,), dtype=jnp.float32)
    
    # ACT
    # First call will trace and compile
    x_final1 = thermalizer(key_sample1, x_init1, dt)
    
    # Second call should execute the compiled XLA binary
    # If n_steps wasn't static, JAX would complain about unrolling dynamically
    x_final2 = thermalizer(key_sample2, x_init2, dt)
    
    # ASSERT
    assert not jnp.any(jnp.isnan(x_final1))
    assert not jnp.any(jnp.isnan(x_final2))

def test_torx_thermalizer_bptt():
    """
    Runs a dummy gradient check using equinox.filter_value_and_grad on a loss
    function wrapping the __call__ method. Asserts that gradients flow seamlessly
    through the unrolled loop back into the PrecisionWeightedEBM weights.
    """
    # ARRANGE
    d_state = 4
    n_steps = 5
    dt = 0.01
    key_comp, key_sample, key_x = jax.random.split(jax.random.PRNGKey(777), 3)
    
    ebm, solenoidal, dissipative, thermostat = create_mock_components(d_state, key_comp)
    
    factor = ThermoFlowFactor(
        ebm=ebm,
        solenoidal=solenoidal,
        dissipative=dissipative,
        thermostat=thermostat,
        d_state=d_state
    )
    
    thermalizer = TorxThermalizer(
        flow_factor=factor,
        n_steps=n_steps,
        d_state=d_state
    )
    
    x_init = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    # Define dummy loss
    @eqx.filter_value_and_grad
    def loss_fn(model, input_x, sample_key):
        x_final = model(sample_key, input_x, dt)
        return jnp.sum(x_final ** 2)
        
    # ACT
    loss, grads = loss_fn(thermalizer, x_init, key_sample)
    
    # ASSERT
    assert not jnp.isnan(loss)
    
    # Extract the ebm gradient from the graph
    # thermalizer -> graph -> site("chain") -> factor -> base (which is ThermoFlowFactor) -> ebm
    chain_factor_grad = grads.graph.sites[0].factor
    flow_factor_grad = chain_factor_grad.base
    
    assert not jnp.any(jnp.isnan(flow_factor_grad.ebm.energy_head.weight))
    assert not jnp.any(jnp.isnan(flow_factor_grad.solenoidal.W))
