import pytest
import jax
import jax.numpy as jnp
from src.echo.primitives.thermalizer import ThermoFlowFactor, TorxThermalizer, ForcedTorxThermalizer
from src.echo.primitives.ebm import GaussianEBM
from src.echo.physics.dissipative import DissipativeFriction
from src.echo.physics.solenoidal import SolenoidalFlow
from src.echo.physics.thermostat import Thermostat

def test_thermo_flow_factor_init_params():
    d_state = 4
    key = jax.random.PRNGKey(0)
    
    # ARRANGE
    ebm = GaussianEBM(d_state=d_state, hidden_size=8, depth=2, key=key)
    dissipative = DissipativeFriction(d_state=d_state, key=key)
    solenoidal = SolenoidalFlow(d_state=d_state, key=key)
    thermostat = Thermostat()
    
    factor = ThermoFlowFactor(ebm=ebm, dissipative=dissipative, solenoidal=solenoidal, thermostat=thermostat, d_state=d_state)
    
    # ACT
    params = factor.init_params(key)
    
    # ASSERT
    assert params == {}

def test_forced_torx_thermalizer_exceptions():
    d_state = 4
    key = jax.random.PRNGKey(0)
    
    # ARRANGE
    ebm = GaussianEBM(d_state=d_state, hidden_size=8, depth=2, key=key)
    dissipative = DissipativeFriction(d_state=d_state, key=key)
    solenoidal = SolenoidalFlow(d_state=d_state, key=key)
    thermostat = Thermostat()
    
    factor = ThermoFlowFactor(ebm=ebm, dissipative=dissipative, solenoidal=solenoidal, thermostat=thermostat, d_state=d_state)
    thermalizer = ForcedTorxThermalizer(flow_factor=factor, d_state=d_state, injection_start_idx=0)
    
    x_init = jnp.ones(d_state)
    dt = 0.1
    
    # ACT & ASSERT
    # 1. Test both seq and omega_seq as None should raise ValueError
    with pytest.raises(ValueError, match="Must provide either seq or omega_seq."):
        thermalizer(key=key, x_init=x_init, dt=dt, seq=None, omega_seq=None)
        
    # 2. Test factor_params=None branch (and dummy_seq via omega_seq)
    # This should pass without error
    omega_seq = jnp.ones((5, d_state))
    traj = thermalizer(key=key, x_init=x_init, dt=dt, seq=None, omega_seq=omega_seq, factor_params=None)
    assert traj.shape == (5, d_state)
