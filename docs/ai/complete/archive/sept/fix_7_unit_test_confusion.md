we are having multiple test modules with confused focus. For example 
the test imports two separate src modules and tests them separately.
If they were tested in integration this is in theory fine. 
so if and only if there is a clean separate then this is a bad smell and we should 
better organize the unit test cases. 

for example

`from src.echo.physics.thermostat import Thermostat
from src.echo.architecture.observer import MarkovBlanketObserver`

then test 1 only tests the Thermostat. and test 2 only test the observer.

## Implementation Plan
1. Move `test_thermostat_omega_ext` from `test_contention_forces.py` to `test_thermostat.py`.
2. Move `test_forced_thermalizer_omega_seq` from `test_contention_forces.py` to `test_observer.py`.
3. Delete `test_contention_forces.py` as it will be empty.
4. Clean up duplicate and unused imports (like `Thermostat` in `test_observer.py` and `MarkovBlanketObserver` in `test_thermostat.py`).
