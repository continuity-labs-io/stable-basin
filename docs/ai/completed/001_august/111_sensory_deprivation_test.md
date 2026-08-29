### Test 2: The Sensory Deprivation Test

_Goal: Prove that a continuous flow of thermodynamic energy (noise) is required
to maintain the boundary. If we drop the external heat bath to absolute zero,
the system loses the energetic tension required to maintain the Markov Blanket,
and the Observer will "flatline" into thermodynamic equilibrium._

**File paths to feed into context:** `src/models/vessel/vessel_baseline.py`
(Read-only reference) `tests/models/vessel/test_sensory_deprivation.py` (Create
this file)

**Raw text prompt to execute:** Please create
`tests/models/vessel/test_sensory_deprivation.py` by cloning the core logic from
`vessel_baseline.py`. We are running the Sensory Deprivation Test to prove that
the Observer requires external thermodynamic noise to maintain its boundary and
internal dynamics.

Modify the baseline code with the following changes:

1. **The Perturbation:** Begin the simulation with the normal, noisy Heat Bath.
   After a short stabilization period (e.g., frame 200), abruptly drop the
   external noise parameter (σdW) in the Heat Bath zone to absolute 0.0.
2. **The Physics Response:** Without the external noise pressing against the
   Cell Wall, the reaction-diffusion dynamics should lose their energetic
   tension. The internal resonating state of the Observer will slowly decay to a
   uniform, dead equilibrium (0.0 variance).
3. **The Output Metric:** Modify the Matplotlib right-hand panel to clearly mark
   the exact frame where the "Noise Cutoff" occurs via a vertical red dashed
   line. Plot the Internal Variance (Entropy) over time. The graph must show a
   healthy, low-amplitude oscillation (a "heartbeat") that slowly flatlines into
   a dead state after the cutoff. Include a `__main__` block to run the
   simulation natively.
