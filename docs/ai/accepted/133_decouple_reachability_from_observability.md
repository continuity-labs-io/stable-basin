# [UPDATE 001] DECOUPLE REACHABILITY FROM OBSERVABILITY (SILENT DRIFT)

**Importance**: CRITICAL (The Paradigm Shift).

This redefines aging in the codebase. Traditional ML assumes a network always
has perfect access to its state. Biology does not. By decoupling reachability
(can the cell fix it?) from observability (does the cell know it's broken?), we
allow ECHO to model "Sensory Blindness." The tissue ages not because it lacks
the physical strength to maintain homeostasis, but because its internal model is
rank-deficient; it doesn't know it's dying, so it generates zero prediction
error.

## 🔧 CORE CHANGES (Infrastructure & Math)

- `src/echo/architecture/markov_hull.py`: We must introduce an explicit
  observation mapping operator (a Sensory Degradation Matrix, $D_s$). When the
  true external state ($\eta$) hits the sensory boundary ($s$), it must pass
  through this matrix: $s = D_s \eta$. If $D_s$ drops rank, the internal state
  becomes blind to specific external dimensions.
- `src/echo/primitives/ebm.py`: The PrecisionWeightedEBM must be strictly
  decoupled so that its Precision output ($\Pi_\theta$) and energy calculations
  are a function of the degraded sensory state, not the omniscient true state.
- `src/echo/architecture/hierarchy.py`: The Joint Free Energy equation $F$ must
  calculate bottom-up surprisal (prediction error) based strictly on the
  observable state. If observability is 0, the gradient passed to the Macro
  level becomes exactly 0.0.

## 🎬 SCRIPTS & DEMOS (The Proof)

- `src/echo/benchmarks/silent_drift_benchmark.py`:
  - **The Scenario**: Initialize a tissue simulation where physical capacity
    (the $Q$ and $\Gamma$ matrices) is perfectly intact. Introduce a slow
    physical perturbation to the true state $x$.
  - **The Twist**: Artificially drop the rank of the Hull's sensory mask over
    time.
  - **The Metric**: The script plots the True State Divergence (which is rising)
    against the Observer's Surprisal (which remains flat at zero). This visually
    and mathematically proves "Silent Drift" (information-limited damage).
