**Context Files to Load / Modify:**
* `src/echo/benchmarks/06_worm_gait_decline.py`

**Task: Paper 1 Physics - Measure the Joint Thermodynamic Basin**
The benchmark script is incorrectly evaluating the Hessian trace of only the isolated Macro EBM, yielding a trace of `0.017` (which is mostly just the structural prior). The true Waddington basin (which reported `108.5` during training) is the Joint Free Energy landscape, where the Macro observer projects precision weights down to the Micro observer. We must evaluate the full hierarchical system.

**Core Objectives:**

**1. Extract Full System States:**
* Rename the `get_macro_states` function to `get_full_states`.
* Inside the function, change `macro_traj = traj[:, graph.d_micro:]` to simply `full_traj = traj`. We need the entire state vector (micro + macro) to evaluate the Joint EBM.
* Update all references in `main()` from `macro_states_*` to `full_states_*`.

**2. Track the Joint EBM:**
* Change the curvature tracker initialization in `main()` from:
  `tracker_A = HessianCurvatureTracker(graph_A.flow_factor.macro_ebm)`
  to:
  `tracker_A = HessianCurvatureTracker(graph_A.ebm)`
* Apply the exact same fix to `tracker_B` (change `graph_B.flow_factor.macro_ebm` to `graph_B.ebm`). 
* *Why:* This utilizes the `PredictiveCodingGraph.ebm` property, which computes the trace of the full 46x46 Hessian matrix of the Joint Free Energy landscape.

**Constraints:**
* Keep the rest of the file (metrics computations, JSON serialization, plotting, chunking) exactly the same. We just need to feed it the full states and the full joint EBM.
