# Stable Basin Issues

## Engineering Tickets

### `MeldLoss` Ablation Tests

- **Issue**: The new MeldLoss functionality is untested and un-ablated on a
  large scale.
- **Task**: Run the loss ablation tests via `make loss-ablation` to get a sense
  of what loss function to use for biological homeostasis modeling.

### Generalize Predictive Coding Graph Architecture

- **Issue**: The current `PredictiveCodingGraph` and `HierarchicalThermoFlowFactor` modules in `src/echo/architecture/` are hardcoded to exactly 2 hierarchical levels (Micro and Macro). While functional, this does not represent a true arbitrary graph topology.
- **Task**: Evaluate the engineering effort required to generalize these modules to support complex, N-level architectures. This likely involves transitioning from explicit `micro_*`/`macro_*` attributes to a generic node-list or graph-traversal implementation for the physics engine and observers.
