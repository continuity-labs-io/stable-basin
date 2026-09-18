# Stable Basin Issues

## Engineering Tickets

### Improve Unit Test Coverage for Outer Loop Scripts

- **Issue**: Our core physics and architecture modules have robust coverage, but our overall project coverage is stuck at ~58%. The outer loop code, such as `src/demo/` scripts and `src/harness/` runners, currently have 0% test coverage.
- **Task**: Write integration and unit tests for the training harnesses and demo scripts. This will ensure that our model runners do not silently fail or suffer from data leakage during large-scale sensor fusion experiments.

### `MeldLoss` Ablation Tests

- **Issue**: The new MeldLoss functionality is untested and un-ablated on a
  large scale.
- **Task**: Run the loss ablation tests via `make loss-ablation` to get a sense
  of what loss function to use for biological homeostasis modeling.

