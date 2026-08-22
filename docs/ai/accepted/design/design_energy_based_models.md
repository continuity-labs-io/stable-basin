Design Document: Stable Basin 2.0 (Torx Evaluation & Hierarchical Enslavement)

1. Overview and Philosophy

The Holy Grail: To understand and model how a dynamical system with hierarchical components exhibits multiple levels of control. Specifically, modeling how slow, macroscopic "order parameters" (the Observer) thermodynamically enslave fast, microscopic variables (the Vessel / Cells), thereby generating a hierarchical, dynamical Markov Blanket (instantiating Karl Friston’s Free Energy Principle and Hermann Haken’s Synergetics).

The Evaluation Pivot: We are evaluating Extropic's Torx framework as a natively probabilistic alternative to deterministic PyTorch workarounds (like custom Triton kernels and manual missing-data hacks). Torx uses Stochastic Differentiable Programming and Directed Factor Graphs (DFGs), which naturally handle missing sensor dropout via posterior sampling. Crucially, this is an evaluation of a paradigm, not a full rip-and-replace. Thermodynamic silicon remains aspirational vaporware until shipped at scale. We are proving Stable Basin's readiness for this architecture while maintaining our core deterministic GPU pathways.

2. Scope Definition

2.1 In Scope (Goals)

Hardware Backend Abstraction: Extend src/core/substrate.py to support Torx/JAX simulators alongside PyTorch (MPS/CUDA/CPU). This ensures we do not thrash the repository; we gracefully evaluate the new paradigm via a routing flag (e.g., --backend=torx).

Namespace Isolation: All Torx-specific explorations will be strictly isolated within a dedicated namespace (src/demo/torx/ and src/models/torx/) to prevent contaminating the core PyTorch codebase during evaluation. We will use descriptive file names rather than numbered prefixes.

Torx DFG Primitives: Explore and implement foundational examples demonstrating Torx's DFG, ChainFactor, and probabilistic sampling workflow.

DFG to Neural Network: Demonstrate taking a defined Torx DFG and mapping it onto a differentiable neural network layout at a single level of abstraction.

Observer Zero Porting: Transition the "Observer Zero" concept from an explicit 2D spatial Reaction-Diffusion model into a statistical, probabilistic Torx DFG.

Hierarchical Enslavement: Build a multi-tier DFG demonstrating order parameters, where top-down priors from a slow macro-level constrain a fast micro-level.

2.2 Out of Scope (Non-Goals)

Full Pipeline Rip-and-Replace: We will not delete or abandon our existing deterministic PyTorch continuous-time models (e.g., Mamba, SSMs). The Torx evaluation will sit alongside them.

Over-Indexing on Proprietary Torx Features: We will focus only on the subset of Torx tools required to achieve hierarchical enslavement.

2.3 Pros and Cons of the Torx Architecture

Pros:

Native Sensor Dropout Handling: Eradicates complex gating logic; missing variables are statistically marginalized or sampled over naturally.

Physics-Aligned: Maps exactly to the theoretical requirements of Markov Blankets.

Future-Proof: Code written in Torx today theoretically compiles directly onto zero-latency thermodynamic silicon when it ships.

Cons:

JAX vs. PyTorch Interoperability: Stable Basin is heavily PyTorch; Torx requires JAX. We will need to manage data boundary translations cleanly. JAX is an excellent ecosystem for differentiable physics, so this is an acceptable migration cost.

Hardware Vaporware Risk: The physical Z1 chips are not yet commercially available; we are reliant on GPU-accelerated simulators (extro-sim) for the near future. The initial simulation overhead is an acceptable tradeoff for testing the paradigm.

3. Phased Implementation Sequence
The implementation will follow a strict, four-phase sequence isolated within the new Torx namespace. Descriptive file names will be used in place of incremental numbering.

Phase 1: Torx Workflow & DFG Basics

Objective: Safely introduce the Torx dependency, implement the backend flag, and learn the workflow of Directed Factor Graphs.

Action: Extend src/core/substrate.py and create the src/demo/torx/ namespace.

Deliverable: A sandbox script (src/demo/torx/dfg_basics.py) that initializes a simple Torx circuit and runs it via the Torx simulators.

Phase 2: DFG to Neural Network (Single Level)

Objective: Prove we can embed neural network architectures within a DFG to learn complex transitions.

Action: Build a Torx DFG where the transition between states is governed by a parameterized neural network factor.

Deliverable: A script (src/demo/torx/dfg_neural_net.py) showing that we can train the parameters of this single-level DFG using Stochastic Differentiable Programming (via JAX/Equinox).

Phase 3: Porting Observer Zero (Macro-State Extraction)

Objective: Establish the macroscopic, slow-moving observer using causal graph topology rather than spatial PDE grids.

Action: Create a Torx DFG representing Observer Zero. This graph will ingest chaotic, high-frequency micro-states and use affine transformations/diffusions to encode a stable, long-term memory state.

Deliverable: A script (src/demo/torx/observer_zero.py) showing the Observer successfully extracting a stable macro-state from noisy inputs.

Phase 4: Hierarchical Enslavement (The Holy Grail)

Objective: Close the cybernetic loop. Prove that the macro-state (Observer) enslaves the micro-state (Vessel) via thermodynamic energy minimization.

Action: Fuse the fast micro-level and the slow macro-level into a single, hierarchical DFG.

Mechanism: Wire the output of Observer Zero (macro-state) back into the input ports of the Vessel (micro-state) as a top-down prior.

Deliverable: A visual dashboard (src/demo/torx/hierarchical_enslavement.py) proving that the slow, heavy mass of Observer Zero acts as an order parameter, mathematically restricting the degrees of freedom of the Vessel.
