# Design Document: Data-Driven System Identification and RL Combinatorial Simulator

## 1. The Combinatorial Bottleneck
The current bottleneck in longevity research is the combinatorial explosion of polypharmacy. With roughly 100 putative longevity interventions, testing all pairs requires ~5,000 trials, and testing trios requires ~160,000 trials. Executing these combinations in multi-year *in vivo* mammalian trials is logistically and financially impossible. The field requires a computational simulator to triage combinations *in silico*.

## 2. Data-Driven System Identification (SysId)
Traditional bottom-up molecular dynamics simulations are computationally intractable for entire tissues. Instead, we use Data-Driven System Identification to extract the macroscopic thermodynamic footprint of an intervention directly from high-frequency cellular perturbation data.

### 2.1 Extracting the Thermodynamic Drug Vector
We bypass the need for longitudinal lifespan data by leveraging short-term (e.g., 20-60 minute) high-frequency recordings (such as HD-MEA or optical imaging).

1. **The Baseline Manifold:** Ingest high-frequency telemetry of healthy, unperturbed tissue. The physics engine (via prediction-error minimization) autonomously learns the baseline physical matrices governing the tissue's homeostatic limit cycle: Friction ($\Gamma_{base}$) and Solenoidal Flow ($Q_{base}$).
2. **The Perturbed Manifold:** Ingest telemetry of the exact same tissue type immediately following an acute perturbation (e.g., the application of a small molecule). The engine calculates the new, altered matrices for the drugged state: $\Gamma_{drug}$ and $Q_{drug}$.
3. **The Subtraction (Vector Isolation):** Mathematically isolate the net physical impact of the intervention by calculating the delta between the states:
   * $\Delta \Gamma_{drug} = \Gamma_{drug} - \Gamma_{base}$
   * $\Delta Q_{drug} = Q_{drug} - Q_{base}$

This $\Delta$ tensor is the isolated "Drug Vector." It mathematically encodes exactly how a specific molecule alters the fundamental thermodynamics of the biological system.

## 3. The Reinforcement Learning (RL) Simulator
Using a library of Thermodynamic Drug Vectors extracted from public or proprietary perturbation datasets, we construct an *in silico* simulation environment (e.g., wrapped in a Farama `Gymnasium` API).

### 3.1 The Environment (The Digital Twin)
The environment initializes with the physics matrices of an aging biological baseline (where $\Gamma$ is degraded and the Hessian Trace is flattened). At each step, the engine advances forward in continuous time using Euler-Maruyama integration, subjected to continuous environmental noise ($\omega$) driving it toward a mathematical crash.

### 3.2 The Action Space (The Drug Library)
An RL agent (e.g., PPO, SAC) navigates a discrete action space of extracted drug vectors. The agent can choose to "inject" a drug, which mathematically adds the corresponding $\Delta \Gamma$ and $\Delta Q$ to the environment's current physics matrices for a set duration (simulating drug pharmacokinetics and half-life).

### 3.3 The Reward Function (Time-in-Basin)
The simulation is evaluated by real-time thermodynamic metrics. The agent is rewarded for maximizing **Time-in-Basin (TiB)**—maintaining a steep Effective Hessian Trace and keeping the Koopman Stability Metric (KSM) near 1.0 for as many chronological steps as possible.

## 4. Autonomous Protocol Discovery
When the RL agent explores the combinatorial space, it will autonomously discover which drugs synergize and which destroy the system (Thermodynamic Shear).

For example, injecting two drugs with contradicting thermodynamic vectors may cause the physics matrices to lose Positive Semi-Definiteness, shattering the attractor basin and instantly terminating the episode. Conversely, the agent may discover non-obvious, chronologically staggered dosing regimens (e.g., injecting Drug A, waiting precisely 48 simulated hours for topological settling, then injecting Drug B) that maximize the biological limit cycle. These RL-derived policies output the exact dosing protocols to be prioritized for real-world wet lab validation.
