# 3. Shannon's Limit (The H-SSM Imperative)

**Status:** Completed

## 1. The Conceptual Shift

Mamba-2's $\mathcal{O}(1)$ memory is a fixed capacity constraint
($d_{state} = 64$). It cannot compress an infinite 1.38 trillion-frame sequence
(800 days at 20kHz) without catastrophic forgetting.

## 2. The Narrative

The architecture must be explicitly defined as a two-tier cascade. Layer 1
(Fast-Mamba) ingests 20kHz HD-MEA data and outputs 1Hz "Macroscopic Kinetic
Tokens" (e.g., PyDMD KSM eigenvalues). Layer 2 (Slow-Mamba) operates entirely on
the 1Hz Macroscopic tokens, effortlessly compressing months of continuous
recording into the 64-D state without violating Shannon's limit.

## 3. The Repo Fix (Implementation Plan)

We formally elevate `hierarchical_ssm.py` from a "demo" to the Core Temporal
Scaffold.

### Action Items

- [x] Update documentation and references to elevate `hierarchical_ssm.py` to
      the "Core Temporal Scaffold".
- [x] Ensure the architecture description accurately reflects the two-tier
      cascade (Fast-Mamba / Slow-Mamba).

### Targeted Files

- `hierarchical_ssm.py`
