# Demo Scripts Recap

The `src/demo` folder contains a suite of conceptual scripts designed to highlight the core capabilities and thermodynamic principles of the Mamba-2 / State Space architecture in action. 

Here is a breakdown of what each of the core 4 demo scripts does:

## [01_bio_blade_engine.py]
**Focus: High-Throughput Ingestion & Scoring**
Demonstrates the raw processing power of the architecture. It simulates the high-throughput ingestion of massive, continuous electrophysiology data (like the MaxWell HD-MEA telemetry). The engine rapidly digests continuous streams and calculates real-time Thermodynamic Metrics (like KSM/CSD scores) to detect critical biological events at breakneck speeds.

## [02_indestructible_edge.py]
**Focus: Fault Tolerance & Self-Healing**
Simulates catastrophic hardware failures during inference, such as massive sensor dropouts or corrupted inputs. This script proves that the network possesses self-healing routing capabilities. Instead of crashing or producing NaNs, the architecture gracefully maintains its internal state representation and continues accurate inference despite the missing or corrupted sensory data.

## [03_multimodal_autopsy.py]
**Focus: Causality & Explainability (LRP)**
Acts as a forensic tool using Layer-wise Relevance Propagation (LRP). It demonstrates the architecture's ability to trace backward in time from a catastrophic event (e.g., a "Structural Collapse" of the organoid) to uncover the latent root cause (such as an early "RNA Stress Alarm") hidden deep within complex, multimodal time-series data. 

## [04_masked_state_space_model.py]
**Focus: Core Algorithm (Mask-Aware SSM)**
A localized demonstration of the underlying mathematical engine. This script shows how the `MaskAwareMambaCell` handles sparse and intermittent multi-modal sensor data. It proves that the model dynamically gates its continuous state transitions based on the availability of sensors, ensuring that the latent state space only updates when valid thermodynamic information is present.
