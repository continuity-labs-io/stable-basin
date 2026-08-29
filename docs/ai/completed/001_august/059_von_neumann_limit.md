# 1. The Von Neumann Limit (The IOPS Crisis)

**Status:** Completed

## 1. The Conceptual Shift

We must stop claiming we are solving a "PCIe Bandwidth" problem. At ~327 MB/s,
20kHz telemetry barely tickles a 32 GB/s PCIe 4.0 bus. The CPU dies from
context-switching.

## 2. The Narrative

The edge-compute architecture uses DPDK/eBPF kernel bypass to switch the NIC
from an interrupt-driven to a polling architecture, dedicating a CPU core to
spin-wait and push telemetry directly to VRAM without choking the OS scheduler.

## 3. The Repo Fix (Implementation Plan)

Update benchmarks to reflect the true bottleneck (IOPS vs Bandwidth).

### Action Items

- [x] Change the benchmark outputs from "Cloud Bandwidth Eq." to "Interrupts
      Bypassed / IOPS" (which sits at nearly 2 million per second for a 96-well
      plate).

### Targeted Files

- `3_telemetry_matrix_bench.py`
- `01_bio_blade_engine.py`
