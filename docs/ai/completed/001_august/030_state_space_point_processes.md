Target File: src/pipeline/sim2real/rna_dataloader.py

Context: We need to fundamentally alter how we handle sparse transcriptomic
reads. The current `align_to_master_clock` method injects 12-D Poisson RNA
counts on the 5-minute mark and pads the rest of the 500Hz grid with NaNs. We
are moving to a Deep Continuous-Time State-Space Point Process (S2P2)
architecture.

Task: Rewrite the data alignment logic to utilize a Marked Temporal Point
Process instead of discrete NaN grids.

1. Create a `StateSpacePointProcess` class that treats each of the 12 Waddington
   RNA anchor flashes as discrete events occurring in continuous time.
2. Instead of returning a Pandas DataFrame full of NaNs, generate and return an
   Event Tensor of shape [Num_Events, 3], where dimension 0 is the exact
   continuous time of the flash (t), dimension 1 is the RNA anchor index (1 to
   12), and dimension 2 is the observed transcriptomic intensity λ(t).
3. Update the Waddington crash simulation (at the 10-minute mark) to manifest as
   an intense compounding of the baseline hazard rate for the panic genes (TP53,
   IL6, CASP3), causing an explosion of asynchronous event markers rather than a
   single discrete spike.
4. Include basic unit tests

Code start:

import numpy as np import pandas as pd import torch

class StateSpacePointProcess: """ Models discrete transcriptomic flashes as a
continuous-time Marked Point Process. Instead of dense grids filled with NaNs,
it generates an asynchronous event stream. """ def **init**(self, anchor_genes,
base_expression, crash_minute=10): self.anchor_genes = anchor_genes
self.base_expression = base_expression self.crash_minute = crash_minute
self.panic_genes = {"TP53", "IL6", "CASP3"}

    def generate_event_stream(self, total_minutes=15):
        """
        Generates a tensor of shape [Num_Events, 3].
        Dim 0: Time (ms)
        Dim 1: Gene Index
        Dim 2: Intensity λ(t)
        """
        events = []

        for gene_idx, gene in enumerate(self.anchor_genes):
            # Base hazard rate (events per minute)
            base_lambda = self.base_expression.get(gene, 1.0)

            # --- PRE-CRASH: Homeostatic Poisson Process ---
            t_min = 0.0
            while t_min < self.crash_minute:
                # Time until next transcriptomic flash (Exponential distribution)
                dt = np.random.exponential(1.0 / max(base_lambda, 0.1))
                t_min += dt

                if t_min < self.crash_minute:
                    intensity = np.random.poisson(base_lambda) + 1
                    events.append([t_min * 60000.0, float(gene_idx), float(intensity)])

            # --- POST-CRASH: Waddington Variance Explosion ---
            t_min = self.crash_minute

            # Panic genes experience a massive compounding hazard rate
            if gene in self.panic_genes:
                crash_lambda = base_lambda * 15.0 # Explosion of stress alarms
            elif gene == "NFE2L2":
                crash_lambda = base_lambda * 0.1  # Protective genes shut down
            else:
                crash_lambda = base_lambda

            while t_min < total_minutes:
                dt = np.random.exponential(1.0 / max(crash_lambda, 0.1))
                t_min += dt

                if t_min < total_minutes:
                    intensity = np.random.poisson(crash_lambda) + 1
                    events.append([t_min * 60000.0, float(gene_idx), float(intensity)])

        if not events:
            return torch.empty((0, 3))

        # Sort all asynchronous events chronologically
        events.sort(key=lambda x: x[0])
        return torch.tensor(events, dtype=torch.float32)

class TranscriptomicLoader: def **init**(self, crash_minute=10):
self.crash_minute = crash_minute self.source_url =
"s3://czb-cellxgene/wyss-coray-microglia-aging.h5ad"

        self.anchor_genes = [
            "NFE2L2", "TP53", "CDKN2A", "TREM2",
            "APOE", "IL6", "GFAP", "MAPT",
            "NANOG", "CASP3", "CAS13", "GAPDH",
        ]
        print(f"[INIT] Psi S2P2 Loader targeting proxy: {self.source_url}")

    def fetch_and_filter_h5ad(self):
        print("[NETWORK] Mocking lazy stream from .h5ad AnnData store...")
        base_expression = {
            "NFE2L2": 12.0, "TP53": 2.0, "CDKN2A": 0.5, "TREM2": 8.0,
            "APOE": 25.0, "IL6": 1.0, "GFAP": 40.0, "MAPT": 30.0,
            "NANOG": 0.1, "CASP3": 2.0, "CAS13": 15.0, "GAPDH": 150.0,
        }
        return base_expression

    def build_continuous_event_tensor(self, total_minutes=15):
        """
        Replaces the old 'NaN' array logic with a continuous Marked Point Process.
        """
        base_expression = self.fetch_and_filter_h5ad()
        point_process = StateSpacePointProcess(
            anchor_genes=self.anchor_genes,
            base_expression=base_expression,
            crash_minute=self.crash_minute
        )
        event_tensor = point_process.generate_event_stream(total_minutes=total_minutes)
        return event_tensor

# ==========================================

# UNIT TESTS

# ==========================================

def run_tests(): print("\n--- Running S2P2 Unit Tests ---") loader =
TranscriptomicLoader(crash_minute=10) event_tensor =
loader.build_continuous_event_tensor(total_minutes=15)

    print(f"Generated Event Tensor Shape: {event_tensor.shape}")

    # Test 1: Shape Validation
    assert event_tensor.ndim == 2, "Tensor must be 2D"
    assert event_tensor.shape[1] == 3, "Tensor must have exactly 3 features: [Time, Gene_Idx, Intensity]"

    # Test 2: Chronological Ordering
    times = event_tensor[:, 0].numpy()
    assert np.all(np.diff(times) >= 0), "Events are not strictly chronological!"

    # Test 3: Waddington Crash Compounding Logic
    # Verify panic gene (e.g., TP53, index 1) fires much more frequently after minute 10
    tp53_idx = 1.0
    pre_crash_tp53 = event_tensor[(event_tensor[:, 0] < 10 * 60000) & (event_tensor[:, 1] == tp53_idx)]
    post_crash_tp53 = event_tensor[(event_tensor[:, 0] >= 10 * 60000) & (event_tensor[:, 1] == tp53_idx)]

    pre_crash_rate = len(pre_crash_tp53) / 10.0 # events per min
    post_crash_rate = len(post_crash_tp53) / 5.0 # events per min

    print(f"TP53 Firing Rate (Pre-Crash):  {pre_crash_rate:.1f} events/min")
    print(f"TP53 Firing Rate (Post-Crash): {post_crash_rate:.1f} events/min")

    assert post_crash_rate > pre_crash_rate * 5, "Panic gene hazard rate did not explode after crash!"
    print("All tests passed! S2P2 continuous pipeline verified.\n")

if **name** == "**main**": run_tests()

    # Execution Demo
    loader = TranscriptomicLoader(crash_minute=10)
    tensor = loader.build_continuous_event_tensor()
    print("Sample Event Stream [Time_ms, Gene_Index, Intensity]:")
    print(tensor[:5]) # Show first 5 events
