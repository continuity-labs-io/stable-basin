import numpy as np
import pandas as pd
import torch

import logging

logger = logging.getLogger(__name__)


class StateSpacePointProcess:
    """
    Models discrete transcriptomic flashes as a continuous-time Marked Point Process.
    Instead of dense grids filled with NaNs, it generates an asynchronous event stream.
    """

    def __init__(self, anchor_genes, base_expression, crash_minute=10):
        self.anchor_genes = anchor_genes
        self.base_expression = base_expression
        self.crash_minute = crash_minute
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
                crash_lambda = base_lambda * 15.0  # Explosion of stress alarms
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


class PsiTranscriptomicLoader:
    def __init__(self, crash_minute=10):
        self.crash_minute = crash_minute
        self.source_url = "s3://czb-cellxgene/wyss-coray-microglia-aging.h5ad"

        self.anchor_genes = [
            "NFE2L2",
            "TP53",
            "CDKN2A",
            "TREM2",
            "APOE",
            "IL6",
            "GFAP",
            "MAPT",
            "NANOG",
            "CASP3",
            "CAS13",
            "GAPDH",
        ]
        logger.info(f"[INIT] Psi S2P2 Loader targeting proxy: {self.source_url}")

    def fetch_and_filter_h5ad(self):
        logger.info("[NETWORK] Mocking lazy stream from .h5ad AnnData store...")
        base_expression = {
            "NFE2L2": 12.0,
            "TP53": 2.0,
            "CDKN2A": 0.5,
            "TREM2": 8.0,
            "APOE": 25.0,
            "IL6": 1.0,
            "GFAP": 40.0,
            "MAPT": 30.0,
            "NANOG": 0.1,
            "CASP3": 2.0,
            "CAS13": 15.0,
            "GAPDH": 150.0,
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
            crash_minute=self.crash_minute,
        )
        event_tensor = point_process.generate_event_stream(total_minutes=total_minutes)
        return event_tensor


if __name__ == "__main__":
    # Execution Demo
    loader = PsiTranscriptomicLoader(crash_minute=10)
    tensor = loader.build_continuous_event_tensor()
    logger.info("Sample Event Stream [Time_ms, Gene_Index, Intensity]:")
    logger.info(tensor[:5])  # Show first 5 events
