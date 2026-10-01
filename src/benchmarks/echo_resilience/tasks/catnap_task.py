import logging
import numpy as np
from torch.utils.data import DataLoader

from src.benchmarks.echo_resilience.task_registry import AgingBenchmarkTask
from src.data.datasets import JAXDictDataset

try:
    from src.data.behavior.catnap_dataset import CatnapContinuousDataset
except ImportError:
    CatnapContinuousDataset = None

logger = logging.getLogger(__name__)


class CatnapTask(AgingBenchmarkTask):
    def get_raw_datasets(self, config) -> tuple:
        h5_path = config.get("dataset", {}).get("h5_path", "data/catnap/trace_features.h5")
        seq_len = config.get("dataset", {}).get("ebm_seq_len", 10)

        if CatnapContinuousDataset is None:
            raise ImportError("CatnapContinuousDataset could not be imported.")

        import pandas as pd
        from pathlib import Path
        h5_obj = Path(h5_path)
        if h5_obj.exists():
            df = pd.read_hdf(h5_path, key="trace metadata")
            if "mouse_id" not in df.columns:
                df["mouse_id"] = df.index # fallback
            
            for mouse_id, group in df.groupby("mouse_id"):
                print(f"Mouse {mouse_id}: {len(group)} rows")
                if len(group) > 1:
                    if "date" in group.columns or "datetime" in group.columns:
                        col = "date" if "date" in group.columns else "datetime"
                        gaps = pd.to_datetime(group[col]).sort_values().diff().dropna()
                        med = gaps.median()
                        print(f"Median gap: {med}")
                        if med > pd.Timedelta(days=1):
                            raise ValueError(f"Median gap {med} > 1 day. A 10-row chunk spans weeks to months and must not be integrated as 10 consecutive frames at dt = 0.01.")
                    elif "age_months" in group.columns:
                        gaps = group["age_months"].sort_values().diff().dropna()
                        med = gaps.median()
                        print(f"Median gap (months): {med}")
                        if med > (1.0 / 30.0):
                            raise ValueError(f"Median gap {med} months > 1 day. A 10-row chunk spans weeks to months and must not be integrated as 10 consecutive frames at dt = 0.01.")

        master_dataset = CatnapContinuousDataset(h5_path=h5_path, sequence_length=seq_len, cohort="all")
        
        from src.benchmarks.echo_resilience.task_registry import build_cohorts
        def young_fn(s):
            return s['chronological_age'] <= 6
        def old_fn(s):
            return s['chronological_age'] >= 24
            
        seed = config.get("experiment", {}).get("seed", 42)
        if len(master_dataset.samples) == 0:
            train_s, val_s, young_s, old_s = [], [], [], []
        else:
            train_s, val_s, young_s, old_s = build_cohorts(master_dataset.samples, "catnap", seed, young_fn, old_fn)
            
        train_data = CatnapContinuousDataset.__new__(CatnapContinuousDataset)
        train_data.samples = train_s
        val_data = CatnapContinuousDataset.__new__(CatnapContinuousDataset)
        val_data.samples = val_s
        young_data = CatnapContinuousDataset.__new__(CatnapContinuousDataset)
        young_data.samples = young_s
        old_data = CatnapContinuousDataset.__new__(CatnapContinuousDataset)
        old_data.samples = old_s

        return train_data, val_data, young_data, old_data

    @property
    def d_sensory(self) -> int:
        return 309

    def get_dataloaders(self, config, d_state: int = None, batch_size: int = 2) -> tuple:
        train_data, val_data, young_data, old_data = self.get_raw_datasets(config)
        
        train_loader = DataLoader(JAXDictDataset(train_data, d_state), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(JAXDictDataset(val_data, d_state), batch_size=batch_size, shuffle=False)
        young_loader = DataLoader(JAXDictDataset(young_data, d_state), batch_size=batch_size, shuffle=False)
        old_loader = DataLoader(JAXDictDataset(old_data, d_state), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, young_loader, old_loader

    def apply_dataset_change(self, trajectory, change_fn=None, **kwargs):
        return trajectory

    def compute_domain_metrics(self, trajectory) -> dict:
        return {"variance_norm": float(np.var(trajectory))}

    def render_animation(self, young_data, old_data, output_path: str, **kwargs):
        logger.info("Catnap animation not yet implemented. Skipping.")

    @property
    def cohort_labels(self):
        return ("<= 6 months", ">= 24 months")
