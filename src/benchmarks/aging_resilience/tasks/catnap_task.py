import logging
import numpy as np
from torch.utils.data import DataLoader

from src.benchmarks.aging_resilience.task_registry import AgingBenchmarkTask
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

        train_data = CatnapContinuousDataset(h5_path=h5_path, sequence_length=seq_len, cohort="train")
        young_data = CatnapContinuousDataset(h5_path=h5_path, sequence_length=seq_len, cohort="young")
        old_data = CatnapContinuousDataset(h5_path=h5_path, sequence_length=seq_len, cohort="old")

        return train_data, young_data, old_data

    @property
    def d_sensory(self) -> int:
        return 309

    def get_dataloaders(self, config, d_state: int = None, batch_size: int = 2) -> tuple:
        train_data, young_data, old_data = self.get_raw_datasets(config)
        
        train_loader = DataLoader(JAXDictDataset(train_data, d_state), batch_size=batch_size, shuffle=True)
        young_loader = DataLoader(JAXDictDataset(young_data, d_state), batch_size=batch_size, shuffle=False)
        old_loader = DataLoader(JAXDictDataset(old_data, d_state), batch_size=batch_size, shuffle=False)

        return train_loader, young_loader, old_loader

    def apply_dataset_change(self, trajectory, change_fn=None, **kwargs):
        return trajectory

    def compute_domain_metrics(self, trajectory) -> dict:
        return {"variance_norm": float(np.var(trajectory))}

    def render_animation(self, young_data, old_data, output_path: str, **kwargs):

    @property
    def cohort_labels(self):
        return ("<= 6 months", ">= 24 months")
        logger.info("Catnap animation not yet implemented. Skipping.")
