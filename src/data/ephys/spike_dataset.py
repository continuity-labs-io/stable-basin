import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset, get_worker_info
from huggingface_hub import snapshot_download

import logging

logger = logging.getLogger(__name__)


class SpikeProphecyDataset(IterableDataset):
    def __init__(self, time_steps: int, split: str = "train", data_dir: str = None):
        """
        IterableDataset for the SpikeProphecy Steinmetz dataset.

        Args:
            time_steps (int): The required number of time steps (history bins) for the sliding window.
            split (str): One of 'train', 'val', or 'test'.
            data_dir (str): Optional path to local dataset cache. If None, it will download via huggingface_hub.
        """
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of 'train', 'val', 'test', got {split}")

        self.time_steps = time_steps
        self.split = split

        if data_dir is None:
            # Download or use cached dataset
            logger.info("Downloading/locating mysteriousauthor/spikeprophecy-steinmetz...")
            self.data_dir = Path(
                snapshot_download(
                    repo_id="mysteriousauthor/spikeprophecy-steinmetz", repo_type="dataset"
                )
            )
        else:
            self.data_dir = Path(data_dir)

        # Load metadata
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")

        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        self.m_max = self.metadata["m_max"]
        self.num_sessions = self.metadata["num_sessions"]

        # We will iterate over sessions 0 to num_sessions - 1
        # In metadata.json, the sessions might be a list or a dict.
        self.sessions_info = self.metadata.get("sessions", [])
        if isinstance(self.sessions_info, dict):
            # If it's a dict like "0": {...}, "1": {...}
            self.session_keys = sorted(self.sessions_info.keys(), key=int)
        else:
            # If it's a list
            self.session_keys = list(range(len(self.sessions_info)))

    def _get_split_bounds(self, sb, total_bins):
        train_end = sb["train_end"]
        val_end = sb["val_end"]
        if self.split == "train":
            return 0, train_end
        elif self.split == "val":
            return train_end, val_end
        elif self.split == "test":
            return val_end, total_bins

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Shard sessions across workers to avoid data duplication
        # e.g., worker 0 gets session 0, 2, 4... worker 1 gets 1, 3, 5...
        sharded_session_keys = self.session_keys[worker_id::num_workers]

        for s_key in sharded_session_keys:
            if isinstance(self.sessions_info, dict):
                session_meta = self.sessions_info[s_key]
                s_idx = int(s_key)
            else:
                session_meta = self.sessions_info[s_key]
                s_idx = s_key

            # File format: session_000.npy, session_001.npy, ...
            session_file = self.data_dir / f"session_{s_idx:03d}.npy"
            if not session_file.exists():
                logger.warning(f"Warning: {session_file} not found. Skipping.")
                continue

            # Use mmap_mode="r" for a flat memory footprint (reads from disk lazily)
            counts = np.load(session_file, mmap_mode="r")
            n_units, total_bins = counts.shape

            sb = session_meta["split_boundaries"]
            start_bin, end_bin = self._get_split_bounds(sb, total_bins)

            # Slide window over the valid split
            # The window size is self.time_steps
            num_windows = (end_bin - start_bin) - self.time_steps + 1
            if num_windows <= 0:
                continue

            for i in range(num_windows):
                window_start = start_bin + i
                window_end = window_start + self.time_steps

                # window: shape [n_units, time_steps]
                window = counts[:, window_start:window_end]

                # Transpose to [time_steps, n_units]
                window_t = window.T

                # We need a fixed state vector size for the state space model.
                # Pad num_neurons to M_max (1240)
                # Padded shape: [time_steps, M_max]
                pad_width = self.m_max - n_units
                if pad_width > 0:
                    padded_window = np.pad(
                        window_t, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
                    )
                elif pad_width < 0:
                    # Unlikely, but safety crop if n_units > m_max
                    padded_window = window_t[:, : self.m_max]
                else:
                    padded_window = window_t

                # Convert to torch tensor (float32 for model consumption)
                tensor_window = torch.from_numpy(padded_window.copy()).float()
                yield tensor_window


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    logger.info("--- Testing SpikeProphecyDataset ---")
    # Initialize dataset with time_steps=10
    dataset = SpikeProphecyDataset(time_steps=10, split="train")

    # Use num_workers=2 to test multi-processing sharding
    dataloader = DataLoader(dataset, batch_size=32, num_workers=2)

    logger.info(f"Dataset M_max: {dataset.m_max}")
    logger.info(f"Total Sessions: {dataset.num_sessions}")

    # Iterate a few batches to verify
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i} shape: {batch.shape}, dtype: {batch.dtype}")
        # Expected shape: [32, 10, 1240]
        if i >= 2:
            break

    logger.info("Test completed successfully.")
