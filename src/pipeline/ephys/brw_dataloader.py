import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py

import logging

logger = logging.getLogger(__name__)


class ContinuousHDMEADataset(Dataset):
    """
    A PyTorch Dataset for loading continuous high-density electrophysiology data
    from raw 3Brain (.brw) files natively via HDF5.

    This bypasses spikeinterface metadata constraints and streams the 20kHz
    telemetry directly from the /3BData/Raw binary block, applying Z-score
    normalization to stabilize continuous-time state-space models.

    Data Source:
    To reproduce this pipeline, download the HD-MEA NEUROPulse Dataset from Zenodo:
    URL: https://zenodo.org/records/13908319
    This dataset contains raw 3Brain .brw data files (BrainWave format) recorded
    from 4,096-channel HD-MEAs (using BioCAM/Accura hardware). It includes both
    spontaneous baseline activity and evoked responses.
    """

    def __init__(
        self, brw_file_path: str, sequence_length: int = 10000, target_channels: int = 1024
    ):
        """
        Args:
            brw_file_path: Path to the .brw file.
            sequence_length: Number of time frames per sequence chunk.
            target_channels: Number of channels to subsample from the recording.
        """
        super().__init__()
        self.brw_file_path = brw_file_path
        self.sequence_length = sequence_length
        self.target_channels = target_channels
        self.total_channels = 4096

        # Open the HDF5 file natively in read-only mode
        self.file = h5py.File(self.brw_file_path, "r")
        self.raw_dataset = self.file["/3BData/Raw"]

        # Calculate total frames from the massive 1D array
        total_elements = self.raw_dataset.shape[0]
        self.total_frames = total_elements // self.total_channels
        self.sampling_rate = 20000.0

        # Calculate number of available non-overlapping chunks
        self.num_chunks = int(self.total_frames // self.sequence_length)

    def __len__(self):
        """Returns the total number of non-overlapping sequence chunks."""
        return self.num_chunks

    def __getitem__(self, idx):
        """
        Fetches a temporal chunk of traces, reshapes it, and applies
        robust Z-score normalization for the Mamba-2 engine.
        """
        if idx >= self.num_chunks or idx < 0:
            raise IndexError("Dataset index out of range.")

        start_frame = idx * self.sequence_length
        end_frame = start_frame + self.sequence_length

        # Slice the 1D dataset and reshape to [Frames, Channels]
        start_idx = start_frame * self.total_channels
        end_idx = end_frame * self.total_channels

        raw_chunk = self.raw_dataset[start_idx:end_idx]
        raw_chunk = raw_chunk.reshape(self.sequence_length, self.total_channels)

        # Spatial Subsampling
        traces = raw_chunk[:, : self.target_channels]

        # Robust Z-Score Normalization
        mean_val = traces.mean()
        std_val = traces.std()
        normalized_traces = (traces - mean_val) / (std_val + 1e-5)

        return torch.tensor(normalized_traces, dtype=torch.float32)

    def __del__(self):
        """Ensure the HDF5 file is cleanly closed."""
        if hasattr(self, "file") and self.file:
            self.file.close()


if __name__ == "__main__":
    # Define default directory relative to the repository root
    default_dir = "data/ephys"

    # Path to the dummy example file
    dummy_file_path = os.path.join(default_dir, "hdmea_neuropulse.brw")

    if not os.path.exists(dummy_file_path):
        logger.warning(f"Warning: Dummy file not found at {dummy_file_path}.")
        logger.info("Please place a valid 'hdmea_neuropulse.brw' file there to run the test.")

    try:
        # Instantiate the dataset
        dataset = ContinuousHDMEADataset(
            brw_file_path=dummy_file_path, sequence_length=10000, target_channels=1024
        )

        logger.info("Dataset successfully initialized.")
        logger.info(f"Total frames: {dataset.total_frames}")
        logger.info(f"Sampling rate: {dataset.sampling_rate} Hz")
        logger.info(f"Total chunks: {len(dataset)}")

        # Instantiate DataLoader with batch size 4
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Iterate over the first batch and print out the resulting tensor shape
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Batch {batch_idx + 1} tensor shape: {batch.shape}")
            break

    except Exception as e:
        logger.error(f"Failed to run the dataset test: {e}")
