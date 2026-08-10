import h5py
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class MaxWellHDMEADataset(Dataset):
    """
    A PyTorch Dataset for loading continuous high-density electrophysiology data
    from raw MaxWell Biosystems (.raw.h5) files natively via HDF5.

    This dataloader extracts the raw voltage trace array, transposes it, and
    applies robust Z-score normalization for stable continuous-time modeling.
    """

    def __init__(self, file_path: str, sequence_length: int = 10000, target_channels: int = 1024):
        """
        Args:
            file_path: Path to the .raw.h5 file.
            sequence_length: Number of time frames per sequence chunk.
            target_channels: Number of channels to subsample from the recording.
        """
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.target_channels = target_channels
        self.file = h5py.File(self.file_path, "r")

        # MaxWell files store the raw voltage trace array under different keys depending on the export
        keys_to_try = ["/sig", "/routing/lsb", "/mapping/sig"]
        self.raw_dataset = None
        for key in keys_to_try:
            if key in self.file:
                self.raw_dataset = self.file[key]
                break

        if self.raw_dataset is None:
            raise KeyError(
                f"Could not find any of the expected MaxWell keys {keys_to_try} in {file_path}"
            )

    def __len__(self):
        """Returns the number of available non-overlapping sequence chunks."""
        # Shape is typically [Channels, TimeFrames]
        total_frames = self.raw_dataset.shape[1]
        return total_frames // self.sequence_length

    def __getitem__(self, idx):
        """
        Fetches a temporal chunk, transposes it to [Time, Channels], and applies
        robust Z-score normalization.
        """
        start_frame = idx * self.sequence_length
        end_frame = start_frame + self.sequence_length

        # Slicing the h5py dataset: [Channels, TimeFrames]
        chunk = self.raw_dataset[: self.target_channels, start_frame:end_frame]

        # Transpose so final shape is [Time, Channels]
        chunk = chunk.T

        # Convert to float32 tensor
        chunk = torch.tensor(chunk, dtype=torch.float32)

        # Robust Z-score normalization
        chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-5)

        return chunk

    def __del__(self):
        if hasattr(self, "file") and self.file is not None:
            self.file.close()
