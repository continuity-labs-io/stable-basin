import os
import torch
import tifffile
import numpy as np
from torch.utils.data import Dataset, DataLoader

import logging

logger = logging.getLogger(__name__)


class MeldTemporalDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10, transform=None):
        """
        Loads sequential TIFF stacks into a 4D spatiotemporal tensor (T, Z, Y, X).
        For multi-channel (C), you would initialize two datasets and stack them.
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Sort files to ensure strict temporal order
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".tif")])

        # Ensure we have enough frames
        assert len(self.files) >= self.sequence_length, "Not enough frames for sequence length."

    def __len__(self):
        # Number of continuous sequences we can extract
        return len(self.files) - self.sequence_length + 1

    def __getitem__(self, idx):
        frames = []
        # Load a continuous chunk of T frames
        for t in range(self.sequence_length):
            file_path = os.path.join(self.data_dir, self.files[idx + t])

            # Read the 3D TIFF stack (Z, Y, X)
            image_3d = tifffile.imread(file_path)
            frames.append(image_3d)

        # Stack into (T, Z, Y, X)
        tensor_4d = np.stack(frames, axis=0)

        # Convert to PyTorch float32 tensor
        tensor_4d = torch.from_numpy(tensor_4d).to(torch.float32)

        if self.transform:
            tensor_4d = self.transform(tensor_4d)

        # Add channel dimension (C, T, Z, Y, X) -> returning a 1-channel block here
        return tensor_4d.unsqueeze(0)


class AOLLSMDataset(Dataset):
    def __init__(self, data_dir, num_frames=199, crop_size=(128, 128, 128), compressor=None, device=None):
        """
        PyTorch Dataset for AO-LLSM sequential TIFF volumes.
        Ingests a directory of sequential TIFF stacks and stacks ch1 & ch2.

        Args:
            data_dir (str): Directory containing the TIFF files or subdirectories.
            num_frames (int): Number of temporal frames to load (e.g. 199).
            crop_size (tuple): Target shape for central spatial crop (Depth, Height, Width).
            compressor (nn.Module, optional): Model to compress raw frames (e.g. SpatialCompressor).
            device (torch.device, optional): Device to run compression on.
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.compressor = compressor
        self.device = device if device else torch.device("cpu")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

        # Identify sample directories
        # Case 1: data_dir contains the stack files directly or has stack0000 subdirectory
        has_direct_stacks = any("stack0000" in f for f in os.listdir(data_dir)) or os.path.isdir(
            os.path.join(data_dir, "stack0000")
        )

        if has_direct_stacks:
            self.sample_dirs = [data_dir]
        else:
            # Case 2: data_dir contains subdirectories, each of which is a sample
            self.sample_dirs = []
            for item in sorted(os.listdir(data_dir)):
                sub_path = os.path.join(data_dir, item)
                if os.path.isdir(sub_path):
                    has_sub_stacks = any(
                        "stack0000" in f for f in os.listdir(sub_path)
                    ) or os.path.isdir(os.path.join(sub_path, "stack0000"))
                    if has_sub_stacks:
                        self.sample_dirs.append(sub_path)

            # Fallback to single sample if no sub-samples found
            if not self.sample_dirs:
                self.sample_dirs = [data_dir]

        logger.info(
            f"[INIT] AOLLSMDataset initialized with {len(self.sample_dirs)} sample(s) in: {data_dir}"
        )

    def __len__(self):
        return len(self.sample_dirs)

    def _find_file(self, sample_dir, t, channel):
        """
        Finds the TIFF file for a specific time step (t) and channel ('ch1' or 'ch2').
        """
        stack_str = f"stack{t:04d}"

        # Try nested directory structure (sample_dir/stackXXXX/)
        nested_dir = os.path.join(sample_dir, stack_str)
        if os.path.isdir(nested_dir):
            for fname in os.listdir(nested_dir):
                if fname.endswith((".tif", ".tiff")) and f"_{channel}_" in fname:
                    return os.path.join(nested_dir, fname)

        # Try flat directory structure (sample_dir/filename_stackXXXX_...)
        for fname in os.listdir(sample_dir):
            if fname.endswith((".tif", ".tiff")) and stack_str in fname and f"_{channel}_" in fname:
                return os.path.join(sample_dir, fname)

        return None

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # List to hold the multi-channel volumes for each time step
        temporal_sequence = []

        for t in range(self.num_frames):
            # 1. Locate files for ch1 and ch2
            ch1_file = self._find_file(sample_dir, t, "ch1")
            ch2_file = self._find_file(sample_dir, t, "ch2")

            if ch1_file is None or ch2_file is None:
                raise FileNotFoundError(
                    f"Could not find both ch1 and ch2 TIFF files for time step {t} in {sample_dir}"
                )

            # 2. Load the volumetric TIFFs using memory mapping for optimal memory management
            ch1_vol_mmap = tifffile.imread(ch1_file, out="memmap")
            ch2_vol_mmap = tifffile.imread(ch2_file, out="memmap")

            # 3. Apply central crop to both volumes
            ch1_crop = self._crop_center(ch1_vol_mmap)
            ch2_crop = self._crop_center(ch2_vol_mmap)

            # Stack the physical channels to form (2, Depth, Height, Width)
            stacked_channels = np.stack([ch1_crop, ch2_crop], axis=0)

            # Convert to PyTorch float32 tensor
            stacked_tensor = torch.from_numpy(stacked_channels).to(torch.float32)

            # Apply compressor frame-by-frame to save memory if provided
            if self.compressor is not None:
                stacked_tensor = stacked_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    compressed = self.compressor(stacked_tensor)
                stacked_tensor = compressed.squeeze(0).squeeze(0).cpu()

            temporal_sequence.append(stacked_tensor)

        # 4. Stack all time steps to form (Time, ...)
        # If not compressed: (num_frames, 2, crop_size[0], crop_size[1], crop_size[2])
        # If compressed: (num_frames, 768)
        sequence_tensor = torch.stack(temporal_sequence, dim=0)

        return sequence_tensor

    def _crop_center(self, volume):
        """
        Extracts a central spatial crop of self.crop_size (Depth, Height, Width).
        Pads with zeros if volume dimensions are smaller than crop_size.
        """
        D, H, W = volume.shape
        Cd, Ch, Cw = self.crop_size

        if D < Cd or H < Ch or W < Cw:
            padded = np.zeros((Cd, Ch, Cw), dtype=volume.dtype)

            d_size = min(D, Cd)
            h_size = min(H, Ch)
            w_size = min(W, Cw)

            d_start_v = max(0, (D - d_size) // 2)
            h_start_v = max(0, (H - h_size) // 2)
            w_start_v = max(0, (W - w_size) // 2)

            d_start_p = (Cd - d_size) // 2
            h_start_p = (Ch - h_size) // 2
            w_start_p = (Cw - w_size) // 2

            padded[
                d_start_p : d_start_p + d_size,
                h_start_p : h_start_p + h_size,
                w_start_p : w_start_p + w_size,
            ] = volume[
                d_start_v : d_start_v + d_size,
                h_start_v : h_start_v + h_size,
                w_start_v : w_start_v + w_size,
            ]
            return padded
        else:
            d_start = (D - Cd) // 2
            h_start = (H - Ch) // 2
            w_start = (W - Cw) // 2
            # Explicitly return a copy so the memory-mapped resource can be released
            return volume[
                d_start : d_start + Cd, h_start : h_start + Ch, w_start : w_start + Cw
            ].copy()


# --- Prototyping Execution ---

