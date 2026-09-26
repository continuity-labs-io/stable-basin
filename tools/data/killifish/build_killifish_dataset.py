import logging
import pathlib
import pandas as pd
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class KillifishContinuousDataset(Dataset):
    """
    PyTorch Dataset for continuous-time physics modeling of African turquoise killifish.
    Parses continuous kinematics features and links them with lifespan metadata.
    """
    def __init__(
        self,
        metadata_csv: str = "data/killifish/data/a1_20241119/26441580/df_reformat_10_20241119.csv",
        kinematics_dir: str = "data/killifish/data/p3_20230526/test/standardization/",
        sequence_length: int = 100
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.kinematics_dir = pathlib.Path(kinematics_dir)
        
        # Load and filter metadata
        logger.info("Loading metadata from %s", metadata_csv)
        self.metadata = pd.read_csv(metadata_csv)
        
        # Filter for natural deaths
        initial_count = len(self.metadata)
        self.metadata = self.metadata[self.metadata['status'] == 'd']
        logger.info("Filtered metadata for natural deaths (status == 'd'). Retained %d of %d records.", len(self.metadata), initial_count)
        
        # Create mapping dictionaries
        # Assuming the CSV contains 'fish_number' and 'lifespan'
        # We also keep 'age_days' as the chronological age for a specific session if it exists per row.
        # If there are multiple sessions per fish, the CSV might have multiple rows.
        self.fish_to_lifespan = {}
        self.session_info = []
        
        for _, row in self.metadata.iterrows():
            fish_num = str(row['fish_number'])
            lifespan = float(row['lifespan'])
            self.fish_to_lifespan[fish_num] = lifespan
            
        logger.info("Created lifespan mapping for %d unique fish.", len(self.fish_to_lifespan))
        
        # Discover kinematic .h5 files
        self.samples = []
        
        logger.info("Scanning for kinematic files in %s", self.kinematics_dir)
        if self.kinematics_dir.exists():
            for h5_file in self.kinematics_dir.rglob("*.h5"):
                # Extract fish number from the directory structure or filename if possible
                # Typically subfolders are named by fish_number
                fish_num = h5_file.parent.name
                if fish_num in self.fish_to_lifespan:
                    # Parse features
                    features = self._load_and_normalize_h5(h5_file)
                    if features is not None and len(features) >= self.sequence_length:
                        # Extract chronological age. 
                        # We attempt to look up age_days from metadata for this specific recording, 
                        # or extract it from filename/folder. For robustness, if the CSV has 1 row per fish, 
                        # age_days might be in that row.
                        fish_records = self.metadata[self.metadata['fish_number'].astype(str) == fish_num]
                        
                        # Fallback heuristic: use the first matched record's age_days
                        chronological_age = float(fish_records['age_days'].iloc[0]) if 'age_days' in fish_records.columns and not fish_records.empty else 0.0
                        ultimate_lifespan = self.fish_to_lifespan[fish_num]
                        
                        # Chunk the sequence
                        num_chunks = len(features) // self.sequence_length
                        for i in range(num_chunks):
                            start_idx = i * self.sequence_length
                            end_idx = start_idx + self.sequence_length
                            chunk = features[start_idx:end_idx]
                            
                            self.samples.append({
                                'trajectory_chunk': torch.tensor(chunk, dtype=torch.float32),
                                'chronological_age': torch.tensor(chronological_age, dtype=torch.float32),
                                'ultimate_lifespan': torch.tensor(ultimate_lifespan, dtype=torch.float32)
                            })
                            
            logger.info("Constructed %d sequence chunks across all valid kinematic files.", len(self.samples))
        else:
            logger.warning("Kinematics directory %s does not exist. Dataset will be empty.", self.kinematics_dir)

    def _load_and_normalize_h5(self, file_path: pathlib.Path) -> np.ndarray:
        """
        Loads continuous features from an .h5 file, skipping metadata columns, and normalizes them.
        """
        try:
            # Read the HDF5 file using pandas
            df = pd.read_hdf(file_path)
            
            # Select columns ending in _m or _s
            valid_columns = [col for col in df.columns if col.endswith('_m') or col.endswith('_s')]
            
            if not valid_columns:
                return None
                
            features = df[valid_columns].values
            
            # Normalize: mean zero, unit variance (column-wise)
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            # Avoid division by zero
            std[std == 0] = 1.0
            features = (features - mean) / std
            
            return features
            
        except Exception as e:
            logger.info("Skipped parsing file %s due to error: %s", file_path, str(e))
            
        return None

    def __len__(self) -> int:
        return len(self.samples)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, idx: int) -> tuple[Float[Tensor, "seq_len features"], Float[Tensor, ""], Float[Tensor, ""]]:
        sample = self.samples[idx]
        return sample['trajectory_chunk'], sample['chronological_age'], sample['ultimate_lifespan']

if __name__ == "__main__":
    # Smoke test instantiation
    dataset = KillifishContinuousDataset()
    logger.info("Dataset initialization complete. Total samples: %d", len(dataset))
