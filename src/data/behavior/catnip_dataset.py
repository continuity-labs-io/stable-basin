import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype

logger = logging.getLogger(__name__)

class CatnipContinuousDataset(Dataset):
    """Dataset for Calico CATNAP HDF5 data."""
    def __init__(self, h5_path: str = "data/catnap/trace_features.h5", sequence_length: int = 10, cohort: str = "all"):
        self.samples = []
        
        try:
            h5_path_obj = Path(h5_path)
            if not h5_path_obj.exists():
                logger.warning(f"File {h5_path} does not exist. Initializing empty dataset.")
                return

            features_df = pd.read_hdf(h5_path, key="all features")
            trace_metadata_df = pd.read_hdf(h5_path, key="trace metadata")
            mouse_metadata_df = pd.read_hdf(h5_path, key="mouse metadata")

            # Merge 'all features' with 'trace metadata' (assuming they join on index)
            df = features_df.merge(trace_metadata_df, left_index=True, right_index=True)

            # Associate each run with a mouse_id and a chronological age
            df = df.merge(mouse_metadata_df, on="mouse_id")
            
            # Identify age column (assuming 'age_months' or 'age')
            age_col = "age_months" if "age_months" in df.columns else "age"
            
            if cohort == "young":
                df = df[df[age_col] <= 6]
            elif cohort == "old":
                df = df[df[age_col] >= 24]
            elif cohort == "train":
                import hashlib
                def is_train(mid):
                    h = hashlib.md5(str(mid).encode()).hexdigest()
                    return int(h, 16) % 10 < 8
                
                train_mask = df["mouse_id"].apply(is_train)
                df = df[train_mask]
            
            # Extract feature columns (excluding metadata columns)
            meta_cols = set(trace_metadata_df.columns).union(set(mouse_metadata_df.columns))
            if "mouse_id" not in meta_cols:
                meta_cols.add("mouse_id")
            
            feature_cols = [c for c in df.columns if c not in meta_cols]
            
            if not feature_cols:
                logger.warning("No feature columns found after removing metadata columns.")
                return

            # Apply z-score normalization to features globally
            features_np = df[feature_cols].values
            std = np.std(features_np, axis=0)
            std[std == 0] = 1.0
            mean = np.mean(features_np, axis=0)
            
            # Update DataFrame with normalized values
            df[feature_cols] = (features_np - mean) / std
            df[feature_cols] = df[feature_cols].fillna(0.0)
            
            # Group by mouse_id and sort chronologically
            for mouse_id, group in df.groupby("mouse_id"):
                group = group.sort_values(by=age_col)
                mouse_features = group[feature_cols].values
                
                num_chunks = len(mouse_features) // sequence_length
                for i in range(num_chunks):
                    start_idx = i * sequence_length
                    end_idx = start_idx + sequence_length
                    chunk = mouse_features[start_idx:end_idx]
                    
                    chrono_age = group.iloc[start_idx][age_col]
                    lifespan = group.iloc[start_idx].get("lifespan", 0)
                    
                    self.samples.append({
                        'trajectory_chunk': torch.tensor(chunk, dtype=torch.float32),
                        'chronological_age': chrono_age,
                        'ultimate_lifespan': lifespan
                    })

        except Exception as e:
            logger.warning(f"Error loading HDF5 file: {e}. Initializing empty dataset.")
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, idx: int) -> Float[Tensor, "seq_len num_features"]:
        return self.samples[idx]['trajectory_chunk']
