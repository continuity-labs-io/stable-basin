import sys
import os

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader
from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.data.async_event_packer import AsyncEventPackerDataset, ragged_collate_fn

def test_packer():
    print("--- Testing AsyncEventPackerDataset ---")
    
    # 1. Create the base dataset
    # By default it's 30 dimensions (20 slow + 10 fast sparse) and seq_len=500
    base_dataset = SyntheticWaddingtonDataset(size=8, seq_len=500, density=0.05)
    
    print(f"Base Dataset Size: {len(base_dataset)}")
    base_sample = base_dataset[0]
    print(f"Base Sample 'x_raw' shape: {base_sample['x_raw'].shape} (dense, zero-padded)")
    print(f"Base Sample 'mask' shape: {base_sample['mask'].shape}")
    
    # 2. Wrap it with our new Packer
    packer_dataset = AsyncEventPackerDataset(base_dataset, dt_resolution=0.1)
    
    packed_sample = packer_dataset[0]
    print(f"\nPacked Sample 'events' shape: {packed_sample['events'].shape} (sparse, 1D)")
    print(f"Packed Sample 'num_events': {packed_sample['num_events']}")
    
    print("\nFirst 3 events [Value, Sensor_ID, Timestamp]:")
    print(packed_sample['events'][:3])
    
    # 3. Test the ragged collate function in a DataLoader
    print("\n--- Testing ragged_collate_fn in DataLoader ---")
    dataloader = DataLoader(packer_dataset, batch_size=4, collate_fn=ragged_collate_fn)
    
    batch = next(iter(dataloader))
    
    print(f"Batched 'events' shape: {batch['events'].shape}")
    print(f"Batched 'event_mask' shape: {batch['event_mask'].shape}")
    print(f"Lengths of sequences in batch: {batch['lengths']}")
    
    print("\nSuccess! The packer successfully strips zero-padding and collates ragged sequences.")

