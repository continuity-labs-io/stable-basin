import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from torch.utils.data import DataLoader
from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.data.async_event_packer import AsyncEventPackerDataset, ragged_collate_fn
from src.models.ssm.dynamic_integrator import DynamicIntegrator

def main():
    print("--- Testing DynamicIntegrator ---")
    
    # 1. Setup Packer Pipeline
    base_dataset = SyntheticWaddingtonDataset(size=8, seq_len=500, density=0.05)
    packer_dataset = AsyncEventPackerDataset(base_dataset, dt_resolution=0.1)
    dataloader = DataLoader(packer_dataset, batch_size=4, collate_fn=ragged_collate_fn)
    
    batch = next(iter(dataloader))
    events = batch['events']
    event_mask = batch['event_mask']
    
    print(f"Input Batched 'events' shape: {events.shape}")
    print(f"Input Batched 'event_mask' shape: {event_mask.shape}")
    
    # 2. Instantiate DynamicIntegrator
    dim = 30 # Waddington dataset produces 30D (20 slow + 10 fast sparse)
    d_state = 16
    integrator = DynamicIntegrator(dim=dim, d_state=d_state)
    
    # 3. Push batch through DynamicIntegrator
    print("\nPushing batch through DynamicIntegrator (this pure PyTorch routing is slow)...")
    h_sequence = integrator(events, event_mask)
    
    print(f"\nOutput 'h_sequence' shape: {h_sequence.shape}")
    print("Expected shape: [Batch, Max_Events, Dim, D_State]")
    
    if h_sequence.shape == (events.shape[0], events.shape[1], dim, d_state):
        print("Success! Shapes match the expectation.")
    else:
        print("Warning: Output shapes do not match expected shapes.")

if __name__ == "__main__":
    main()
