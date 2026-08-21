import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class FinalSparkDataset(Dataset):
    """
    FinalSpark Whole-Life Telemetry Dataset.
    
    Origin: Open and remotely accessible Neuroplatform for research in wetware computing
    Sampling: 30kHz-resolution raw activity samples. Data is provided in HDF5 format.
    
    This dataset uses event-driven windowing: It iterates over detected spikes (events) 
    and leverages the `segment_index` parquet file to efficiently query the raw HDF5 
    table for a precise biological time window around each event, completely avoiding 
    costly full-table scans.
    
    It also implements Multimodal Environmental Fusion by pairing the high-frequency
    electrical snippets with low-frequency incubator telemetry (e.g. door opening state)
    to study how macroscopic perturbations alter the thermodynamic vector.
    """
    def __init__(self, export_dir: str = "data/ephys/finalspark/fs437_export", seq_len: int = 1024, max_events: int = None):
        self.export_dir = export_dir
        self.seq_len = seq_len
        self.package_path = os.path.join(export_dir, "fs437_package.hdf5")
        self.raw_path = os.path.join(export_dir, "fs437_raw.hdf5")
        self.index_path = os.path.join(export_dir, "fs437_segment_index.parquet")
        
        if not os.path.exists(self.package_path) or not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"FinalSpark HDF5 files not found in {export_dir}")
            
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Segment index {self.index_path} is required for efficient raw queries.")

        # 1. Load the lightweight Events table
        self.events = pd.read_hdf(self.package_path, key="fs437_wholelife_events")
        self.events['time_of_event'] = pd.to_datetime(self.events['time_of_event'], utc=True)
        
        # Optionally limit events for faster testing
        if max_events is not None:
            self.events = self.events.head(max_events)
            
        # 2. Load the Environmental Metadata (Incubator Door Openings)
        self.door = pd.read_hdf(self.package_path, key="fs437_wholelife_incubator_door_opening")
        self.door['time'] = pd.to_datetime(self.door['time'], utc=True)
        # Sort for efficient asof/nearest matching
        self.door = self.door.sort_values('time')
        
        # 3. Load the Segment Index
        self.segment_index = pd.read_parquet(self.index_path)
        
        # Convert index times to UTC for clean matching
        if self.segment_index['t_start'].dt.tz is None:
            self.segment_index['t_start'] = self.segment_index['t_start'].dt.tz_localize('UTC')
            self.segment_index['t_end'] = self.segment_index['t_end'].dt.tz_localize('UTC')

    def __len__(self):
        return len(self.events)
        
    def __getitem__(self, idx):
        event = self.events.iloc[idx]
        event_time = event['time_of_event']
        electrode = event['electrode']
        
        # Match event to segment index
        seg_mask = (self.segment_index['electrode'] == electrode) & \
                   (self.segment_index['t_start'] <= event_time) & \
                   (self.segment_index['t_end'] >= event_time)
        
        segs = self.segment_index[seg_mask]
        
        voltage = np.zeros(self.seq_len, dtype=np.float32)
        
        if len(segs) > 0:
            seg = segs.iloc[0]
            row_start = int(seg['row_start'])
            row_end = int(seg['row_end'])
            
            # Read only the segment row range using pd.HDFStore to ensure schemas parse correctly
            with pd.HDFStore(self.raw_path, mode='r') as store:
                chunk = store.select('fs437_wholelife_raw', start=row_start, stop=row_end+1)
            
            chunk['time'] = pd.to_datetime(chunk['time'], utc=True)
            
            # Find the spike center in the chunk
            # Since chunk is small, this is extremely fast
            time_diffs = (chunk['time'] - event_time).abs()
            spike_idx = time_diffs.idxmin()
            
            # Extract window around spike
            half_seq = self.seq_len // 2
            
            # We want to slice [spike_idx - half_seq : spike_idx + half_seq]
            # Since index might not be contiguous or start at 0, use iloc
            pos = chunk.index.get_loc(spike_idx)
            
            start_pos = max(0, pos - half_seq)
            end_pos = start_pos + self.seq_len
            
            snippet = chunk['voltage_uv'].values[start_pos:end_pos]
            
            # Copy into our zero-padded voltage array
            length = min(len(snippet), self.seq_len)
            voltage[:length] = snippet[:length]

        # Get environmental state (door open/close) at the time of the event
        # Using searchsorted for O(log N) nearest historical state
        door_idx = self.door['time'].searchsorted(event_time, side='right') - 1
        if door_idx >= 0:
            env_door = float(self.door.iloc[door_idx]['fs437_wholelife_incubator_door_opening'])
        else:
            env_door = 0.0

        return {
            "x_raw": torch.from_numpy(voltage).unsqueeze(-1), # (seq_len, 1)
            "env_door": torch.tensor([env_door], dtype=torch.float32),
            "electrode": torch.tensor([electrode], dtype=torch.float32)
        }
