import os
import h5py
import numpy as np
import scipy.signal as signal
from scipy.stats import linregress
import pandas as pd
from datetime import datetime

def parse_time(t_str):
    # 'start: 2020-01-23 11:01:21;\nstop: 2020-01-23 11:04:21\n'
    parts = t_str.split(';')
    start_str = parts[0].replace('start: ', '').strip()
    return datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")

def compute_metrics(filepath, fs=20000.0, chunk_size=200000):
    with h5py.File(filepath, "r") as f:
        sig = f["sig"]
        n_channels, n_samples = sig.shape
        n_channels = min(1024, n_channels)
        
        t_str = f['time'][0].decode('utf-8')
        start_time = parse_time(t_str)
        
        sos = signal.butter(3, [300, 3000], btype='bandpass', fs=fs, output='sos')
        
        chunk0 = sig[:n_channels, :chunk_size].astype(np.float32)
        chunk0_filt = signal.sosfiltfilt(sos, chunk0, axis=1)
        median_val = np.median(chunk0_filt, axis=1, keepdims=True)
        mad = np.median(np.abs(chunk0_filt - median_val), axis=1) / 0.6745
        mad = np.maximum(mad, 1e-6)
        
        thresholds = -5.0 * mad
        
        bin_samples = int(fs * 10 / 1000)
        n_bins = int(np.ceil(n_samples / bin_samples))
        pfr = np.zeros(n_bins)
        
        for start in range(0, n_samples, chunk_size):
            end = min(n_samples, start + chunk_size)
            chunk = sig[:n_channels, start:end].astype(np.float32)
            chunk_filt = signal.sosfiltfilt(sos, chunk, axis=1)
            
            local_min = np.zeros_like(chunk_filt, dtype=bool)
            local_min[:, 1:-1] = (chunk_filt[:, 1:-1] < chunk_filt[:, :-2]) & (chunk_filt[:, 1:-1] < chunk_filt[:, 2:])
            
            is_below = chunk_filt < thresholds[:, None]
            spikes = is_below & local_min
            
            ch_idx, t_idx = np.where(spikes)
            t_global = t_idx + start
            
            bin_idx = t_global // bin_samples
            np.add.at(pfr, bin_idx, 1)
            
        return pfr, start_time, n_channels

def calc_tau(pfr, dt):
    if len(pfr) <= 1 or np.var(pfr) < 1e-8:
        return 0.0
    r1 = np.corrcoef(pfr[:-1], pfr[1:])[0,1]
    if r1 > 0:
        return -dt / np.log(np.clip(r1, 1e-6, 0.9999))
    return 0.0

files = {
    "control": "data/ephys/pharmacological_shock/Drug_2953_control.raw.h5",
    "3uM": "data/ephys/pharmacological_shock/Drug_2953_3uM.raw.h5",
    "10uM": "data/ephys/pharmacological_shock/Drug_2953_10uM.raw.h5",
    "30uM": "data/ephys/pharmacological_shock/Drug_2953_30uM.raw.h5",
    "50uM": "data/ephys/pharmacological_shock/Drug_2953_50uM.raw.h5"
}

results = {}
for cond, path in files.items():
    print(f"Processing {cond}...")
    pfr, start_time, n_channels = compute_metrics(path)
    
    # calc full metrics
    dt = 0.01 # 10 ms
    tau = calc_tau(pfr, dt)
    
    mean_rate = np.mean(pfr) / dt / n_channels
    thresh_burst = np.mean(pfr) + 3*np.std(pfr)
    burst_rate = np.sum(pfr > thresh_burst) / (len(pfr) * dt)
    
    # 30s windows
    win_samples = 30.0 / dt
    win_size = int(win_samples)
    n_windows = len(pfr) // win_size
    wins = []
    for i in range(n_windows):
        w_pfr = pfr[i*win_size:(i+1)*win_size]
        wins.append(calc_tau(w_pfr, dt))
        
    results[cond] = {
        "start_time": start_time,
        "tau": tau,
        "mean_rate": mean_rate,
        "burst_rate": burst_rate,
        "tau_windows": wins
    }

# 1. Sort by time and print
print("\nFiles in time order:")
sorted_conds = sorted(results.keys(), key=lambda k: results[k]['start_time'])
for cond in sorted_conds:
    print(f"{cond}: {results[cond]['start_time']}")

# 2. Control file fit line
ctrl_wins = results["control"]["tau_windows"]
# Times for windows in minutes from start of control
# Window size is 30s = 0.5 mins. Center of window: 0.25, 0.75, 1.25, etc.
t_mins = np.array([0.25 + 0.5 * i for i in range(len(ctrl_wins))])

slope, intercept, r_value, p_value, std_err = linregress(t_mins, ctrl_wins)
print(f"\nControl drift slope: {slope:.6f} Tau/min")

# 3. Predict drift vs observed drop
control_start = results["control"]["start_time"]
drug_start = results["50uM"]["start_time"]
minutes_between = (drug_start - control_start).total_seconds() / 60.0

predicted_drift = slope * minutes_between
observed_drop = results["control"]["tau"] - results["50uM"]["tau"]

print(f"Minutes between control and 50uM: {minutes_between:.1f}")
print(f"Predicted drift: {predicted_drift:.6f}")
print(f"Observed drop: {observed_drop:.6f}")

time_gate_pass = abs(predicted_drift) < (1/3.0) * abs(observed_drop)
gate_str = "PASS" if time_gate_pass else "FAIL"

# 4 & 5. Output Table and Gate
print("\n--- METRIC TABLE ---")
print("Condition\tStartTime\tTau(AC)\tMeanRate\tBurstRate")
for cond in sorted_conds:
    r = results[cond]
    st = r["start_time"].strftime("%H:%M:%S")
    t = r["tau"]
    m = r["mean_rate"]
    b = r["burst_rate"]
    print(f"{cond}\t{st}\t{t:.5f}\t{m:.5f}\t{b:.5f}")

print(f"\nTime gate: {gate_str}")

