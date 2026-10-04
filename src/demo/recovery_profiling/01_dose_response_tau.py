import os
import h5py
import numpy as np
import scipy.signal as signal
import pandas as pd

def compute_pfr(filepath, fs=20000.0, chunk_size=200000, mad_mults=[-4.5, -5.0, -5.5], bin_sizes=[5, 10, 20]):
    with h5py.File(filepath, "r") as f:
        sig = f["sig"]
        n_channels, n_samples = sig.shape
        n_channels = min(1024, n_channels)
        
        sos = signal.butter(3, [300, 3000], btype='bandpass', fs=fs, output='sos')
        
        chunk0 = sig[:n_channels, :chunk_size].astype(np.float32)
        chunk0_filt = signal.sosfiltfilt(sos, chunk0, axis=1)
        median_val = np.median(chunk0_filt, axis=1, keepdims=True)
        mad = np.median(np.abs(chunk0_filt - median_val), axis=1) / 0.6745
        mad = np.maximum(mad, 1e-6)
        
        thresholds_dict = {mult: mult * mad for mult in mad_mults}
        
        # Prepare structures for PFR
        # We need pfr for each mult and bin_size
        pfr_dict = {}
        for mult in mad_mults:
            for bz in bin_sizes:
                bin_samples = int(fs * bz / 1000)
                n_bins = int(np.ceil(n_samples / bin_samples))
                pfr_dict[(mult, bz)] = np.zeros(n_bins)
                
        for start in range(0, n_samples, chunk_size):
            end = min(n_samples, start + chunk_size)
            chunk = sig[:n_channels, start:end].astype(np.float32)
            chunk_filt = signal.sosfiltfilt(sos, chunk, axis=1)
            
            local_min = np.zeros_like(chunk_filt, dtype=bool)
            local_min[:, 1:-1] = (chunk_filt[:, 1:-1] < chunk_filt[:, :-2]) & (chunk_filt[:, 1:-1] < chunk_filt[:, 2:])
            
            for mult in mad_mults:
                thresh = thresholds_dict[mult]
                is_below = chunk_filt < thresh[:, None]
                spikes = is_below & local_min
                
                ch_idx, t_idx = np.where(spikes)
                t_global = t_idx + start
                
                for bz in bin_sizes:
                    bin_samples = int(fs * bz / 1000)
                    bin_idx = t_global // bin_samples
                    np.add.at(pfr_dict[(mult, bz)], bin_idx, 1)
                    
        return pfr_dict, n_samples, fs

def calc_tau(pfr, dt):
    if len(pfr) <= 1:
        return 0.0
    # Add a tiny noise to prevent identical values or flatlines
    if np.var(pfr) < 1e-8:
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
    pfr_dict, n_samples, fs = compute_pfr(path)
    results[cond] = pfr_dict

# We want Tau(AC) for default (10ms, -5.0) and all other combos
mad_mults = [-4.5, -5.0, -5.5]
bin_sizes = [5, 10, 20]

# For each combo, let's extract Tau (AC) full recording
# we also need 30s windows for default setting (10ms, -5.0)
tau_full = {}
tau_windows = {}

for cond in files.keys():
    tau_full[cond] = {}
    
    for mult in mad_mults:
        for bz in bin_sizes:
            dt = bz / 1000.0
            pfr = results[cond][(mult, bz)]
            tau = calc_tau(pfr, dt)
            tau_full[cond][(mult, bz)] = tau
            
            # for default, compute windows
            if mult == -5.0 and bz == 10:
                # 30s windows
                win_samples = 30.0 / dt
                win_size = int(win_samples)
                n_windows = len(pfr) // win_size
                wins = []
                for i in range(n_windows):
                    w_pfr = pfr[i*win_size:(i+1)*win_size]
                    wins.append(calc_tau(w_pfr, dt))
                tau_windows[cond] = wins

# Gate evaluations

# 3. Robustness
# "PASS if the direction of the control-vs-drug change is the same in all six settings."
# Assuming the "drug" is 50uM. Or we can just use the highest dose.
# I will check if sign(Tau_50uM - Tau_control) is identical for all 9 combinations.
combo_directions = []
for mult in mad_mults:
    for bz in bin_sizes:
        diff = tau_full["50uM"][(mult, bz)] - tau_full["control"][(mult, bz)]
        combo_directions.append(np.sign(diff))

if len(set(combo_directions)) == 1 and combo_directions[0] != 0:
    robustness_pass = "PASS"
else:
    robustness_pass = "FAIL"

# 4. Dose gate
# "PASS if mean Tau changes in the same direction at every step up in dose, 
# and each step is larger than the spread across 30 s windows within a file."
doses = ["control", "3uM", "10uM", "30uM", "50uM"]
dose_directions = []
step_larger_than_spread = True

for i in range(1, len(doses)):
    prev_cond = doses[i-1]
    curr_cond = doses[i]
    
    prev_mean = np.mean(tau_windows[prev_cond])
    curr_mean = np.mean(tau_windows[curr_cond])
    diff = curr_mean - prev_mean
    dose_directions.append(np.sign(diff))
    
    # Spread within a file -> standard deviation
    prev_spread = np.std(tau_windows[prev_cond])
    curr_spread = np.std(tau_windows[curr_cond])
    
    # "each step is larger than the spread"
    # let's require abs(diff) > prev_spread AND abs(diff) > curr_spread
    if abs(diff) <= prev_spread or abs(diff) <= curr_spread:
        step_larger_than_spread = False

if len(set(dose_directions)) == 1 and dose_directions[0] != 0 and step_larger_than_spread:
    dose_pass = "PASS"
else:
    dose_pass = "FAIL"

# 5. Session gate
# "if a second control recording (or vehicle) exists, PASS if |control2 - control1| < 1/3 x |drug - control1|. If none exists, report "session gate: NOT TESTABLE"."
session_pass = "NOT TESTABLE"


# Output table
print("\n--- METRIC TABLE ---")
print("Condition\tTau(Full)\tTau(WinMean)\tTau(WinStd)")
for cond in doses:
    tf = tau_full[cond][(-5.0, 10)]
    wm = np.mean(tau_windows[cond])
    ws = np.std(tau_windows[cond])
    print(f"{cond}\t{tf:.5f}\t{wm:.5f}\t{ws:.5f}")

print(f"\nRobustness gate: {robustness_pass}")
print(f"Dose gate: {dose_pass}")
print(f"Session gate: {session_pass}")
