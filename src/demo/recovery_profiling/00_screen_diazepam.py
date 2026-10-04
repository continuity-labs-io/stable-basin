import os
import h5py
import numpy as np
import scipy.signal as signal
from scipy.stats import median_abs_deviation
import torch
import pandas as pd
from src.metrics.entropy_production import entropy_production_mou
from src.metrics.spectral import SpectralMetrics
from pydmd import OptDMD
from sklearn.decomposition import PCA

def detect_spikes(filepath, fs=20000.0, chunk_size=200000):
    with h5py.File(filepath, "r") as f:
        sig = f["sig"]
        n_channels, n_samples = sig.shape
        n_channels = min(1024, n_channels)
        
        sos = signal.butter(3, [300, 3000], btype='bandpass', fs=fs, output='sos')
        
        print("Estimating MAD...")
        chunk0 = sig[:n_channels, :chunk_size].astype(np.float32)
        chunk0_filt = signal.sosfiltfilt(sos, chunk0, axis=1)
        
        # Calculate MAD:
        median_val = np.median(chunk0_filt, axis=1, keepdims=True)
        mad = np.median(np.abs(chunk0_filt - median_val), axis=1) / 0.6745
        # Avoid zero mad
        mad = np.maximum(mad, 1e-6)
        thresholds = -5.0 * mad
        
        bin_10ms = int(fs * 0.01)
        bin_100ms = int(fs * 0.1)
        n_bins_10ms = int(np.ceil(n_samples / bin_10ms))
        n_bins_100ms = int(np.ceil(n_samples / bin_100ms))
        
        pfr_10ms = np.zeros(n_bins_10ms)
        rates_100ms = np.zeros((n_channels, n_bins_100ms))
        
        print("Processing chunks...")
        for start in range(0, n_samples, chunk_size):
            end = min(n_samples, start + chunk_size)
            chunk = sig[:n_channels, start:end].astype(np.float32)
            chunk_filt = signal.sosfiltfilt(sos, chunk, axis=1)
            
            is_below = chunk_filt < thresholds[:, None]
            # local minimum check
            local_min = np.zeros_like(chunk_filt, dtype=bool)
            local_min[:, 1:-1] = (chunk_filt[:, 1:-1] < chunk_filt[:, :-2]) & (chunk_filt[:, 1:-1] < chunk_filt[:, 2:])
            spikes = is_below & local_min
            
            ch_idx, t_idx = np.where(spikes)
            t_global = t_idx + start
            
            bin_idx_10 = t_global // bin_10ms
            np.add.at(pfr_10ms, bin_idx_10, 1)
            
            bin_idx_100 = t_global // bin_100ms
            np.add.at(rates_100ms, (ch_idx, bin_idx_100), 1)
            
        return pfr_10ms, rates_100ms, n_channels

def get_metrics(pfr_10ms, rates_100ms, n_channels):
    # pfr_10ms is (bins,)
    # rates_100ms is (n_channels, bins)
    
    # basic
    mean_rate = np.mean(pfr_10ms) / 0.01 / n_channels
    thresh_burst = np.mean(pfr_10ms) + 3*np.std(pfr_10ms)
    burst_rate = np.sum(pfr_10ms > thresh_burst) / (len(pfr_10ms) * 0.01)
    sync = np.std(pfr_10ms) / (np.mean(pfr_10ms) + 1e-6)
    variance_pfr = np.var(pfr_10ms)
    
    # AC lag-1
    if len(pfr_10ms) > 1:
        r1 = np.corrcoef(pfr_10ms[:-1], pfr_10ms[1:])[0,1]
        tau_ac = -0.01 / np.log(np.clip(r1, 1e-6, 0.9999)) if r1 > 0 else 0
    else:
        tau_ac = 0
        
    # DMD on top 20 PCs
    n_bins_100 = rates_100ms.shape[1]
    rates_100ms_norm = rates_100ms - np.mean(rates_100ms, axis=1, keepdims=True)
    if rates_100ms_norm.shape[1] > 20:
        pca = PCA(n_components=20)
        X_pca = pca.fit_transform(rates_100ms_norm.T).T # (20, bins)
    else:
        X_pca = rates_100ms_norm
    
    tau_dmd = 0.0
    try:
        dmd = OptDMD(svd_rank=min(20, X_pca.shape[1]-1))
        dmd.fit(X_pca)
        lambdas = dmd.eigs
        valid = np.abs(lambdas) < 0.9999
        if np.any(valid):
            max_lam = np.max(np.abs(lambdas[valid]))
            tau_dmd = -0.1 / np.log(max_lam)
    except Exception as e:
        print(f"DMD failed: {e}")
        
    # Existing repo code
    spec = SpectralMetrics()
    # psd
    freqs, psd = spec.calculate_psd(torch.tensor(pfr_10ms, dtype=torch.float32), 100.0)
    psd_power = torch.sum(psd).item()
    
    # PLV and PAC on top 2 PCs
    if X_pca.shape[0] >= 2:
        pc1 = torch.tensor(X_pca[0], dtype=torch.float32)
        pc2 = torch.tensor(X_pca[1], dtype=torch.float32)
        plv = spec.calculate_plv(pc1, pc2).item()
        pac = spec.calculate_cfc_pac(pc1, pc2).item()
    else:
        plv, pac = 0.0, 0.0
        
    # MOU Entropy
    mou_res = entropy_production_mou(X_pca.T, fs=10.0, lag_samples=1)
    mou_phi = mou_res.phi
    
    return {
        'Mean Firing Rate': mean_rate,
        'Burst Rate': burst_rate,
        'Synchrony (CV)': sync,
        'Variance (PFR)': variance_pfr,
        'Tau (AC)': tau_ac,
        'Tau (DMD)': tau_dmd,
        'Total PSD Power': psd_power,
        'PLV (PC1/PC2)': plv,
        'PAC (PC1/PC2)': pac,
        'MOU Entropy (PCs)': mou_phi
    }

print("Processing Control...")
pfr_10_c, rates_100_c, ch_c = detect_spikes('data/ephys/pharmacological_shock/Drug_2953_control.raw.h5')
print("Processing Drug...")
pfr_10_d, rates_100_d, ch_d = detect_spikes('data/ephys/pharmacological_shock/Drug_2953_10uM.raw.h5')

# We need A, B, D segments
n_bins10_c = len(pfr_10_c)
A_10 = pfr_10_c[:n_bins10_c//2]
B_10 = pfr_10_c[n_bins10_c//2:]

n_bins100_c = rates_100_c.shape[1]
A_100 = rates_100_c[:, :n_bins100_c//2]
B_100 = rates_100_c[:, n_bins100_c//2:]

n_bins10_d = len(pfr_10_d)
D_10 = pfr_10_d[n_bins10_d//2:]

n_bins100_d = rates_100_d.shape[1]
D_100 = rates_100_d[:, n_bins100_d//2:]

print("Computing metrics A...")
m_A = get_metrics(A_10, A_100, ch_c)
print("Computing metrics B...")
m_B = get_metrics(B_10, B_100, ch_c)
print("Computing metrics D...")
m_D = get_metrics(D_10, D_100, ch_d)

table = []
for k in m_A.keys():
    a = m_A[k]
    b = m_B[k]
    d = m_D[k]
    
    mean_ab = (a + b) / 2
    diff_ab = abs(a - b)
    diff_d = abs(d - mean_ab)
    
    # PASS if |D - mean(A,B)| >= 3 x |A - B|
    # edge case: if diff_ab == 0
    if diff_ab == 0:
        if diff_d > 0:
            ratio = float('inf')
            is_pass = 'PASS'
        else:
            ratio = 0.0
            is_pass = 'FAIL'
    else:
        ratio = diff_d / diff_ab
        is_pass = 'PASS' if ratio >= 3 else 'FAIL'
        
    table.append({
        'Metric': k,
        'A': a,
        'B': b,
        'D': d,
        'Ratio': ratio,
        'Result': is_pass
    })

df = pd.DataFrame(table)
print("\n--- RESULTS ---")
print(df.to_string(index=False))
