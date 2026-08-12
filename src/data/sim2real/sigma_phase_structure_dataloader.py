import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torchdiffeq import odeint

import logging

logger = logging.getLogger(__name__)


class MorphologicalVectorField(nn.Module):
    def __init__(self, dim=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, 128), nn.Tanh(), nn.Linear(128, dim))

    def forward(self, t, z):
        # t is a scalar tensor of time, z is the state
        # expand t to append to z
        t_vec = torch.ones_like(z[..., :1]) * t
        zt = torch.cat([z, t_vec], dim=-1)
        return self.net(zt)


class SigmaPhaseLoader:
    def __init__(self, target_components=100):
        """
        MELD Sigma (Phase Structure) Dataloader V1
        Scaffold for streaming Quantitative Phase Imaging (QPI) from cloud storage,
        extracting single-cell super-voxels, and compressing to 100D latent vectors.
        """
        self.target_components = target_components
        self.source_url = "s3://czb-open-data/qpi-timelapse/sample_01.zarr"
        logger.info(f"[INIT] Sigma Dataloader targeting proxy: {self.source_url}")

    def fetch_and_segment(self):
        """
        [TASK 1] DATA ENGINEER: CLOUD INGEST & 3D SEGMENTATION
        Connect to an actual public AWS OME-Zarr dataset using `zarr` and `dask`.
        Implement `cellpose` to isolate a single 'Super-Voxel' (Cell) from the 3D grid.
        """
        logger.info("[NETWORK] Mocking lazy stream from OME-Zarr store...")
        logger.info("[COMPUTE] Mocking Cellpose3D centroid extraction...")

        # Simulating the final flattened feature extraction for 1 cell over 15 minutes.
        # Real microscopes might take a snapshot every 1 minute.
        # Shape: (16 time steps, 800 raw spatial features like volume, optical density, etc.)
        time_minutes = np.arange(0, 16, 1.0)

        # Random walk to simulate structural shape drift
        raw_spatial_features = np.cumsum(np.random.normal(0, 0.1, (len(time_minutes), 800)), axis=0)

        return time_minutes, raw_spatial_features

    def compress_to_latent(self, raw_spatial_features):
        """
        [TASK 2] DATA ENGINEER: THE SPATIAL VAE (sVAE) COMPRESSOR
        Upgrade this from basic PCA to a PyTorch Spatial Variational Autoencoder
        to capture non-linear structural topology.
        """
        logger.info(
            f"[ML] Compressing {raw_spatial_features.shape[1]} raw features to {self.target_components} dimensions..."
        )

        # We use PCA here just to build the V1 plumbing and prove the API works
        pca = PCA(n_components=min(self.target_components, raw_spatial_features.shape[0]))
        latent_vectors = pca.fit_transform(raw_spatial_features)

        # Pad with zeros if we have fewer time-steps than target components (for the dummy run)
        if latent_vectors.shape[1] < self.target_components:
            padding = np.zeros(
                (latent_vectors.shape[0], self.target_components - latent_vectors.shape[1])
            )
            latent_vectors = np.hstack((latent_vectors, padding))

        return latent_vectors

    def align_to_master_clock(self, time_minutes, latent_vectors, master_time_ms):
        """
        [TASK 3] DATA ENGINEER: THE MULTI-SCALE TIME BRIDGE
        The Sigma laser takes 1 picture every minute.
        The Omega laser fires at 500 Hz (every 2 milliseconds).
        How do we map the slow 1-minute shape data onto the 2ms electrical grid?
        """
        logger.info(
            "[ALIGNMENT] Interpolating slow morphology to the 500Hz master clock using Piecewise Neural ODE..."
        )

        # Convert master clock to minutes
        master_time_min = master_time_ms / 60000.0
        aligned_sigma = np.zeros((len(master_time_min), self.target_components))

        # --- PIECEWISE CONTINUOUS-TIME NEURAL ODE ---
        # Instead of cubic splines, we integrate the biological velocity field.
        # We piecewise evaluate each 1-minute interval, anchored at the true observation.
        vector_field = MorphologicalVectorField(dim=self.target_components)
        vector_field.eval()

        latent_vectors_tensor = torch.tensor(latent_vectors, dtype=torch.float32)

        with torch.no_grad():
            for i in range(len(time_minutes) - 1):
                t_start = time_minutes[i]
                t_end = time_minutes[i + 1]
                z_start = latent_vectors_tensor[i]

                # Group master evaluation times falling into this interval
                if i == len(time_minutes) - 2:
                    # Extrapolate for the final observation onwards
                    mask = master_time_min >= t_start
                else:
                    mask = (master_time_min >= t_start) & (master_time_min < t_end)

                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue

                eval_times_np = master_time_min[indices]
                eval_times_t = torch.tensor(eval_times_np, dtype=torch.float32)

                # odeint requires the first time in t_eval to be the time of the initial state.
                if not torch.isclose(
                    eval_times_t[0], torch.tensor(t_start, dtype=torch.float32), atol=1e-5
                ):
                    t_eval = torch.cat([torch.tensor([t_start], dtype=torch.float32), eval_times_t])
                    skip_first = True
                else:
                    t_eval = eval_times_t
                    skip_first = False

                # Integrate the continuous biological velocity dz(t)/dt = f(z(t), t)
                pred_z = odeint(vector_field, z_start, t_eval)

                if skip_first:
                    pred_z = pred_z[1:]

                aligned_sigma[indices] = pred_z.numpy()

        # Format output
        cols = [f"Sigma_PC{i:03d}" for i in range(1, self.target_components + 1)]
        df_sigma = pd.DataFrame(aligned_sigma, columns=cols)
        df_sigma.insert(0, "Time_ms", master_time_ms)

        return df_sigma


# ==========================================
# EXECUTION (Drop this in the Jupyter Notebook)
# ==========================================
if __name__ == "__main__":
    # 1. Boot the Master MELD Clock
    # Simulating 15 minutes. A 500Hz burst (2ms) for 4.5 seconds every 5 minutes.
    logger.info("Initializing MELD Master Clock...")
    master_clock_ms = []
    for minute in [0, 5, 10, 15]:
        # 4.5 seconds of 500Hz = 2250 frames per burst
        burst = np.linspace(minute * 60000, minute * 60000 + 4500, 2250)
        master_clock_ms.extend(burst)
    master_clock_ms = np.array(master_clock_ms)

    # 2. Run the Dataloader
    loader = SigmaPhaseLoader()
    times, raw_feats = loader.fetch_and_segment()
    latents = loader.compress_to_latent(raw_feats)
    final_df = loader.align_to_master_clock(times, latents, master_clock_ms)

    logger.info("\n[SUCCESS] Sim2Real Sigma Tensor Generated.")
    logger.info(final_df.head())
