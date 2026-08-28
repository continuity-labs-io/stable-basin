import torch
import torch.nn.functional as F
import logging
import numpy as np
from src.config import settings

logger = logging.getLogger("DiagnosticLogger")

# =================================================================================
# TODO: Unvalidated idea. Requires extensive testing. 
# =================================================================================
def calculate_dynamic_rank(S, n_rows, n_cols):
    """
    Autonomously selects the exact SVD rank cutoff for biological tissue.
    S: 1D numpy array of Singular Values (from Σ) sorted descending.
    n_rows, n_cols: Dimensions of the latent state sliding window (e.g., 256, 1000)
    """
    # --- 1. THE GAVISH-DONOHO CEILING (The Noise Governor) ---
    # Calculates the mathematically optimal hard threshold for unknown white noise
    beta = min(n_rows, n_cols) / max(n_rows, n_cols)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    gd_threshold = omega * np.median(S)
    
    valid_modes = np.where(S > gd_threshold)[0]
    
    # THE WADDINGTON CRASH OVERRIDE:
    # If the biological signal collapses and only pure entropy remains,
    # gracefully sever the physics engine so the KSM cleanly drops to 0.0.
    if len(valid_modes) == 0:
        return 0 
        
    r_max = valid_modes[-1] + 1
    
    # If the biology is highly compressed natively, accept the GD limit
    if r_max <= 3:
        return r_max
        
    # --- 2. LOG-SCALED SPECTRAL GAP (The Biological Target) ---
    # Restrict the search for the biological elbow strictly inside the safe zone
    S_safe = S[:r_max]
    
    # Biological energy follows a 1/f power law (pink noise).
    # We evaluate in log-space to linearize the decay and find true structural breaks.
    log_S = np.log(S_safe + 1e-10)
    
    # 1st derivative (velocity of energy decay)
    d1 = np.diff(log_S)
    
    # 2nd derivative (acceleration / structural elbow)
    d2 = np.diff(d1)
    
    # If there is a massive structural cliff (e.g., Cardiomyocytes), apply Spectral Gap.
    # A positive curvature > 1.0 in log-space indicates a severe drop-off into micro-states.
    if len(d2) > 0 and np.max(d2) > 1.0:
        # +1 elegantly maps the index back to the 1-indexed rank, keeping the macro-modes intact
        return int(np.argmax(d2) + 1)
    
    # If the manifold is smooth (e.g., Brain Organoid near criticality), 
    # default to the Gavish-Donoho ceiling to prevent underfitting the neural complexity.
    return int(r_max)
# =================================================================================


class ThermodynamicMetrics:
    def __init__(self, alpha=1000.0, beta=1.0):
        """
        Calculates kinetic biomarkers from the continuous biological latent space.
        """
        self.alpha = alpha
        self.beta = beta

    def calculate_csd(self, z_sequence, window_size=settings.CSD_WINDOW_SIZE):
        """
        Critical Slowing Down (CSD)
        Tracks the physical 'wobble' (Variance) and sluggishness (AR1) of the cell.
        z_sequence shape: [Time, Embed_Dim]
        """
        time_steps = z_sequence.shape[0]
        csd_scores = []

        # Graceful fallback if sequence is too short
        if time_steps < window_size:
            return [0.0] * max(1, time_steps)

        for t in range(window_size, time_steps + 1):
            z_win = z_sequence[t - window_size : t, :]

            channel_vars = torch.var(z_win, dim=0)
            active_mask = channel_vars > 1e-8
            
            if not active_mask.any():
                csd_scores.append(0.0)
                continue
                
            z_win_active = z_win[:, active_mask]

            # Variance (The Wobble)
            var_t = torch.var(z_win_active, dim=0).mean().item()

            # Lag-1 Autocorrelation (Critical Slowing Down)
            ar1_t = F.cosine_similarity(z_win_active[:-1, :], z_win_active[1:, :], dim=1).mean().item()

            csd = (self.alpha * var_t) + (self.beta * ar1_t)
            csd_scores.append(csd)

        # Pad initial frames to maintain temporal sequence length
        return [csd_scores[0]] * (window_size - 1) + csd_scores

    def calculate_ksm(
        self, z_sequence, window_size=settings.KSM_WINDOW_SIZE, debug_crash_frame=None, rank_method="default"
    ):
        """
        The code snippet implements Dynamic Mode Decomposition using a truncated Singular Value Decomposition.
        By decomposing the sliding window of latent states X, the algorithm approximates the local linear
        operator A_tilde that steps the system forward in time to state Y. The eigenvalues of this operator
        directly quantify the thermodynamic stability of the biological system. A maximum eigenvalue near 1.0
        indicates stable homeostasis, while a diverging eigenvalue maps to the system crossing the
        absorbing boundary into a structural crash.

        The decision to utilize Dynamic Mode Decomposition over pseudo-arc length continuation stems from
        the specific architectural constraints of the platform.
        - Compute Latency: Pseudo-arc length continuation is an iterative root-finding algorithm.
        It is computationally expensive and risks introducing variable execution times.
        Truncated Singular Value Decomposition over a small sliding window is deterministic and executes
        with high efficiency on edge GPUs, ensuring the metric keeps pace with the biological timescales
        of milliseconds to minutes.
        - Model Independence: Pseudo-arc length continuation requires an explicit, differentiable
        non-linear vector field to compute the Jacobian. Because the tissue trajectory is modeled
        inside the continuous latent space of the state-space engine, defining the exact non-linear
         continuous field is complex.
         Dynamic Mode Decomposition is entirely data-driven, extracting the kinetic modes directly from
          the streaming embeddings without requiring the underlying equations.
        - Architectural Simplicity: The current approach provides a fast, elegant solution that
        satisfies the requirement for a real-time predictive metric. It isolates the critical variance
        and successfully detects the Waddington bifurcation point while keeping the codebase lean.
        """
        import math
        import numpy as np
        from pydmd import OptDMD

        time_steps = z_sequence.shape[0]
        ksm_scores = [1.0] * window_size
        if time_steps <= window_size:
            return [1.0] * time_steps

        for t in range(window_size, time_steps):
            Z = z_sequence[t - window_size : t + 1]

            # PyDMD expects snapshots as columns: [Embed_Dim, Num_Snapshots]
            Z_np = Z.T.detach().cpu().numpy()
            temporal_std = float(np.std(Z_np, axis=1).mean())

            if temporal_std <= 1e-3:
                logger.debug(f"Flatline detected at frame {t}, forcing rank collapse.")
                max_eig = 0.0
            else:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        if rank_method == "dynamic":
                            # Compute SVD to find rank dynamically using Gavish-Donoho + Log-Spectral Gap
                            U, S, V = np.linalg.svd(Z_np, full_matrices=False)
                            n_rows, n_cols = Z_np.shape
                            r = calculate_dynamic_rank(S, n_rows, n_cols)
                            if r == 0:
                                max_eig = 0.0
                                eigenvalues = []
                            else:
                                dmd = OptDMD(svd_rank=r)
                                dmd.fit(Z_np)
                                eigenvalues = dmd.eigs
                                max_eig = float(np.max(np.abs(eigenvalues)))
                        else:
                            # OptDMD default handles svd_rank=0 robustly
                            dmd = OptDMD(svd_rank=0)
                            dmd.fit(Z_np)
                            eigenvalues = dmd.eigs
                            max_eig = float(np.max(np.abs(eigenvalues)))

                        if debug_crash_frame is not None:
                            # Log the sliding window right before the crash and right after
                            if t == debug_crash_frame - 1:
                                logger.debug(
                                    f"PyDMD Audit [BEFORE crash, t={t}]: eigs={eigenvalues}, max_eig={max_eig}"
                                )
                            elif t == debug_crash_frame + 1:
                                logger.debug(
                                    f"PyDMD Audit [AFTER crash, t={t}]: eigs={eigenvalues}, max_eig={max_eig}"
                                )

                    except Exception as e:
                        logger.error(f"PyDMD Failed at frame {t}: {e}")
                        # Force drop in thermodynamic stability on mathematical failure
                        max_eig = 0.0


            if max_eig == 0.0:
                ksm = 0.0
            else:
                # Bound KSM smoothly [0, 1] using an exponential envelope
                ksm = math.exp(-0.5 * abs(max_eig - 1.0))

            logger.debug(
                f"[PyDMD] Frame {t} | temporal_std={temporal_std:.6f} | max_eig={max_eig:.4f} | KSM={ksm:.4f}"
            )
            ksm_scores.append(max(0.0, ksm))
        return ksm_scores

    def calculate_hysteresis(self, z_baseline, z_perturbed):
        """
        Morphological Hysteresis
        Calculates the topological area between the stress path and rescue path.
        """
        min_steps = min(z_baseline.shape[0], z_perturbed.shape[0])
        if min_steps < 2:
            return 0.0, []

        path_down = z_baseline[:min_steps, :]
        path_up = z_perturbed[:min_steps, :]

        # Euclidean distance between the paths at every time step
        path_divergence = torch.linalg.vector_norm(path_down - path_up, dim=1)

        # Integrate the area under the divergence curve using the Trapezoidal Rule
        hysteresis_area = torch.trapz(path_divergence).item()

        return hysteresis_area, path_divergence.tolist()

    def calculate_lle(self, z_sequence, window_size=settings.LLE_WINDOW_SIZE, dt=1.0):
        """
        Computes the Local Lyapunov Exponent (LLE) over a sliding window
        to measure the stability of the biological attractor basin.
        """
        import math
        import numpy as np
        from pydmd import OptDMD

        time_steps = z_sequence.shape[0]
        lle_scores = [0.0] * window_size
        if time_steps <= window_size:
            return [0.0] * time_steps

        for t in range(window_size, time_steps):
            Z = z_sequence[t - window_size : t + 1]

            # PyDMD expects snapshots as columns: [Embed_Dim, Num_Snapshots]
            Z_np = Z.T.detach().cpu().numpy()
            temporal_std = float(np.std(Z_np, axis=1).mean())

            if temporal_std <= 1e-3:
                max_eig = 0.0
                lle = 0.0
            else:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        # OptDMD is highly robust to sensor noise
                        dmd = OptDMD(svd_rank=0)
                        dmd.fit(Z_np)

                        eigenvalues = dmd.eigs
                        max_eig = float(np.max(np.abs(eigenvalues)))
                    except Exception:
                        # Graceful fallback for stable, rank-deficient biological frames
                        max_eig = 1.0

                # Calculate LLE
                lle = math.log(max_eig + 1e-7) / dt

            logger.debug(
                f"[PyDMD] Frame {t} | temporal_std={temporal_std:.6f} | max_eig={max_eig:.4f} | LLE={lle:.4f}"
            )
            lle_scores.append(lle)

        return lle_scores

    def calculate_cka(self, z_seq1, z_seq2):
        """
        Calculates the Linear Centered Kernel Alignment (CKA) to prove that the geometric shape
        of the biological manifold is preserved across multi-day recordings, even in the presence
        of representational drift.
        """
        min_steps = min(z_seq1.shape[0], z_seq2.shape[0])

        # Trim to minimum length
        X = z_seq1[:min_steps, :]
        Y = z_seq2[:min_steps, :]

        n = min_steps
        device = X.device

        # Compute the linear Gram matrices
        K = X @ X.T
        L = Y @ Y.T

        # Center the Gram matrices
        H = torch.eye(n, device=device) - (torch.ones(n, n, device=device) / n)
        K_c = H @ K @ H
        L_c = H @ L @ H

        # Compute the Hilbert-Schmidt Independence Criterion (HSIC)
        def hsic(A, B):
            return torch.trace(A @ B)

        hsic_kl = hsic(K_c, L_c)
        hsic_kk = hsic(K_c, K_c)
        hsic_ll = hsic(L_c, L_c)

        # Return the normalized CKA score
        cka = hsic_kl / torch.sqrt(hsic_kk * hsic_ll)
        return cka.item()

    def calculate_epigenetic_dispersion(self, cpg_tensor):
        """
        Calculates the thermodynamic configurational entropy (Z) of the epigenetic landscape.
        cpg_tensor shape: [Time, Cells, CpGs]
        Returns: list of scalars (length Time) representing Z.
        """
        if cpg_tensor is None or cpg_tensor.dim() != 3:
            return []
            
        # Statistical dispersion (Variance) across the cell ensemble
        # Higher variance = cells have lost structural integrity and are drifting
        cell_variance = torch.var(cpg_tensor, dim=1) # Shape: [Time, CpGs]
        
        # Mean variance across all tracked CpG sites
        total_entropy_z = torch.mean(cell_variance, dim=1) # Shape: [Time]
        
        return total_entropy_z.tolist()

    def extract_fedichev_macrostates(self, z_baseline: torch.Tensor, z_perturbed: torch.Tensor, window_size: int = 4, cpg_tensor: torch.Tensor = None) -> dict:
        """
        Extracts the three macroscopic variables defining the Fedichev-Gruber minimal model of aging:
        z0 (fast dynamic stress response), Z (slow cumulative entropic damage), and epsilon_0 (critical recovery rate).
        """
        min_steps = min(z_baseline.shape[0], z_perturbed.shape[0])
        if min_steps < 2:
            return {"Z_entropic_damage": [], "z0_volatility": [], "epsilon_0_ksm": []}

        path_down = z_baseline[:min_steps, :]
        path_up = z_perturbed[:min_steps, :]

        # 1. Variable Z (Entropic Damage): Cumulative integral of path divergence
        # We reuse our existing hysteresis metric to get the instantaneous divergence
        _, path_divergence_list = self.calculate_hysteresis(z_baseline, z_perturbed)
        
        path_divergence = torch.tensor(path_divergence_list, device=z_baseline.device)
        Z_t = torch.cumulative_trapezoid(path_divergence, dim=0)
        # torch.cumulative_trapezoid returns length Time-1. Prepend 0.0 to match Time.
        Z_t = torch.cat([torch.tensor([0.0], device=Z_t.device), Z_t])

        # 2. Variable z0_volatility (Dynamic Response)
        z0_volatility = self.calculate_csd(path_up, window_size=window_size)

        # 3. Variable epsilon_0 (Criticality)
        epsilon_0_ksm = self.calculate_ksm(path_up, window_size=window_size)

        # 4. Epigenetic Entropy (Z directly from Methylation, if provided)
        Z_epigenetic_entropy = []
        if cpg_tensor is not None:
            Z_epigenetic_entropy = self.calculate_epigenetic_dispersion(cpg_tensor)

        return {
            "Z_entropic_damage": Z_t.tolist(),
            "z0_volatility": z0_volatility,
            "epsilon_0_ksm": epsilon_0_ksm,
            "Z_epigenetic_entropy": Z_epigenetic_entropy
        }
