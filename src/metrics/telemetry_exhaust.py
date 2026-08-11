import rerun as rr
import torch
import numpy as np

class TelemetryExhaust:
    """
    Asynchronous bridge between the continuous-time physics engine and the Rerun viewer.
    Logs macroscopic variables and high-dimensional phase space without blocking inference.
    """
    
    def __init__(self, mode: str = "connect", save_path: str = "flight_recorder.rrd"):
        """
        Initializes the Rerun session.
        Args:
            mode: "connect" to connect to a live TCP viewer, "save" to save to a local file.
            save_path: The file path to save the recording if mode is "save".
        """
        rr.init("stable_basin_telemetry", spawn=False)
        
        if mode == "connect":
            rr.connect()
        elif mode == "save":
            rr.save(save_path)
        else:
            raise ValueError("Mode must be 'connect' or 'save'.")

    def update_time(self, frame_idx: int, time_sec: float) -> None:
        """
        Sets the global time context for all subsequent logs.
        """
        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("time_sec", duration=time_sec)

    def log_fedichev_macrostates(
        self, 
        z0_volatility: float, 
        Z_entropic_damage: float, 
        epsilon_0_ksm: float, 
        lle_chaos: float
    ) -> None:
        """
        Logs the macroscopic variables defined by the Fedichev-Gruber minimal model.
        """
        rr.log("fedichev_macrostates/z0_volatility", rr.Scalars(z0_volatility))
        rr.log("fedichev_macrostates/Z_entropic_damage", rr.Scalars(Z_entropic_damage))
        rr.log("fedichev_macrostates/epsilon_0_ksm", rr.Scalars(epsilon_0_ksm))
        rr.log("early_warning_radar/lle_chaos", rr.Scalars(lle_chaos))

    def log_attractor_basin(self, latent_tensor: torch.Tensor) -> None:
        """
        Logs the phase space geometry as a 3D point cloud.
        """
        # Ensure tensor is on CPU and converted to numpy for zero-copy efficiency
        if isinstance(latent_tensor, torch.Tensor):
            points = latent_tensor.detach().cpu().numpy()
        else:
            points = np.asarray(latent_tensor)
            
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected shape [Num_Points, 3], got {points.shape}")
            
        rr.log("consciousness_manifold/attractor_basin", rr.Points3D(points))

    def log_infrastructure(self, vram_mb: float, perfusion_rate: float) -> None:
        """
        Logs hardware footprint and continuous thermodynamic flux.
        """
        rr.log("infrastructure/vram_mb", rr.Scalars(vram_mb))
        rr.log("infrastructure/perfusion_rate", rr.Scalars(perfusion_rate))
