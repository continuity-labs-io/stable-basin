# Telemetry Exhaust

**Context:** We are building a high-performance, asynchronous "Flight Recorder"
dashboard for our continuous-time biological physics engine (Project CHRONOS).
To prevent the visualization from blocking our 20kHz inference loop, we are
decoupling the UI by streaming our metrics to `rerun-sdk`. We need to log the
macroscopic variables defined by the Fedichev-Gruber minimal model of aging: z₀
(Dynamic Response), Z (Entropic Damage), and ε₀ (Critical Recovery Rate).

**Task:** Create a new module at `src/metrics/telemetry_exhaust.py`. Define a
`TelemetryExhaust` class that initializes a Rerun session and provides fast,
non-blocking methods to log our PyTorch tensors and thermodynamic scalars.

**Requirements:**

1. **Initialization:** The `__init__` method should call
   `rr.init("chronos_flight_recorder", spawn=False)`. It should accept a flag to
   either connect to a live TCP viewer (`rr.connect()`) or save to a local file
   (`rr.save()`).
2. **Time Context:** Create a method
   `update_time(self, frame_idx: int, time_sec: float)` that sets both the
   sequence frame and the physical time in seconds using `rr.set_time_sequence`
   and `rr.set_time_seconds`.
3. **Macrostates Logger:** Create a method
   `log_fedichev_macrostates(self, z0_volatility: float, Z_entropic_damage: float, epsilon_0_ksm: float, lle_chaos: float)`.
   This should log each variable as a `rr.TimeSeriesScalar` under the
   hierarchical path `fedichev_macrostates/...` and `early_warning_radar/...`.
4. **Phase Space Logger:** Create a method
   `log_attractor_basin(self, latent_tensor: torch.Tensor)`. It should accept a
   tensor of shape `[Num_Points, 3]` and log it as a 3D point cloud
   (`rr.Points3D`) under the path `consciousness_manifold/attractor_basin`. This
   visualizes the structural geometry of the continuous state.
5. **Telemetry Logger:** Create a method
   `log_infrastructure(self, vram_mb: float, perfusion_rate: float)`. This
   should log the hardware footprint and continuous thermodynamic flux.
6. **Code Quality:** Ensure all PyTorch tensors are moved to CPU and converted
   to NumPy arrays before passing to Rerun to ensure zero-copy efficiency where
   possible. Type hint everything perfectly.

**Dependencies:** `rerun-sdk` needs to be added to `environment.yml` and
`requirements.txt`.
