# Implement ODE-RNN and GRU-D Learning Models

Implement an exact ODE-RNN (using `torchdiffeq`) and a GRU-D. Evaluate them against the FHN simulator.

## Architectural Assessment
Yes, both **ODE-RNN** and **GRU-D** are sequential deep learning models designed specifically for irregularly sampled time series. 
- **GRU-D** utilizes learned exponential decays between observations based on the time elapsed ($\Delta t$) since the last valid observation.
- **ODE-RNN** uses a Neural ODE (via `torchdiffeq`) to continuously integrate the hidden state between observations, applying standard RNN cell updates when an observation occurs.

## Harness Integration Strategy

It will be extremely easy to hook these into our existing `SensorFusionPredictor` and benchmark harness!

### 1. Extending the Enum
We simply add them to our `SSMType` enum in `src/models/simulators/sensor_fusion_predictor.py`:
```python
class SSMType(str, Enum):
    # ... existing models ...
    GRU_D = "gru_d"
    ODE_RNN = "ode_rnn"
```

### 2. Time-Delta Calculation in the Predictor
Both models require the time elapsed since the last observation ($\Delta t$). Since our harness passes `x_raw` and `mask`, we can dynamically compute the time deltas directly inside the `SensorFusionPredictor.forward()` method (similar to our `forward_fill` logic).

```python
if self.ssm_type in ["gru_d", "ode_rnn"]:
    # Compute time deltas dynamically based on the mask
    # For every step where mask == 0, delta_t increments.
    # Where mask == 1, delta_t resets to 0.
    delta_t = compute_time_deltas(mask)
    
    latent_x, latent_gate = self.fusion(x_raw, mask)
    h = self.ssm(latent_x, delta_t)
```

### 3. Harness & Makefile
Because our harness dynamically instantiates models from the CLI arguments, we won't need to change `sensor_fusion_runner.py` at all! We just add them to the `--models` CLI argument in our Makefile:
```makefile
--models baseline mask_aware transformer gru_d ode_rnn \
```
