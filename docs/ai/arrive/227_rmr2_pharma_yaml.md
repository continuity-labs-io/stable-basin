We are implementing the RMR2 Surrogate Endpoint simulation. First, let's create the configuration file mapping the biological drugs to the physics engine.

1. Create a new file `configs/rmr2_simulation.yaml`.
2. Populate it with this exact configuration:
```yaml
dataset:
  name: "catnap"
  h5_path: "data/catnap/trace_features.h5"
  ebm_seq_len: 20

observer:
  micro:
    d_internal: 128
    d_sensory: 309
    d_active: 64
    d_external: 64
    ebm_hidden_size: 256
    ebm_depth: 3
    temperature: 1.0
    n_steps: 1
    ebm_type: "structured_low_rank"
    precision_rank: 8
  macro:
    d_internal: 16
    d_sensory: 8
    d_active: 4
    d_external: 4
    ebm_hidden_size: 64
    ebm_depth: 2
    temperature: 1.0
    n_steps: 1
    ebm_type: "dense"

experiment:
  seed: 42
  dt: 0.05
  num_runs: 5
  N_steps: 100

rmr2_pharmacology:
  baseline_shift:
    # Rapamycin + Exercise provides a global homeostatic boost
    Pi: 0.1
    Gamma: 0.1
    T: 0.1  # (T reduction)
    Q: 0.1
  drugs:
    partial_reprogramming: {mechanism: "Pi", effect: 0.5}
    mscs: {mechanism: "Pi", effect: 0.3}
    d_pufas: {mechanism: "Gamma", effect: 0.4}
    lc_facs: {mechanism: "Gamma", effect: 0.3}
    il11_inhibitor: {mechanism: "T", effect: 0.4}
    casin: {mechanism: "T", effect: 0.3}
    oxytocin: {mechanism: "Q", effect: 0.4}
    rmsa: {mechanism: "Q", effect: 0.2}
  arms:
    - {name: "00_Control_Aged", drugs: [], base_treated: false}
    - {name: "01_Base_Rapa_Exer", drugs: [], base_treated: true}
    - {name: "02_Base_MSCs", drugs: ["mscs"], base_treated: true}
    - {name: "03_Base_IL11", drugs: ["il11_inhibitor"], base_treated: true}
    - {name: "04_Base_DPUFAs", drugs: ["d_pufas"], base_treated: true}
    - {name: "05_Base_Reprogramming", drugs: ["partial_reprogramming"], base_treated: true}
    - {name: "06_Base_Oxy", drugs: ["oxytocin"], base_treated: true}
    - {name: "07_LeaveOut_Reprogramming", drugs: ["mscs", "d_pufas", "lc_facs", "il11_inhibitor", "casin", "oxytocin", "rmsa"], base_treated: true}
    - {name: "08_LeaveOut_IL11", drugs: ["partial_reprogramming", "mscs", "d_pufas", "lc_facs", "casin", "oxytocin", "rmsa"], base_treated: true}
    - {name: "09_LeaveOut_DPUFAs", drugs: ["partial_reprogramming", "mscs", "lc_facs", "il11_inhibitor", "casin", "oxytocin", "rmsa"], base_treated: true}
    - {name: "10_All_8_Drugs", drugs: ["partial_reprogramming", "mscs", "d_pufas", "lc_facs", "il11_inhibitor", "casin", "oxytocin", "rmsa"], base_treated: true}

paths:
  output_dir: "output/benchmarks/rmr2"
  model_weights: "output/benchmarks/aging_resilience/05_worm_gait_decline_trained_engine.eqx"
