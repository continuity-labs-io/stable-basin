We now need to create the Catnap YAML configuration to explicitly force the Roadmap Level 2 dimensionality fixes.

1. **Create the YAML Config (`configs/catnap_experiments.yaml`):**
   - Create a new file with the following configuration:
     ```yaml
     dataset:
       name: "catnap"
       h5_path: "data/catnap/trace_features.h5"
       ebm_seq_len: 20
       intervention_seq_len: 50
       batch_size: 2
     
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
         
         # REQUIRED: Roadmap Level 2 Fix for High Dimensionality
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
         ebm_type: "dense"  # Macro latent is small enough for exact math
     
     evaluation:
       curvature_estimator: "hutchinson"
       hutchinson_probes: 20
       
     experiment:
       seed: 42
       dt: 0.05
       num_runs: 5
       N_steps: 100

     intervention:
       lambda_A: 1.0
       lambda_B: 5.0
       lambda_sweep: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

     optimization:
       learning_rate: 0.0001
       max_grad_norm: 0.1
       max_epochs: 10
       early_stopping_patience: 3

     graph:
       n_steps: 1

     paths:
       output_dir: "output/benchmarks/catnap"
       model_weights: "output/benchmarks/catnap/trained_engine.eqx"
       output_plot: "output/benchmarks/catnap/plot.png"
       output_metrics: "output/benchmarks/catnap/metrics.json"
     ```
