# Migration to NVIDIA GPU for JAX Pipeline

## Background
The `stable-basin` killifish aging pipeline currently utilizes a hybrid PyTorch and JAX stack. While the PyTorch modules (`baseline_transformer`, etc.) seamlessly execute on Apple Silicon (`mps`), the JAX pipeline hits fundamental limitations with the `jax-metal` plugin.

## Technical Issue
The script `06_fit_lambda.py` relies on `jnp.linalg.cholesky()` (which maps to XLA's `mhlo.cholesky`) in `hierarchical_factor.py`. The Apple `jax-metal` backend currently does not support this operation, leading to the following hard crash:
```
jax.errors.JaxRuntimeError: UNKNOWN: /Users/ry/gh/stable-basin/src/echo/architecture/hierarchical_factor.py:109:18: error: failed to legalize operation 'mhlo.cholesky'
```

## Proposed Solution
Because the core math requires continuous state-space linear algebra operations unsupported by Apple Silicon's XLA compiler, we must migrate the execution of the JAX stages (scripts `04`, `05`, and `06`) to an **NVIDIA GPU instance in the cloud** (e.g., AWS EC2, GCP, RunPod).

### Required Changes for Cloud Execution
1. **Dockerization / Environment Setup**: Provide a `Dockerfile` with the appropriate CUDA/CuDNN drivers and `jax[cuda]` installation.
2. **Data Sync**: Implement an S3/GCS sync mechanism so the output of `03_aging_transformer.py` (and the `killifish_experiments` dataset) can be handed off to the cloud worker.
3. **Runner Script Mod**: Ensure the runner pipeline can delegate specific pipeline stages to the remote environment.
