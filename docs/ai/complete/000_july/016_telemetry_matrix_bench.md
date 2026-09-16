Write a rigorous benchmarking loop that feeds the SpikeProphecy dataloader into
the 'SpikeForecaster'.

Use 'torch.cuda.memory_allocated' and 'time.perf_counter' to track the peak GPU
VRAM usage and the absolute inference latency per batch.

Format the output into a clean, reaksmle CLI table that contrasts the local VRAM
footprint against the theoretical bandwidth required if this data were streamed
to a cloud Kubernetes cluster.
