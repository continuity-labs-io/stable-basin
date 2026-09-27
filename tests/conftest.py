import os

# JAX Apple Silicon Metal GPU support is experimental and often crashes during tests.
# This prevents "unknown attribute code: 22" StableHLO errors during pytest collection.
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
