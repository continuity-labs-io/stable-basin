We are executing the final step: Phase 4: CI/CD, Sanity Checks, & Detox (The Safety Net).

Our goal is to clean out the exploratory "vibe code", lock down our environment using a robust Dockerfile (to prevent mamba-ssm CUDA compilation nightmares for future users), and create a preflight command in the Makefile.

Please execute the following steps:

1. The Great Purge (Archiving Vibe Code)
The src/demo/raw/ directory and several scripts in src/demo/ contain legacy scripts used to find the math. They are cluttering the codebase.

Create a new directory at the root of the project called archive/.

Move the entire src/demo/raw/ folder into archive/raw_demos/.

Move the exploratory 01_ through 05_ scripts from src/demo/ into archive/demos/. (Leave 10_human_rejuvenation_sim.py and 11_ratchet_simulator.py where they are, as they are active simulators).

Delete any empty directories left behind.

2. Lock the Dependencies (requirements.txt)
Ensure the requirements.txt at the root of the project has exactly these standard dependencies (no versions specified except where critical, allowing pip to resolve):

```Plaintext
torch>=2.1.0
numpy
pandas
matplotlib
scipy
h5py
pyyaml
ray[tune]
wandb
pydmd
timm
scikit-learn
torchdiffeq```

3. The Reproducibility Lock (Dockerfile)
Compiling causal-conv1d and mamba-ssm requires a specific CUDA/PyTorch alignment. Create a Dockerfile at the root of the project that uses a stable PyTorch devel image so compilation always works out-of-the-box for researchers evaluating our paper.

Create Dockerfile:

```Dockerfile
# Use a PyTorch base image with CUDA development tools required to compile Mamba/Triton
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Set environment variables to prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Mamba and its dependencies explicitly
# Packaging and Ninja are required for compiling custom CUDA kernels
RUN pip install --no-cache-dir packaging ninja
RUN pip install --no-cache-dir causal-conv1d>=1.2.0
RUN pip install --no-cache-dir mamba-ssm

# Copy the rest of the repository
COPY . .

# Default command
CMD ["bash"]
```

4. The CI/CD Preflight (Makefile)
We need a single command to run our smoke_test.py (which we built in Phase 1) before pushing to GitHub or launching a cloud cluster.

Add the following targets to the Makefile:

```Makefile
.PHONY: preflight docker-build

preflight:
	@echo "Running Preflight Smoke Tests..."
	python -m src.harness.smoke_test
	@echo "Smoke tests passed! The registry is stable."

docker-build:
	@echo "Building Stable Basin Docker Image..."
	docker build -t stable-basin:latest .
```

5. Update the README.md
Replace the "Quickstart: Running the Routes" section in README.md with the new pipeline instructions:

```Markdown
### Quickstart: Push-Button Cloud Execution

Stable Basin uses declarative YAML configurations for distributed parallel execution.

**1. Run the CI/CD Smoke Test**
Ensure your hardware and registry are perfectly configured:
`make preflight`

**2. Execute the Clinical Autopsy Pipeline**
This command spins up parallel workers to evaluate the architectures simultaneously against the pharmacological crash dataset:
`make clinical-autopsy`
*All inference latencies, KSM traces, and MambaLRP Causal Autopsy JSONs will automatically sync to your W&B cloud dashboard.*
```

6. Verification
Run make preflight to confirm the smoke test still passes after the file system cleanup.
