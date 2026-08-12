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
