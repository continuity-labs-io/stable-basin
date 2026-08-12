I am refactoring this repository from a specific biological research codebase into a general-purpose, pip-installable PyTorch primitive library called `stable_basin`. The goal is to make it look and feel like popular ML libraries (e.g., timm).

Please execute the following restructuring:
1. Create a root package directory named `stable_basin/`.
2. Inside `stable_basin/`, create empty submodules with __init__.py files for our generic primitives: `nn`, `metrics`, and `interpretability`.
3. Move all biological, demo, and simulation code (e.g., `src/demo`, `src/pipeline`, `src/data`, `src/harness`) into a new top-level folder called `examples/`.
4. Generate a modern `pyproject.toml` file to make `stable_basin` pip-installable. Name it `stable-basin`, set version to `0.1.0`. Add dependencies for `torch`, `numpy`, `scipy`, `matplotlib`, and `pydmd`.
5. Set the description to: "Continuous-time state space models and latent thermodynamics for irregular time-series."

include src/config.py in analysis
