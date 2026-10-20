import modal
import subprocess

# 1. Define the environment (Modal builds the CUDA/JAX container invisibly)
image = modal.Image.debian_slim(python_version="3.11").pip_install("jax[cuda12]", "wandb", "pyyaml")
app = modal.App("jax-experiments")

# 2. Tell Modal to attach a GPU, mount your data/configs, and securely inject W&B API keys
@app.function(
    image=image, 
    gpu="A10G", # Ask for a cheap GPU (~$1.10/hr)
    timeout=3600, # Hard cutoff at 1 hour so you never overspend
    mounts=[modal.Mount.from_local_dir("./data", remote_path="/root/data"),
            modal.Mount.from_local_dir("./configs", remote_path="/root/configs"),
            modal.Mount.from_local_dir("./src", remote_path="/root/src")], # Adjust to your actual source dir
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def train(config_name: str):
    # 3. Just run your existing training script!
    print(f"Offloading {config_name} to cloud GPU...")
    subprocess.run(["python", "-m", "src.main", "--config", f"/root/configs/{config_name}"])

# The local entrypoint
@app.local_entrypoint()
def main(config: str = "experiment_1.yaml"):
    train.remote(config)
