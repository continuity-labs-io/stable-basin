import os
import logging
import yaml
from src.benchmarks.aging_resilience.task_registry import get_benchmark_task

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Setting up Worm Gait Animation")
    
    config_path = "configs/aging_resilience.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    task = get_benchmark_task(config)
    
    logger.info("Loading data via task...")
    _, ds_young, ds_old = task.get_raw_datasets(config)
    
    # Take the first sequence from each
    young_data = ds_young[0].numpy() if hasattr(ds_young[0], 'numpy') else ds_young[0]
    old_data = ds_old[0].numpy() if hasattr(ds_old[0], 'numpy') else ds_old[0]
    
    output_dir = "output/benchmarks/aging_resilience"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "10_worm_gait_animation.gif")
    
    logger.info("Rendering animation (this may take a minute)...")
    
    task.render_animation(
        young_data=young_data,
        old_data=old_data,
        output_path=output_path,
        frames=500,
        fps=30
    )

if __name__ == "__main__":
    main()

