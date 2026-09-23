import os
import logging
from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset
from src.utils.animation import create_worm_gait_animation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Setting up Worm Gait Animation")
    data_path = "data/worm/EigenWorms_TEST.ts"
    
    if not os.path.exists(data_path):
        logger.error(f"Biological data not found at {data_path}. Please ensure data is present.")
        return

    logger.info("Loading young worm data (baseline)...")
    ds_young = RealEigenwormDataset(data_path, seq_len=500, is_aged=False)
    
    logger.info("Loading aged worm data (thermodynamic degradation)...")
    ds_old = RealEigenwormDataset(data_path, seq_len=500, is_aged=True)
    
    # Take the first sequence from each
    # Shape: (500, 6)
    young_data = ds_young[0].numpy()
    old_data = ds_old[0].numpy()
    
    output_dir = "output/echo/benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "09_worm_gait_animation.gif")
    
    logger.info("Rendering animation (this may take a minute)...")
    
    create_worm_gait_animation(
        young_data=young_data,
        old_data=old_data,
        output_path=output_path,
        frames=500,
        fps=30
    )

if __name__ == "__main__":
    main()
