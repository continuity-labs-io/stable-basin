"""
Script to compile the LaTeX paper by injecting data and configurations.
"""

import json
import yaml
import jinja2
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_yaml(filepath: Path) -> dict:
    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Warning: YAML file not found at {filepath}. Using empty dictionary.")
        return {}

def load_json(filepath: Path) -> dict:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Warning: JSON file not found at {filepath}. Using empty dictionary.")
        return {}

def main():
    root_dir = Path(__file__).resolve().parent.parent
    
    # Define file paths
    meta_path = root_dir / "paper" / "paper_metadata.yaml"
    config_intervention_path = root_dir / "configs" / "worm_gait_intervention.yaml"
    metrics_decline_path = root_dir / "output" / "echo" / "benchmarks" / "06_worm_gait_metrics.json"
    metrics_intervention_path = root_dir / "output" / "echo" / "benchmarks" / "07_worm_gait_intervention_metrics.json"
    
    # Load data sources
    logger.info("Loading data sources...")
    meta = load_yaml(meta_path)
    config_intervention = load_yaml(config_intervention_path)
    metrics_decline = load_json(metrics_decline_path)
    metrics_intervention = load_json(metrics_intervention_path)
    
    # Setup LaTeX-Safe Jinja2 Environment
    logger.info("Setting up Jinja2 environment...")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(root_dir / "paper"),
        block_start_string=r'\BLOCK{',
        block_end_string='}',
        variable_start_string=r'\VAR{',
        variable_end_string='}',
        comment_start_string=r'\#{',
        comment_end_string='}',
    )
    
    template = env.get_template("sharpening_the_tack.tex.j2")
    
    # Render template
    logger.info("Rendering LaTeX template...")
    rendered_content = template.render(
        meta=meta,
        config_intervention=config_intervention,
        metrics_decline=metrics_decline,
        metrics_intervention=metrics_intervention
    )
    
    # Write output
    output_path = root_dir / "paper" / "sharpening_the_tack.tex"
    logger.info(f"Writing rendered output to {output_path}...")
    with open(output_path, "w") as f:
        f.write(rendered_content)
        
    logger.info("Successfully injected metrics into LaTeX template.")

if __name__ == "__main__":
    main()
