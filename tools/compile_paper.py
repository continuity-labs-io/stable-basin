"""
Script to compile the LaTeX paper by injecting data and configurations.
"""

import json
import yaml
import jinja2
import logging
import math
from pathlib import Path
from llm_ghostwriter import LLMGhostwriter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def sig_figs_filter(value, sig_figs):
    if value == 0:
        return "0"
    try:
        value = float(value)
        precision = int(sig_figs - math.floor(math.log10(abs(value))) - 1)
        if precision <= 0:
            return str(int(round(value, precision)))
        else:
            return f"{round(value, precision):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)

SYSTEM_PROMPT = """You are a rigorous computational biology researcher writing a paper in LaTeX format. You must ONLY use the empirical numbers provided in the JSON payload. Do not hallucinate external studies, metrics, or calculations. Maintain an objective, academic tone. Do not use markdown formatting; output LaTeX-safe plaintext."""

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
    env.filters['sig_figs'] = sig_figs_filter
    
    template = env.get_template("sharpening_the_tack.tex.j2")
    
    # Generate AI content
    logger.info("Drafting AI content...")
    ghostwriter = LLMGhostwriter()
    
    ai_abstract = ghostwriter.draft_section(
        system_prompt=SYSTEM_PROMPT,
        user_prompt="Write a concise, 150-word scientific abstract summarizing the core question and hypothesis.",
        context_data=meta
    )
    
    ai_results_pathology = ghostwriter.draft_section(
        system_prompt=SYSTEM_PROMPT,
        user_prompt="Write a 1-paragraph summary explaining how the Gaussian model failed to detect aging, but the PrecisionWeightedEBM succeeded, specifically citing the Cohen's d and KS-statistic.",
        context_data=metrics_decline
    )
    
    ai_results_intervention = ghostwriter.draft_section(
        system_prompt=SYSTEM_PROMPT,
        user_prompt="Write a 1-paragraph summary explaining the thermodynamic rescue, specifically citing the massive spike in the Hessian trace and the intervention's Cohen's d effect size.",
        context_data=metrics_intervention
    )
    
    # Render template
    logger.info("Rendering LaTeX template...")
    rendered_content = template.render(
        meta=meta,
        config_intervention=config_intervention,
        metrics_decline=metrics_decline,
        metrics_intervention=metrics_intervention,
        ai_abstract=ai_abstract,
        ai_results_pathology=ai_results_pathology,
        ai_results_intervention=ai_results_intervention
    )
    
    # Write output
    output_path = root_dir / "paper" / "sharpening_the_tack.tex"
    logger.info(f"Writing rendered output to {output_path}...")
    with open(output_path, "w") as f:
        f.write(rendered_content)
        
    logger.info("Successfully injected metrics into LaTeX template.")

if __name__ == "__main__":
    main()
