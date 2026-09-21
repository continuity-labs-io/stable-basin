"""
Script to compile the LaTeX paper by injecting data and configurations.
"""

import json
import yaml
import jinja2
import logging
import argparse
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
    parser = argparse.ArgumentParser(description="Compile LaTeX paper.")
    parser.add_argument("--paper", type=str, default="sharpening_the_tack", help="Name of the paper directory")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    paper_dir = root_dir / "paper" / args.paper
    
    # Define file paths
    meta_path = paper_dir / "paper_metadata.yaml"
    
    # Load metadata
    logger.info("Loading paper metadata...")
    meta = load_yaml(meta_path)
    
    context = {"meta": meta}
    
    # Load data sources
    logger.info("Loading data sources...")
    for key, rel_path in meta.get("data_sources", {}).items():
        filepath = root_dir / rel_path
        if filepath.suffix in ['.yaml', '.yml']:
            context[key] = load_yaml(filepath)
        elif filepath.suffix == '.json':
            context[key] = load_json(filepath)
        else:
            logger.warning(f"Unknown file extension for data source {key}: {filepath}")
    
    
    # Setup LaTeX-Safe Jinja2 Environment
    logger.info("Setting up Jinja2 environment...")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(paper_dir),
        block_start_string=r'\BLOCK{',
        block_end_string='}',
        variable_start_string=r'\VAR{',
        variable_end_string='}',
        comment_start_string=r'\#{',
        comment_end_string='}',
    )
    env.filters['sig_figs'] = sig_figs_filter
    
    template = env.get_template(f"{args.paper}.tex.j2")
    
    # Generate AI content
    logger.info("Drafting AI content...")
    ghostwriter = LLMGhostwriter()
    ghostwriter_results = {}
    
    for key, section in meta.get("ghostwriter_sections", {}).items():
        logger.info(f"Drafting AI content for {key}...")
        ctx_key = section.get("context", "meta")
        ctx_data = context.get(ctx_key, {})
        ghostwriter_results[key] = ghostwriter.draft_section(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=section["prompt"],
            context_data=ctx_data
        )
    
    # Render template
    logger.info("Rendering LaTeX template...")
    rendered_content = template.render(
        **context,
        **ghostwriter_results
    )
    
    # Write output
    output_path = paper_dir / f"{args.paper}.tex"
    logger.info(f"Writing rendered output to {output_path}...")
    with open(output_path, "w") as f:
        f.write(rendered_content)
        
    logger.info("Successfully injected metrics into LaTeX template.")

if __name__ == "__main__":
    main()
