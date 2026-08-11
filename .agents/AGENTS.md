### Staging Prompts Guardrail
- **Read-Only Directory (`docs/ai/accepted/`)**: Do not modify or edit any markdown files within `docs/ai/accepted/`. These are staging prompts intended for human review before being coded. Treat this directory as strictly read-only.

### Output Directory Structure
- **Mirrored Paths**: When scripts generate output files (e.g., plots, CSVs), they should save them in a directory structure within `output/` that mirrors the source script's path. For example, a script running from `src/demo/` should save its outputs to `output/demo/`. Ensure the output directories are created if they do not exist.
