# Refactor Tools Directory

## Objective
Refactor the `tools/` directory to improve organization by extracting code auditing tools and paper generation tools into their own dedicated packages, while keeping shared utilities at the root of `tools/`.

## Steps

1. **Create Package Directories**:
   - Create `tools/code_auditor/`
   - Create `tools/paper/`
   - Create `__init__.py` files in both directories to ensure they act as proper Python packages.

2. **Move Code Auditor Files**:
   - Move `tools/multi_agent_auditor.py` to `tools/code_auditor/`
   - Move `tools/personas.py` to `tools/code_auditor/`

3. **Move Paper Files**:
   - Move `tools/compile_paper.py` to `tools/paper/`
   - Move `tools/llm_ghostwriter.py` to `tools/paper/`

4. **Retain Shared Utilities**:
   - Keep `tools/genai_client.py` in the root `tools/` directory as it is generic and shared across multiple tools.

5. **Update Imports**:
   - In `tools/paper/llm_ghostwriter.py`, update the import of `genai_client` to absolute imports (e.g., `from tools.genai_client import get_client, get_best_model`).
   - *Note: Ensure no `sys.path.append` hacks are used, as per repository invariants. The repository is installed via `pip install -e .`.*

6. **Validation**:
   - Run syntax checks on all moved Python files.
   - Run existing unit tests or smoke tests to confirm import paths resolve correctly.
   - Run `make paper`, `make audit`, and `make preflight` and ensure they all pass.
   
