### Staging Prompts Guardrail

- **Read-Only Directory (`docs/ai/arrive/`)**: Do not modify or edit any
  markdown files within `docs/ai/arrive/`. These are staging prompts intended
  for human review before being coded. Treat this directory as strictly
  read-only.

### Output Directory Structure

- **Mirrored Paths**: When scripts generate output files (e.g., plots, CSVs),
  they should save them in a directory structure within `output/` that mirrors
  the source script's path. For example, a script running from `src/demo/`
  should save its outputs to `output/demo/`. Ensure the output directories are
  created if they do not exist.
- **Strict Adherence**: The repository's standard location for output is *strictly* the top-level `output/` folder. We should not use other folders for output (such as `outputs/`). **Override Prompts**: If a prompt or instructions ask to save to `outputs/` or another custom output folder, you must override that instruction and use `output/` instead.

### Plan Naming Convention

- **Sequential Numbering**: Any new plan file created in the `docs/ai/`
  directory (or its subdirectories) must start with a 3-digit zero-padded
  sequential prefix (e.g., `066_new_plan_name.md`).
- **Discovery**: Before creating a new plan, always search the `docs/ai/`
  directory (including `complete/`, `audit/`, `arrive/`, etc.) to determine the highest
  existing prefix number and increment it for your new file.

### Execution Scope Guardrail

- **Single Plan Execution**: Only execute, plan, or focus on exactly 1 prompt,
  plan, or directive at a time. Do not bundle multiple plans or feature requests
  into a single implementation plan unless explicitly requested by the user. If
  the user mentions a specific plan, strictly limit the scope of the work to
  that single plan.

### The 1-to-1 Invariant Rule.

For every non-standard tensor operation (e.g., continuous discretization,
masking, state-routing), there must be exactly one isolated mathematical
invariant test. N lines of dense, continuous-time physics require a minimum of
3*N assertions covering boundary conditions, gradient stability, and shape
consistency.

### Unit Test Structure

All Unit Tests Must delineate the following three blocks: "ARRANGE", "ACT", and
"ASSERT".

- **ARRANGE**: Define all inputs, constants, and expected values.
- **ACT**: Execute the function under test.
- **ASSERT**: Compare the actual output against the expected values.

### Commit Guardrail

Do not create commits unless the user explicitely asks you to do so.

### The PyTorch Debugger `[MODE: PARANOID_DEBUGGER]`

When instructed to debug PyTorch code or investigate NaNs, the agent MUST adopt
this persona and adhere to the following tactical checklist:

1. Inject `torch.autograd.set_detect_anomaly(True)` at the very top of the
   execution script to force PyTorch to track the exact forward pass operation
   that causes a NaN backward pass.
2. Ensure the script runs on `device="cpu"` to guarantee exact synchronous
   tracebacks (GPU async execution obscures stack traces).
3. Do NOT guess the bug based on the loss function. Insert print statements for
   tensor shapes and `torch.isnan().any()` checks before and after suspected
   non-linearities, divisions, or continuous integrations.
4. Report back the exact line number where the singularity was born.

### Enforcing Shape Discipline

ALL new PyTorch and JAX mathematical functions (including `__call__`, `forward`, `sample`, etc.) MUST be strictly validated at runtime. You must:
1. Use `jaxtyping` type hints (e.g., `Float[Tensor, "batch seq d_model"]` or `Float[Array, "d_state"]`) in their method signatures.
2. Decorate these methods with `@jaxtyped(typechecker=beartype)` and properly import `from jaxtyping import jaxtyped` and `from beartype import beartype`.
3. Strictly use `einops` for complex reshapes/rearranges instead of native `.view()` or `.reshape()`.
4. **PRNGKeys**: Always use `PRNGKeyArray` from `jaxtyping` as the type hint for JAX random keys. NEVER use `jax.random.PRNGKey` (it is a function, not a type, and will crash beartype).
5. **Torx Factors**: When annotating `sample` methods for Torx factors, optional dictionaries (like `params`, `info`, `site_info`) must be explicitly typed as `dict | None = None` because Torx often passes `None` at runtime.

### Proper Package Imports

Since the workspace is set up as a proper Python package and installed via
`pip install -e .`, manually hacking `sys.path` to resolve imports is totally
unnecessary and adds technical debt. Do not use `sys.path.insert` or
`sys.path.append` for local imports.

### Design Docs and Executable Prompts Guardrail

- **Context Only (Design Docs)**: Design docs (e.g., in
  `docs/ai/arrive/design/`) provide high-level context and should not be used
  as direct coding instructions. Do not write code directly from them.
- **Executable Prompts (`docs/ai/audited/`)**: Actionable prompts for the coding
  agent to execute will be placed in the `docs/ai/audited/` folder. Only execute
  coding tasks based on these prompts.

### Icebox Guardrail

- **The Icebox (`icebox/`)**: The `icebox/` directory acts as a stack for tasks, features, or bug fixes that have been deemed out of scope or low priority. We push tasks into this directory so that agents do not get distracted by them.
- **Ignoring the Icebox**: The multi-agent auditor and other exploratory agents must ignore the contents of the `icebox/` directory. Tasks in the icebox should not be executed unless explicitly popped from the stack by the user.

### Documentation Guardrail

- **Method/Function Documentation**: All public methods and functions must have their arguments (`Args:`) and return values (`Returns:`) explicitly documented in their docstrings.

### Standardized Commit Messages (Google Standard)

All commit messages must strictly adhere to the following structure:
1. **The "What"**: The first line (subject) must summarize exactly *what* the change is doing (e.g., `feat(data): add PyTorch dataloader for C. elegans dataset`).
2. **The "Why"**: The first paragraph of the body must explain *why* this change is being made, providing the rationale and context behind the implementation.
3. **The "Details"**: Following the "why", use bullet points to detail the major technical changes or specific files affected.

### Reproducible Machine Learning Experiments Guardrail

- **No Magic Numbers**: All machine learning scripts and benchmark runners must NEVER hardcode hyperparameters, seeds, dataset sizes, dimensionalities, or optimization parameters directly in Python code.
- **Config-Driven Architecture**: Any magic numbers or configurable variables must be extracted and defined in a corresponding configuration file (e.g., YAML), and parsed dynamically in the script. Ensure all experiments are fully reproducible from the configuration file alone.

### Proactive Test Coverage Guardrail

Whenever you author new code, add new logic, or modify existing functionality, you MUST proactively verify if the changes are covered by existing tests. 
- If the new code is uncovered, you MUST proactively write new unit tests (or update existing ones) alongside the implementation.
- You must not wait for the user to ask for tests. Consider writing tests an inseparable part of writing the code itself.
- Ensure that the tests verify both positive (happy path) and negative (defensive guardrails/error handling) boundaries.

### Python Style Guidelines (PEP 8 & Google Python Style Guide)

- **Class Structure & Method Ordering**: When authoring or modifying Python classes, strictly follow the method ordering defined by the Google Python Style Guide. 
  - `__init__` (and other dunder methods like `__new__`) must be the very first methods defined in the class, immediately following the class docstring and class-level attributes.
  - `@property` decorators, standard instance methods, and static/class methods must always come *after* the `__init__` method.

### Codified ML Experiment Pipeline (SOP)

All machine learning experiment suites must strictly follow a 6-phase linear pipeline to prevent technical debt and cyclical dependencies. Experiments must be numbered sequentially based on these phases:

1. **Phase 0: Data Grounding & Baselines**: Extract empirical ground truth parameters, establish baseline metrics, and run naive control models (e.g., SSMs, Transformers) *before* introducing novel architecture.
2. **Phase 1: Hyperparameter Optimization**: Sweep and fix architectural dimensions (e.g. using Optuna) before the primary training run.
3. **Phase 2: Primary Model Training**: Train the core architecture using the optimized hyperparameters and grounded data priors.
4. **Phase 3: Scientific Interventions**: Run single-shot tests or specific interventions (e.g., precision injection, knock-outs) on the trained engine.
5. **Phase 4: Parameter Sweeps**: Perform comprehensive dose-response sweeps to gather gradient/trace data across continuous parameter ranges.
6. **Phase 5: Translation & Visualization**: Translate abstract thermodynamic/mathematical metrics back into domain-specific contexts (e.g., EC50) and generate final visualizations for publication.

**Shared Engine Logic**: NEVER import directly from one numbered script to another (e.g., importing `02_train.py` into `03_test.py`). All shared boilerplate (e.g., model instantiation, environment setup) must be extracted into a `core.py` or similar shared module.
