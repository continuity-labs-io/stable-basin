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

ALL new PyTorch neural network modules must use `jaxtyping` (e.g.,
`Float[Tensor, "batch seq d_model"]`) in their method signatures. Additionally,
strictly use `einops` for complex reshapes/rearranges instead of native
`.view()` or `.reshape()`.

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

### Documentation Guardrail

- **Method/Function Documentation**: All public methods and functions must have their arguments (`Args:`) and return values (`Returns:`) explicitly documented in their docstrings.

### Standardized Commit Messages (Google Standard)

All commit messages must strictly adhere to the following structure:
1. **The "What"**: The first line (subject) must summarize exactly *what* the change is doing (e.g., `feat(data): add PyTorch dataloader for C. elegans dataset`).
2. **The "Why"**: The first paragraph of the body must explain *why* this change is being made, providing the rationale and context behind the implementation.
3. **The "Details"**: Following the "why", use bullet points to detail the major technical changes or specific files affected.
