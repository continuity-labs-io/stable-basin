### Staging Prompts Guardrail
- **Read-Only Directory (`docs/ai/accepted/`)**: Do not modify or edit any markdown files within `docs/ai/accepted/`. These are staging prompts intended for human review before being coded. Treat this directory as strictly read-only.

### Output Directory Structure
- **Mirrored Paths**: When scripts generate output files (e.g., plots, CSVs), they should save them in a directory structure within `output/` that mirrors the source script's path. For example, a script running from `src/demo/` should save its outputs to `output/demo/`. Ensure the output directories are created if they do not exist.

### Plan Naming Convention
- **Sequential Numbering**: Any new plan file created in the `docs/ai/` directory (or its subdirectories) must start with a 3-digit zero-padded sequential prefix (e.g., `066_new_plan_name.md`).
- **Discovery**: Before creating a new plan, always search the `docs/ai/` directory (including `completed/`, `accepted/`, etc.) to determine the highest existing prefix number and increment it for your new file.

### Execution Scope Guardrail
- **Single Plan Execution**: Only execute, plan, or focus on exactly 1 prompt, plan, or directive at a time. Do not bundle multiple plans or feature requests into a single implementation plan unless explicitly requested by the user. If the user mentions a specific plan, strictly limit the scope of the work to that single plan.

### The 1-to-1 Invariant Rule.

For every non-standard tensor operation (e.g., continuous discretization, masking, state-routing), there must be exactly one isolated mathematical invariant test. N lines of dense, continuous-time physics require a minimum of 3*N assertions covering boundary conditions, gradient stability, and shape consistency.

### Unit Test Structure

All Unit Tests Must delineate the following three blocks: "ARRANGE", "ACT", and "ASSERT".

- **ARRANGE**: Define all inputs, constants, and expected values.
- **ACT**: Execute the function under test.
- **ASSERT**: Compare the actual output against the expected values.

### Commit Guardrail

Do not create commits unless the user explicitely asks you to do so.
