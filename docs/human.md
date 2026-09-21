# Human-AI Coding Protocol

**Objective:** Abstract complex architecture out of working memory and execute a
deterministic, pre-computed AI assembly line to eliminate cognitive fatigue and
LLM context-collapse.

**Role:** Act as an Executive QA Inspector handing a deterministic Execution
Ticket to a compiler.

---

## Part 1: The Artifact (The Execution Ticket Template)

Whenever a new feature or module is required, never "brainstorm" with the AI on
the fly. Instead, draft an Execution Ticket using this strict 4-part schema.

### 1. [CONTEXT FILES]: The Sandbox

Explicitly define which files the AI is allowed to read and which it must
create.

- _Purpose:_ Severely limits the AI's "attention mechanism." Prevents it from
  scanning the codebase, hallucinating logic, and mutating unrelated files.

### 2. [TASK]: The Core Mission

Provide a 1-2 sentence definition of the module's sole programmatic
responsibility.

- _Purpose:_ Prevents scope creep. The AI focuses 100% of its parameters on
  solving one isolated objective (e.g., "build the pure functional Optax loop").

### 3. [CORE OBJECTIVES]: Deterministic Translation

Translate the high-level theoretical physics into explicit programmatic
instructions (e.g., "Compute `jax.hessian` and return its trace").

- _Purpose:_ Provides the "What" and the "How" without requiring the AI to grasp
  the biological "Why." Removes the AI's ability to guess or invent its own
  math.

### 4. [CONSTRAINTS]: The Guardrails

Hardcode system invariants (e.g., "PyTorch is strictly for data, JAX is strictly
for compute," "Use DLPack," "Peaceful logging only").

- _Purpose:_ Enforces the **Framework Firewall**. Prevents the AI from
  accidentally leaking PyTorch tensors into JAX compiled functions or utilizing
  inappropriate external libraries.

---

## Part 2: The Execution Protocol (The Mechanical Loop)

When executing tickets during the week, do not try to hold the architecture in
your head. Follow this rigid mechanical loop:

### Step 1: Isolate (The Context Flush)

- Open the AI coding agent.
- **Strict Rule:** You MUST start a brand new, empty chat window. This flushes
  the AI’s context window. Never feed the agent multiple tickets sequentially in
  the same thread.

### Step 2: Inject

- Inject the only next numbered markdown ticket into the chat window and press
  enter.

### Step 3: The "Firewall Review" (QA Inspection)

- The agent reads the ticket and generates the code.
- **Your Job:** You are not here to read every line of complex matrix math. Your
  sole job is to verify the **[CONSTRAINTS]**.
- _Checklist:_ Did it import `torch` inside a pure JAX/Equinox file? Did it use
  dramatic `print()` statements instead of peaceful `logger.info()`? Did it
  attempt to rewrite unrelated files?
- If constraints are broken: Reject the code and instruct the AI to fix the
  specific violation. If constraints are met: Accept the code.

### Step 4: The CI Gate (Local Testing)

Run a local test to guarantee structural integrity before committing. The
testing strategy depends on the module being built:

- **For Data Infrastructure:** Rely on the **Deterministic Synthetic Fallback**
  (e.g., seeded AR processes or 6D limit cycles). Ensure the file executes
  offline and yields correct tensor shapes `[batch, seq, features]` without
  triggering massive dataset downloads.
- **For Core Math / Estimators:** Run the isolated `pytest` suite to verify the
  logic against synthetic mathematical ground-truths (e.g., closed-form MOU
  simulations).
- **For Compute Engines (JAX/Equinox):** Run a "Smoke Test"—execute a 1-step
  forward/backward pass utilizing small, random dummy JAX arrays (e.g.,
  `jax.random.normal`) to guarantee the XLA compilation (`@eqx.filter_jit`)
  succeeds, shapes align, and memory doesn't explode.

### Step 5: Commit & Clear

- Once the CI gate passes, commit the code to Git.
- **Critical:** Close the AI chat window completely to flush its memory.
- Move to the next numbered ticket and repeat from Step 1.
