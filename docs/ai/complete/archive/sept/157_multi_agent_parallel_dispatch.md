# Prompt 3: Multi-Agent Parallel Dispatch (The Experts)

We will now implement the parallel "Panel of Experts" using `asyncio.gather` to prevent persona bleed.

Please update `tools/nextgen_auditor.py`:
1. Create an async helper method: `async def _run_expert_agent(self, persona_prompt: str, diff: str, repo_context: str, test_logs: str) -> str:`. It should construct a prompt combining these inputs and call `await self.client.aio.models.generate_content(...)` using `self.model_name`. Configure it with `google.genai.types.GenerateContentConfig(temperature=0.1)`. Return the `.text` of the response.
2. Inside `execute_audit`, define two strict persona prompts:
   - **The Mathematical Physicist**: Instruct it to verify continuous-time Euler-Maruyama discretization (e.g., ZOH limits in `masr_mamba.py`), and ensure thermodynamic invariants (e.g. Q skew-symmetry in `solenoidal.py`, Gamma positive-definiteness via Cholesky in `dissipative.py`).
   - **The Systems Architect**: Instruct it to review for PyTorch/JAX VRAM bottlenecks, O(1) SSM constraints, Triton kernels block sizing, and `jaxtyping` validation. Explicitly forbid `torch` imports inside `src/echo/` and `jax` imports inside `src/models/`.
3. Use `asyncio.gather` to dispatch both the Physicist and the Architect simultaneously using the `_run_expert_agent` method. Await their results and store them in `physicist_report` and `architect_report`.
