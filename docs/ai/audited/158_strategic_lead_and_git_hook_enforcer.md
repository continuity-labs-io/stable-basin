# Prompt 4: The Strategic Lead & Git Hook Enforcer (Structured Outputs)

Finally, we need the "Strategic Lead" coordinator agent to synthesize the expert reports using Gemini's Structured Outputs, and physically block the commit if critical bugs are found.

Please update `tools/nextgen_auditor.py`:
1. Import `from google.genai import types`.
2. Define a `types.Schema` for the final audit. It must be a `Type.OBJECT` with properties:
   - `physics_review` (Type.STRING)
   - `systems_review` (Type.STRING)
   - `strategic_verdict` (Type.STRING)
   - `roi_approved` (Type.BOOLEAN)
   - `critical_bugs_found` (Type.BOOLEAN)
   - `actionable_fixes` (Type.ARRAY of Type.STRING)
   Make sure to include all of these in the `required` list.
3. Create a prompt for the Strategic Lead that takes the `physicist_report`, the `architect_report`, and the `test_logs`. Tell it to synthesize them, apply the 10^15 ROI framework, check for multi-rate polling NaN contagion, and enforce that active code never imports from `src/icebox/`.
4. Call `await self.client.aio.models.generate_content` using `response_mime_type="application/json"` and passing the schema to `response_schema` in the GenerateContentConfig.
5. Parse the returned JSON text into a Python dictionary.
6. Format a beautiful Markdown report from the JSON and save it to `logs/audits/AUDIT_<commit_hash>.md`.
7. **THE GATEKEEPER LOGIC**: Check the `critical_bugs_found` boolean from the JSON. 
   - If `True`, print a loud ANSI red warning to the console, list the `actionable_fixes`, and call `sys.exit(1)` to abort the git commit. 
   - If `False`, print a green success message and call `sys.exit(0)`.
8. Write a bash script `scripts/setup_git_hooks.sh` that copies or symlinks `tools/nextgen_auditor.py` execution into `.git/hooks/pre-commit` and makes it executable.

### Testing

Either you or the human should execute this testing:

Smoke test: run `bash scripts/setup_git_hooks.sh`

Sanity check: Intentionally introduce a bug (e.g., add import torch to src/echo/physics/dissipative.py or break the positive-definite math) and try to run `git commit -m "test"`.

Verify that the AI Architect/Physicist will detect it, the Coordinator will flag `critical_bugs_found: true`, and your terminal rejects the commit.
