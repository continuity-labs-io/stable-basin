# Prompt 2: Execution-in-the-Loop & Focused Context

Now we need to give the AI real runtime context and focused repository context,
rather than dumping the entire repo into the prompt.

Please update `tools/nextgen_auditor.py`:
1. Add `run_preflight_tools(self) -> str:` 
   - Use `subprocess.run` to execute `make preflight` (which runs torchfix and
     pytest).
   - Set a timeout of 120 seconds. Capture both `stdout` and `stderr`.
   - Return a formatted string combining both outputs. If it times out or fails,
     gracefully return the error message so the LLM can see what broke.
2. Add `get_modified_files(self) -> list[str]:`
   - Use `subprocess.check_output` to run `git diff --name-only HEAD~1 HEAD` to
     get the list of modified files. Return them as a list of strings.
3. Add `generate_focused_context(self, modified_files: list[str]) -> str:`
   - Run the `repomix` CLI command via `subprocess`. Use the `--include` flag to
     only package the specific modified files (e.g., `repomix --include
     file1,file2 --output cache/focused_context.xml`). If `modified_files` is
     empty, just run normal `repomix`.
   - Read the generated `cache/focused_context.xml` into a string, delete the
     temp file, and return the string.
4. Update `execute_audit(self)` to call these new methods and store their
   results in variables (`test_logs` and `repo_context`).
