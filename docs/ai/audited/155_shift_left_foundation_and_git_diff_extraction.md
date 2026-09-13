# Prompt 1: Shift-Left Foundation & Git Diff Extraction

We are upgrading our `watchdog_auditor.py` script to a V2 architecture called `nextgen_auditor.py`. We are shifting from a passive, post-commit watchdog daemon to an active, pre-commit Git hook.

Please do the following:
1. Create a new file: `tools/nextgen_auditor.py`.
2. Set up the basic asynchronous scaffold using Python's `asyncio`.
3. Import the new `google-genai` SDK (`from google import genai`). Do NOT use the legacy `google.generativeai` SDK.
4. Create a class `NextGenAuditor`. In its `__init__`, initialize `self.client = genai.Client()` and default `self.model_name = "gemini-2.5-pro"`.
5. Write a method `get_staged_diff(self) -> str:` that uses `subprocess` to run `git diff HEAD~1 HEAD` (or `--cached` if running pre-commit). Return the output as a string. If the command fails or returns empty, catch the exception and return a string stating no staged changes were found.
6. Create an `async def execute_audit(self):` method that currently just calls `get_staged_diff()` and prints the length of the diff.
7. Set up the `if __name__ == "__main__":` block to run `execute_audit()` via `asyncio.run()`.
