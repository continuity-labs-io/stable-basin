import os
import asyncio
import subprocess
from google import genai

class NextGenAuditor:
    def __init__(self):
        # genai.Client() automatically checks the GEMINI_API_KEY environment variable, 
        # but we can explicitly pass it if needed.
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-pro")
        
    def get_staged_diff(self) -> str:
        """
        Runs git diff to get staged changes (or latest commit if no staged changes).
        Returns the output as a string, or a fallback message if no changes.
        """
        try:
            # Try --cached for pre-commit hooks
            result = subprocess.run(
                ["git", "diff", "--cached"],
                capture_output=True,
                text=True,
                check=True
            )
            diff_output = result.stdout.strip()
            
            if not diff_output:
                # Fallback to HEAD~1 HEAD if there are no staged changes, for testing purposes
                result = subprocess.run(
                    ["git", "diff", "HEAD~1", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                diff_output = result.stdout.strip()
                
            if not diff_output:
                return "No staged changes found."
                
            return diff_output
        except Exception as e:
            # Catching exception and returning fallback string
            return f"No staged changes found. (Error: {e})"

    async def execute_audit(self):
        diff = self.get_staged_diff()
        print(f"Length of staged diff: {len(diff)}")

if __name__ == "__main__":
    auditor = NextGenAuditor()
    asyncio.run(auditor.execute_audit())
