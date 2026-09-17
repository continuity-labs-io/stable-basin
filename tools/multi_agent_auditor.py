import os
import sys
import json
import asyncio
import subprocess
from google import genai
from google.genai import types

from personas import PHYSICIST_PROMPT, ARCHITECT_PROMPT, STRATEGIC_LEAD_PROMPT

class MultiAgentAuditor:
    def __init__(self, mode="repo", n_commits=1):
        # genai.Client() automatically checks the GEMINI_API_KEY environment
        # variable, but we can explicitly pass it if needed.
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
        self.mode = mode
        self.n_commits = n_commits
        
    def get_diff(self) -> str:
        """
        Returns the appropriate git diff based on the selected mode.
        """
        if self.mode == "repo":
            return "N/A - Auditing entire repository."
            
        try:
            cmd = ["git", "diff"]
            if self.mode == "staged":
                cmd.append("--cached")
            elif self.mode == "commits":
                cmd.extend([f"HEAD~{self.n_commits}", "HEAD"])
                
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            diff_output = result.stdout.strip()
            
            if not diff_output:
                return f"No changes found for mode: {self.mode}."
                
            return diff_output
        except Exception as e:
            return f"Failed to get diff for mode {self.mode}. (Error: {e})"

    def run_preflight_tools(self) -> str:
        try:
            result = subprocess.run(
                ["make", "preflight"],
                capture_output=True,
                text=True,
                timeout=120
            )
            return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except subprocess.TimeoutExpired as e:
            # e.stdout and e.stderr are available on TimeoutExpired
            stdout = e.stdout.decode('utf-8') if isinstance(e.stdout, bytes) else e.stdout
            stderr = e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else e.stderr
            return f"Preflight timed out after 120s.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        except Exception as e:
            return f"Preflight failed: {str(e)}"

    def get_modified_files(self) -> list[str]:
        if self.mode == "repo":
            return []
            
        try:
            cmd = ["git", "diff", "--name-only"]
            if self.mode == "staged":
                cmd.append("--cached")
            elif self.mode == "commits":
                cmd.extend([f"HEAD~{self.n_commits}", "HEAD"])
                
            output = subprocess.check_output(cmd, text=True)
            return [line.strip() for line in output.strip().split("\n") if line.strip()]
        except Exception as e:
            print(f"Warning: Failed to get modified files: {e}")
            return []

    def generate_focused_context(self, modified_files: list[str]) -> str:
        os.makedirs("cache", exist_ok=True)
        output_file = "cache/focused_context.xml"
        cmd = ["repomix", "--output", output_file]
        
        if modified_files:
            cmd.extend(["--include", ",".join(modified_files)])
            
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    context = f.read()
                os.remove(output_file)
                return context
            else:
                return "Failed to generate context: Output file not found."
        except subprocess.CalledProcessError as e:
            return f"Failed to generate focused context. Return code: {e.returncode}\nSTDERR: {e.stderr}"
        except Exception as e:
            return f"Failed to generate focused context: {str(e)}"

    async def _run_expert_agent(self, persona_prompt: str, diff: str, repo_context: str, test_logs: str) -> str:
        prompt = f"{persona_prompt}\n\n### REPOSITORY CONTEXT ###\n{repo_context}\n\n### TEST LOGS ###\n{test_logs}\n\n### STAGED DIFF ###\n{diff}\n"
        
        chat = self.client.aio.chats.create(
            model=self.model_name,
            config=genai.types.GenerateContentConfig(temperature=0.1)
        )
        response = await chat.send_message(prompt)
        return response.text

    async def execute_audit(self):
        diff = self.get_diff()
        print(f"Length of diff: {len(diff)}")
        
        print("Running preflight tools...")
        test_logs = self.run_preflight_tools()
        
        modified_files = self.get_modified_files()
        print(f"Modified files: {len(modified_files)} files found.")
        
        print("Generating focused context...")
        repo_context = self.generate_focused_context(modified_files)
        print(f"Length of test logs: {len(test_logs)}")
        print(f"Length of repo context: {len(repo_context)}")
        
        print("Dispatching expert agents in parallel... takes ~1 minute")
        physicist_report, architect_report = await asyncio.gather(
            self._run_expert_agent(PHYSICIST_PROMPT, diff, repo_context, test_logs),
            self._run_expert_agent(ARCHITECT_PROMPT, diff, repo_context, test_logs)
        )
        
        print("\n" + "="*40)
        print("THE MATHEMATICAL PHYSICIST REPORT")
        print("="*40)
        print(physicist_report)
        
        print("\n" + "="*40)
        print("THE SYSTEMS ARCHITECT REPORT")
        print("="*40)
        print(architect_report)

        print("\n" + "="*40)
        print("DISPATCHING STRATEGIC LEAD...")
        print("="*40)

        audit_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "physics_review": types.Schema(type=types.Type.STRING),
                "systems_review": types.Schema(type=types.Type.STRING),
                "strategic_verdict": types.Schema(type=types.Type.STRING),
                "roi_approved": types.Schema(type=types.Type.BOOLEAN),
                "critical_bugs_found": types.Schema(type=types.Type.BOOLEAN),
                "actionable_fixes": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "title": types.Schema(type=types.Type.STRING, description="Short, concise title of the fix"),
                            "description": types.Schema(type=types.Type.STRING, description="Detailed explanation of what needs to be fixed"),
                            "severity": types.Schema(type=types.Type.STRING, description="Severity of the issue: Critical, High, Medium, or Low"),
                            "ai_execution_instructions": types.Schema(type=types.Type.STRING, description="Instructions for an AI on how to address this fix, explicitly mandating the workflow: develop a plan, get it reviewed, implement, ensure tests pass, and commit.")
                        },
                        required=["title", "description", "severity", "ai_execution_instructions"]
                    )
                ),
            },
            required=[
                "physics_review",
                "systems_review",
                "strategic_verdict",
                "roi_approved",
                "critical_bugs_found",
                "actionable_fixes"
            ]
        )

        strategic_prompt_full = f"{STRATEGIC_LEAD_PROMPT}\n\n### PHYSICIST REPORT ###\n{physicist_report}\n\n### ARCHITECT REPORT ###\n{architect_report}\n\n### TEST LOGS ###\n{test_logs}\n\n### STAGED DIFF ###\n{diff}\n"

        chat = self.client.aio.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=audit_schema,
            )
        )
        response = await chat.send_message(strategic_prompt_full)
        
        try:
            audit_result = json.loads(response.text)
        except json.JSONDecodeError:
            print("Failed to parse JSON from Strategic Lead.")
            sys.exit(1)
            
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        except Exception:
            commit_hash = "unknown"
            
        md_report = f"# Strategic Audit Report ({commit_hash})\n\n"
        md_report += f"## Physics Review\n{audit_result.get('physics_review')}\n\n"
        md_report += f"## Systems Review\n{audit_result.get('systems_review')}\n\n"
        md_report += f"## Strategic Verdict\n{audit_result.get('strategic_verdict')}\n\n"
        md_report += f"**ROI Approved:** {audit_result.get('roi_approved')}\n"
        md_report += f"**Critical Bugs Found:** {audit_result.get('critical_bugs_found')}\n\n"
        md_report += "## Actionable Fixes (Triage Required)\n\n"
        
        for i, fix in enumerate(audit_result.get('actionable_fixes', [])):
            md_report += f"### Fix {i+1}: {fix.get('title')}\n"
            md_report += f"**Severity:** {fix.get('severity')}\n\n"
            md_report += f"{fix.get('description')}\n\n"
            md_report += f"**AI Execution Plan:**\n{fix.get('ai_execution_instructions')}\n\n"
            md_report += "**Triage Decision (Human Reviewer):**\n"
            md_report += "- [ ] Fix Now (Execute the AI Plan)\n"
            md_report += "- [ ] Backlog\n"
            md_report += "- [ ] Ignore\n\n"
            md_report += "---\n\n"
            
        os.makedirs("logs/audits", exist_ok=True)
        report_path = f"logs/audits/AUDIT_{commit_hash}.md"
        with open(report_path, "w") as f:
            f.write(md_report)
            
        print(f"Saved audit report to {report_path}")
        
        if audit_result.get("critical_bugs_found"):
            print("\n\033[91mCRITICAL BUGS FOUND! COMMIT REJECTED.\033[0m")
            print("Actionable Fixes:")
            for fix in audit_result.get("actionable_fixes", []):
                print(f" - [{fix.get('severity')}] {fix.get('title')}")
            sys.exit(1)
        else:
            print("\n\033[92mAUDIT PASSED. COMMIT APPROVED.\033[0m")
            sys.exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Auditor")
    parser.add_argument("--mode", type=str, choices=["repo", "staged", "commits"], default="repo", help="Audit mode: 'repo' (entire repo), 'staged' (staged changes), or 'commits' (previous N commits).")
    parser.add_argument("--n-commits", type=int, default=1, help="Number of previous commits to audit (only used in 'commits' mode).")
    args = parser.parse_args()

    auditor = MultiAgentAuditor(mode=args.mode, n_commits=args.n_commits)
    asyncio.run(auditor.execute_audit())
