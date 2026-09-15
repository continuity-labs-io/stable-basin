# Fix 4: Enforce Codebase Hygiene and Vectorize Mamba

**Severity:** Critical

## Description
Active code is importing from src/icebox/models/ssm/masr_mamba.py, violating strict codebase hygiene. Additionally, this module contains an O(N) Python loop that destroys State-Space Model performance.

## AI Execution Plan
Develop a plan to remove all dependencies on src/icebox/, migrate necessary Mamba logic to src/models/, and vectorize the O(N) loop using parallel associative scans. Get it reviewed, implement, ensure tests pass, and commit.
