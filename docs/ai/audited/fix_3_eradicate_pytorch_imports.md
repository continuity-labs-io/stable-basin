# Fix 3: Eradicate PyTorch Imports from src/echo/

**Severity:** Critical

## Description
The src/echo/ directory must remain a pure JAX/Equinox environment. PyTorch imports in benchmark and toy data scripts violate framework boundaries and cause cross-framework memory fragmentation.

## AI Execution Plan
Develop a plan to remove all PyTorch imports from src/echo/ by refactoring data loaders or moving scripts to appropriate directories. Get it reviewed, implement, ensure tests pass, and commit.
