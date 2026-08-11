# Logging Standards

This document establishes the rule for how we handle output and logging in the
Stable Basin Benchmark codebase.

## The Rule: No `print()` Statements in Production Code

- **Core Codebase**: The use of standard `print()` statements is **strictly
  prohibited** in all production code (e.g., inside `src/models/`,
  `src/metrics/`, `src/pipeline/`, `src/utils/`, etc.).
- **Production Logging**: All outputs must use a standard production-level
  logger (e.g., Python's built-in `logging` module). This ensures that logs can
  be properly routed, formatted, timestamped, and filtered by severity level
  (DEBUG, INFO, WARNING, ERROR, CRITICAL).
- **Exceptions**: `print()` statements are only acceptable within the
  `src/demo/` directory, where scripts are meant to be run interactively in the
  terminal by a user to demonstrate functionality.

## Best Practices

When writing or modifying core modules:

1. Always initialize a logger at the top of the file:
   ```python
   import logging

   logger = logging.getLogger(__name__)
   ```
2. Use appropriate log levels rather than printing everything:
   - `logger.debug()` for granular information useful only when troubleshooting.
   - `logger.info()` for standard operational milestones.
   - `logger.warning()` for unexpected but recoverable edge cases.
   - `logger.error()` or `logger.exception()` for failures.
