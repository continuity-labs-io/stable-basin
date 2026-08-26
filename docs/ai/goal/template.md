The Autonomous Goal Template
[Context Assembly]
(Before typing the prompt, use the IDE's UI to @mention ONLY the files involved)

@Target_Files (Files that need to be modified)

@Reference_Files (Files needed for context, types, or interfaces, but not modified)

[The Prompt]

OBJECTIVE:
[One clear, declarative sentence defining the exact final state of the codebase. e.g., "Migrate architecture X to Y" or "Implement Z feature."]

EXECUTION STEPS:

In [File A], [Specific, mechanical action needed]

In [File B], [Specific, mechanical action needed]

[Any necessary wiring/routing/type mapping between the two]

INVARIANTS (CRITICAL: Do NOT Modify):

[Specific function, class, or business logic that must remain strictly untouched]

[Specific design pattern, UI styling, or library that the AI might be tempted to "clean up" but must leave alone]

EXIT CONDITION:

Run: [Insert exact CLI command, e.g., pytest path/to/test.py, npm run build, or tsc --noEmit]

If the command fails, autonomously read the terminal output, self-correct the code, and rerun.

You are only done when the command succeeds with zero errors.
