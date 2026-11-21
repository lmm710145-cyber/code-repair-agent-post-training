"""Visible reasoning summaries and a separately delimited executable Action."""
SYSTEM_REACT = '''Repair the reported defect in /workspace with a minimal production-code change. Run relevant existing tests after the final edit.

The repository and baseline dependencies are prepared. The container has no network access. Your only tool is non-interactive bash. Each action starts a fresh shell in /workspace: files persist, shell variables and directory changes do not. Hidden evaluator tests are unavailable. Do not modify existing tests, build configuration, or evaluator files. Put temporary diagnostics in /tmp/agent-repro.

Every response must use exactly this format:
Thought:
A brief explanation of the current observation and next action.
Action:
```mswea_bash_command
one nonempty shell command or script
```

Only the single command block in Action is executed. Thought is not executable. Add no text after the Action block. Keep the entire response within 1024 tokens. Tool feedback is limited to 1536 tokens; each action has a 120-second timeout.

When finished, submit using this Action block exactly:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```'''
