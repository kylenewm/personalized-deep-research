# Code Review (Subagent)

Spawn a fresh agent to review work with no context bias.

## What This Does

Launches a separate agent to review the current changes. Fresh eyes catch issues the working agent missed.

## Steps

1. Get the diff:
```bash
git diff HEAD
```

2. Spawn review subagent using Task tool:
   - subagent_type: "Explore"
   - prompt: |
     Review this diff for:
     1. Bugs or logic errors
     2. Missing edge cases
     3. Code that doesn't match the stated intent
     4. Security issues
     5. Unnecessary complexity

     Be critical. List specific issues with file:line references.

     Diff:
     [paste diff here]

3. Report findings back to user

## When to Use

- Before `/ship`
- After complex changes
- When unsure if solution is correct
