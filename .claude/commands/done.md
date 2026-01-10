# Mark Work Done

Before marking any work complete, verify it actually works.

## Steps

1. **Run tests**
```bash
pytest tests/unit/ -v
```

2. **Run linter**
```bash
ruff check src/
```

3. **Re-read the original request** - scroll up, find what user asked for

4. **Check each requirement is met** - don't assume, verify

5. **Only if ALL pass:** mark the task complete

## If anything fails

- Fix it first
- Re-run verification
- Do NOT mark done until it passes

## Spawn verification subagent

Use Task tool to spawn a verification agent:
- subagent_type: "Explore"
- prompt: "Review the changes made in this session. Check: 1) Do they meet the original request? 2) Are there obvious bugs? 3) Are tests passing? Report issues found."
