# Ship It

Full workflow: verify → commit → push → PR

## Prerequisites

Only run this when you believe work is complete.

## Steps

1. **Verify first**
```bash
pytest tests/unit/ -v
ruff check src/
```
If either fails, STOP. Fix first.

2. **Check diff makes sense**
```bash
git diff
```
Review: does this match what was requested? Any accidental changes?

3. **Commit**
```bash
git add -A && git commit -m "$(cat <<'EOF'
Your message here.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

4. **Push**
```bash
git push -u origin $(git branch --show-current)
```

5. **Create PR**
```bash
gh pr create --title "Title" --body "## Summary
- What changed

## Test plan
- How to verify

---
Generated with Claude Code"
```

6. **Return PR URL**
