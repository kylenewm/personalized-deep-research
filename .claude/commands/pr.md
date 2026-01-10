# Create Pull Request

Create a PR for the current branch.

## Steps

1. Check current state:
```bash
git status
git log main..HEAD --oneline
git diff main...HEAD --stat
```

2. Push if needed:
```bash
git push -u origin $(git branch --show-current)
```

3. Create PR:
```bash
gh pr create --title "Title here" --body "$(cat <<'EOF'
## Summary
- Bullet points of what changed

## Test plan
- [ ] How to verify this works

---
Generated with [Claude Code](https://claude.ai/code)
EOF
)"
```

4. Return the PR URL
