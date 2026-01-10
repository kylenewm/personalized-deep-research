# Claude Code Workflow Guide

## Quick Start

1. Open terminal in project directory
2. Run `claude`
3. Type `/test` to verify commands work

---

## Daily Workflow

### Starting a Session

1. Claude should automatically read STATE.md (if not, it's not following rules)
2. Tell Claude what you want to do
3. If it doesn't push back with questions, remind it: "push back first"

### While Working

```
/test          # run tests frequently
/save          # if conversation getting long
```

### Finishing Work

```
/test          # make sure tests pass
/done          # full verification (tests + lint + requirements)
/review        # spawn fresh agent to catch blind spots
/ship          # commit + push + PR (one command)
```

---

## Commands Reference

| Command | When to Use |
|---------|-------------|
| `/test` | After any code change |
| `/done` | Before saying "this is complete" |
| `/review` | Before shipping, catches issues you missed |
| `/ship` | Ready to commit + PR |
| `/save` | Long conversation, switching tasks |
| `/sandbox` | Testing pipeline changes without API costs |

---

## Multi-Terminal Setup (5 Parallel Claudes)

### Directory Structure

```
~/Downloads/
├── deep-research-v0/          # base (shared state files)
├── deep-research-v0-t1/       # terminal 1
├── deep-research-v0-t2/       # terminal 2
├── deep-research-v0-t3/       # terminal 3
├── deep-research-v0-t4/       # terminal 4
└── deep-research-v0-t5/       # terminal 5
```

### How It Works

- Each terminal has its own code copy (no conflicts)
- STATE.md, LOG.md, CLAUDE.md are symlinked (shared across all)
- Work in isolation, merge when done

### Starting Parallel Sessions

```bash
# Terminal 1
cd ~/Downloads/deep-research-v0-t1 && claude

# Terminal 2
cd ~/Downloads/deep-research-v0-t2 && claude

# etc...
```

### Merging Work Back

When a terminal finishes:
```bash
cd ~/Downloads/deep-research-v0-t1
./scripts/merge-to-base.sh
```

---

## Notifications

System notifications fire when Claude needs input. You'll hear/see an alert.

---

## If Things Go Wrong

- **Claude not following rules**: Start fresh session (`esc esc`, then `claude`)
- **Tests failing**: Don't mark done, fix first
- **Merge conflicts**: Resolve manually, then continue
- **Session going in circles**: Abandon it (10-20% abandonment is normal)

---

## Files That Matter

| File | What | Check When |
|------|------|------------|
| STATE.md | Current work | Start of session |
| LOG.md | History | Want to know what happened |
| CLAUDE.md | Rules | Claude not behaving |
| .claude/commands/ | Slash commands | Commands not working |

---

## Testing This Setup

1. Open new terminal
2. `cd ~/Downloads/deep-research-v0 && claude`
3. Type `/test`
4. Should run pytest without permission prompts

If it works, you're ready.
