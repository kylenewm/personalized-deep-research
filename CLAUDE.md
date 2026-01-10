# Project: Deep Research

## Before Anything Else

Read STATE.md. Then read this file.

---

## Testing

```bash
./venv/bin/pytest tests/unit/ -v        # fast
./venv/bin/pytest tests/integration/ -v  # slower
```

---

## Slash Commands

| Command | What |
|---------|------|
| `/test` | Run pytest |
| `/test-cycle` | Generate + run progressively |
| `/done` | Verify before complete |
| `/review` | Subagent review |
| `/ship` | verify → commit → PR |
| `/save` | Update STATE.md + LOG.md |
| `/summarize` | AI explain changes |
| `/sandbox` | Test pipeline (no API costs) |

**Flow:** `work → /test → /done → /review → /ship`

---

## Files

| File | Purpose |
|------|---------|
| STATE.md | Current work, decisions |
| LOG.md | History (append-only) |
| ARCHITECTURE.md | System design |
| INVARIANTS.md | Contracts (never weaken) |
| WORKFLOW.md | How to use this setup |

---

## Architecture

Read ARCHITECTURE.md before structural changes.

Update it when: adding modules, changing data flow, modifying graph/state.

---

## Context Preservation

Long conversation → run `/save`
