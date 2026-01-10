# CLAUDE.md

## Before Anything Else

Read STATE.md. Then read this file. If you skip this, you will duplicate work or break things.

---

## Three Rules (Non-Negotiable)

### 1. Push Back First

Before implementing: What's wrong with this? What will break? Is there a simpler way? Should we do this at all?

Say it out loud. Respectful pushback > silent compliance.

### 2. Test Before Done

No fix is complete without a test that proves it works. Run it. See it pass.

```bash
pytest tests/unit/        # fast, isolated
pytest tests/integration/ # test mode
```

Tests validate requirements, not rubber-stamp code. Ask: "Would this catch a regression?"

### 3. Write As You Go

Update immediately, not later:
- **STATE.md** — current work, blockers, decisions
- **LOG.md** — what happened (append-only)

---

## Slash Commands

| Command | What it does |
|---------|--------------|
| `/test` | Run pytest |
| `/done` | Verify before marking complete |
| `/review` | Subagent reviews changes (fresh eyes) |
| `/ship` | verify → commit → push → PR |
| `/save` | Save context to STATE.md + LOG.md |
| `/sandbox` | Test pipeline without Tavily costs |

**Flow:** `work → /test → /done → /review → /ship`

---

## Architecture

Read ARCHITECTURE.md and INVARIANTS.md before structural changes.

Update ARCHITECTURE.md when adding modules, changing data flow, or modifying graph/state.

Never weaken INVARIANTS.md.

---

## Context Preservation

Long conversation = context risk. Before it's lost:
1. Update STATE.md
2. Append to LOG.md
3. Continue

Or just run `/checkpoint`.

---

## Don't

- Don't say "great idea" then implement — question first
- Don't mark done without running tests
- Don't write weak tests that just pass
- Don't batch STATE.md/LOG.md updates
- Don't add features beyond what was asked
- Don't refactor unrequested code
- Don't over-engineer for hypothetical futures
- Don't create files unless necessary
- Don't guess — investigate first
