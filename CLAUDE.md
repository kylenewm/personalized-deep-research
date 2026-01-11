# Project: Deep Research

## Before Anything Else

Read STATE.md. Then read this file.

---

## Quality Gates (Non-Negotiable)

**Optimize for PRECISION, not completion speed.**

1. **Unit test each fix** — Before marking any issue as fixed, write or run a unit test that proves the fix works
2. **Integration test after sections** — After completing a related group of fixes, run integration tests
3. **No marking done without verification** — If you can't prove it works, it's not done

Never rush. Never bullshit.

**Workflow:** Keep working through fixes sequentially. Do not stop to ask "want to continue?" — just continue unless interrupted.

---

## Anti-Patterns (This Project)

| Pattern | Why It Fails |
|---------|--------------|
| Locked verbatim quotes | Unreadable |
| Free generation + post-verify | Already hallucinated |
| Multi-layer verification | Bugs compound |
| Infra before problem | Wasted effort |

---

## Before Any Change (Invariant Check)

1. Read INVARIANTS.md — does this change violate any?
2. Core principle: **LLM points, code extracts verbatim, LLM cannot write fact text**
3. If unsure, ask: "Is LLM writing content that goes in the report?"

**Valid:**
- LLM points to content (keywords)
- LLM rejects/filters facts
- LLM selects best among duplicates
- Code cleans formatting (strips `**`, `####`)

**Invalid (violates I1):**
- LLM writes fact text
- LLM rewrites/paraphrases facts
- LLM "cleans up" fact wording

---

## Testing

```bash
./venv/bin/pytest tests/unit/ -v        # fast
./venv/bin/pytest tests/integration/ -v  # slower
```

**Testing Philosophy:**

| Tier | Default approach | Flexibility |
|------|------------------|-------------|
| Unit | Mock data, no pipeline | Can use test mode if logic requires real data |
| Integration | TEST MODE (minimal searches, tiny scope) | Can expand if specific test needs it |
| E2E | Full pipeline | Use sparingly |

**Test Quality (Critical):**
- Do NOT write easy tests just to pass — tests validate requirements, not rubber-stamp code
- Do NOT overfit tests to current implementation — test the actual contract/behavior
- If a test reveals the code is wrong, FIX THE CODE not the test
- Ask: "Would this test catch a regression?" If not, it's too weak

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
