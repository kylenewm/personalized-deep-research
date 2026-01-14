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

## Running the Pipeline (IMPORTANT)

### Load Environment Variables

**ALWAYS load .env before running the pipeline.** API keys are stored there.

```python
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path('.env'))
```

### Correct Invocation Format

**CRITICAL: Use `messages` format, NOT `research_topic`.**

```python
from langchain_core.messages import HumanMessage

# CORRECT - pipeline will work
result = await deep_researcher.ainvoke({
    'messages': [HumanMessage(content='Your query here')]
}, config=config)

# WRONG - supervisor gets empty context, no research happens
result = await deep_researcher.ainvoke({
    'research_topic': 'Your query here'  # DON'T DO THIS
}, config=config)
```

### Bypass Review Mode

The pipeline has a brief review mode that will INTERRUPT execution waiting for human input. **Always bypass it:**

```python
config = {
    'configurable': {
        'review_mode': 'none',       # Skip human review checkpoints
        'allow_clarification': False, # No clarification questions
    }
}
```

Without this, the pipeline stops after brief generation with 0 sources.

### Always Ask About Saving Data

**Before running any report/pipeline**, ask the user:

> "Do you want me to save the extractions and source data for downstream work (re-rendering, iteration)?"

If yes, save to `tests/fixtures/gold_queries/{name}.json`:
```python
save_data = {
    'query': query,
    'research_brief': result.get('research_brief', ''),
    'sources': result.get('source_store', []),
    'hybrid_report': result.get('hybrid_report', {}),
    'captured_at': datetime.now().isoformat(),
}
```

This allows re-rendering without re-running the expensive research step.

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

## Subagents

| Agent | Purpose | When to Use |
|-------|---------|-------------|
| `code-architect` | Design before implementing | New pipeline nodes, architectural changes |
| `verify-app` | Test implementation works | After implementing, before declaring done |
| `code-simplifier` | Reduce complexity | After feature complete, code feels bloated |
| `build-validator` | Check deployment readiness | Before releases |
| `oncall-guide` | Debug issues | When investigating pipeline failures |

**How to invoke:** Ask Claude to "use code-architect to design this" or "spawn verify-app to test"

---

## Workflow (Boris-Style)

For non-trivial tasks:

```
1. Think        → Plan mode or code-architect (design first)
2. Implement    → Write the code
3. Verify       → verify-app OR /test (prove it works)
4. Simplify     → Optional: code-simplifier if complex
5. Review       → /review (fresh subagent eyes)
6. Ship         → /ship (test → commit → push → PR)
```

**This project's workflow emphasis:**
- ALWAYS verify with tests before marking done (Quality Gates above)
- ALWAYS check INVARIANTS.md before structural changes
- Use /sandbox for pipeline testing without API costs

**Shortcuts:**
- Bug fix: implement → /test → /done → /commit
- Pipeline change: code-architect → implement → /test → verify-app → /ship

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
