# STATE.md

## What We're Building

Deep Research Agent — AI-powered research agent that searches the web, gathers sources, and generates verified research reports with anti-hallucination verification.

## Current Priorities

1. [x] All CRITICAL issues resolved (7/7)
2. [x] All HIGH issues resolved (4/4)
3. [x] Citation-First Evaluation implemented (deterministic approach)

**COMPLETE:** 156 unit tests pass. All audit issues addressed. New citation-first metrics added.

## Key Decisions (Collaborative)

| Date | Decision | Why | Alternatives Rejected |
|------|----------|-----|----------------------|
| 2026-01-07 | Use STATE.md as single source of truth | Agents read one file, not many | Multiple scattered files |
| 2026-01-07 | Append-only LOG.md | Preserves history, prevents context loss | Replace (loses history) |
| 2026-01-07 | Skill for complex tasks, CLAUDE.md for always-on | Separation of always-on vs opt-in behavior | Everything in CLAUDE.md |
| 2026-01-08 | Citation-first evaluation (deterministic) | Separates validity from coverage, no LLM extraction issues | RAGAS (doesn't handle numbered citations), LLM extraction (paraphrasing problem) |

## Open Issues

**CRITICAL (from 2026-01-07 audit):**
- [x] Hallucination rate definition wrong (evaluation.py:645) — FIXED + unit tested (8 tests)
- [x] Claim-to-citation matching broken (evaluation.py:195-240) — FIXED + unit tested (11 tests)
- [x] Citation format mismatch (prompts.py vs evaluation.py) — FIXED + unit tested (10 tests)
- [x] Dual source storage divergence (utils.py + researcher.py) — ANALYZED: not a bug, documented architecture
- [x] Silent truncation at 3+ points — FIXED + unit tested (14 tests)
- [x] LLM in "deterministic" claim gate — ANALYZED: not an I2 violation (I2 covers quote verification only)
- [x] Tokenization breaks acronyms (verify.py:24-35) — FIXED + unit tested (4 tests)

**HIGH:**
- [x] Extract node may miss multi-paragraph quotes — FIXED: increased max_words 60→100 for consistency (3 tests)
- [x] Round-robin diversity (extract.py:135) — ANALYZED: not a bug, IS enforced correctly (3 tests)
- [x] Jaccard window off-by-one (verify.py:93) — ANALYZED: not a bug, added 4 edge case tests
- [x] Vague claims auto-pass (claim_gate.py:86) — FIXED: track skipped claims + improved regex (20 tests)

**FOUND IN RE-AUDIT:**
- [x] document_processing.py:286 still used max_words=60 — FIXED
- [x] verify.py:171 returns {} when disabled (snippets stay PENDING) — FIXED: now marks as SKIP (2 tests)

See LOG.md for full 41-issue audit.

## Current Test Strategy

- Unit: pytest for individual nodes
- Integration: scripts/test_e2e_quick.py for full pipeline
- Manual: Run with test queries, verify citations

## New Metrics (Citation-First Evaluation)

Added deterministic metrics that don't rely on LLM claim extraction:

| Metric | What it Measures | Formula |
|--------|------------------|---------|
| **Citation Validity** | Are our citations correct? | `valid_citations / (valid + invalid)` |
| **Citation Coverage** | What % of sentences have citations? | `cited_sentences / total_sentences` |

These metrics are DETERMINISTIC (same input → same output) and separate validity from coverage.

## Blockers

None.

## Last Updated

2026-01-08 — Citation-first evaluation implemented. 156 unit tests pass.
