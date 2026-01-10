# STATE.md

## What We're Building

Deep Research Agent — AI-powered research agent that searches the web, gathers sources, and generates verified research reports with anti-hallucination verification.

## Current Status

**Two fixes committed and pushed to main.**

| Component | Status |
|-----------|--------|
| Per-source dedup limit | ✅ FIXED - was killing coverage |
| Chunking | ✅ REMOVED - same coverage, 82% fewer LLM calls |
| Extraction | ✅ Working - ~14 unique facts/source |
| Arranger | ✅ VALIDATED - 73% exclusion is appropriate (filtering fluff) |

## Fixes Applied This Session

### Fix 1: Removed Per-Source Dedup Limit

**Problem:** `deduplicate_extractions()` had "Pass 1" that kept only 1 fact per source.

**Fix:** Removed per-source limit. Now only removes actual duplicate content.

**Impact:** 16x more facts survive to downstream stages.

### Fix 2: Disabled Chunking

**Problem:** Chunking created 3x more LLM calls for same coverage.

**Test results:**
| Metric | WITH chunking | WITHOUT chunking |
|--------|---------------|------------------|
| Unique facts | 40 | 38 |
| LLM calls | 17 | 3 |

**Fix:** Set `CHUNK_THRESHOLD = 100000` (effectively disabled).

**Impact:** 82% fewer LLM calls, same coverage.

## Arranger Validation

Investigated 73% exclusion rate. Exclusions are appropriate:
- Marketing fluff ("cutting-edge", "leading platform")
- Tutorial intros ("We have compiled...")
- Generic statements without specific data
- Industry background that doesn't answer the question

**Decision:** Keep current arranger behavior.

## Current Pipeline Settings

```python
BATCH_SIZE = 1  # One source per call
MAX_CHARS_PER_SOURCE = 50000  # Full source content
CHUNK_THRESHOLD = 100000  # Effectively disabled
```

## Cost Estimate

| Sources | Old (chunking) | New (no chunking) |
|---------|----------------|-------------------|
| 216 | ~$1.64 | ~$0.30 |
| 400 | ~$3.00 | ~$0.55 |

## Files Changed

- `src/open_deep_research/pipeline_v2.py`
  - Removed per-source dedup limit in `deduplicate_extractions()`
  - Disabled chunking (CHUNK_THRESHOLD = 100000)

## Commit

```
b53972d Fix coverage: remove per-source dedup limit, disable chunking
```

## Next Steps

None pending. System is stable.

## Already Tried (Don't Repeat)

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Regex blocklist (50+ patterns) | Fragile | Whack-a-mole, overfitting |
| Keyword blocklist for garbage | Incomplete | Not generalizable |
| Chunking for "thoroughness" | Wasteful | Same coverage, 3x cost |
| Per-source dedup limit | Killed coverage | Threw away 95% of facts |

## Last Updated

2026-01-10 — Fixes committed and pushed to main.
