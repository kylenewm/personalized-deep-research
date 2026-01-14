# Audit Synthesis and Prioritization

**Date:** 2026-01-13
**Synthesized by:** Claude Opus

## Overview

Four audit agents reviewed the codebase for:
1. Hallucination risks
2. Efficiency issues
3. Fragility/brittleness
4. Bugs and errors

This document synthesizes findings and separates real issues from false positives.

---

## Fix Now (Will Crash or Lose Data)

### 1. asyncio.gather without return_exceptions
**File:** `pipeline_v2.py:1289`

One failed LLM call kills the whole report. Easy 5-line fix.

```python
# Before
exec_summary, analysis, conclusion = await asyncio.gather(...)

# After
results = await asyncio.gather(..., return_exceptions=True)
exec_summary = results[0] if not isinstance(results[0], Exception) else ""
# etc.
```

### 2. Division by zero on empty keywords
**File:** `pointer_extract.py:347`

Add early return if keywords empty after normalization.

```python
if not keywords_lower:
    return None, -1, -1, [], 0.0
```

### 3. Add timeout to LLM calls
**Recommendation:** 20 seconds per call

```python
result = await asyncio.wait_for(llm_call(), timeout=20)
```

---

## Fix Soon (Real Degradation)

### 4. Span verification after normalization

**What it means:** When we extract text, we record WHERE in the source it came from (span_start, span_end). Later, `verify_span()` checks that the extracted text actually exists at that position.

**The bug:** Cleanup normalizes whitespace (multiple spaces → single space), but verification uses exact string matching. So:
- Original: `"The   model   achieved"`  (3 spaces)
- Cleaned: `"The model achieved"` (1 space)
- `verify_span()` looks for cleaned text at original position → doesn't find it → rejects valid extraction

**Fix:** Either normalize both sides before comparing, or use fuzzy matching with high threshold (0.95+).

### 5. BATCH_SIZE increase from 1 to 5-10

**What it means:** Currently we make 1 LLM call per source. With 50 sources, that's 50 separate API calls.

**Why it's expensive:** Each API call has overhead (latency, token minimums). Batching 5 sources into 1 call could reduce costs 80%.

**Why it was set to 1:** Comment says "for thoroughness" - concern that batching multiple sources might confuse the LLM or reduce extraction quality.

**Fix:** Test with BATCH_SIZE=5, compare extraction quality. If quality holds, keep the savings.

---

## Don't Fix (Working as Designed)

### Analysis/Conclusion sections uncited
That's intentional synthesis. The LLM is SUPPOSED to draw conclusions. Just needs clear UI labeling that these sections are AI interpretation, not verified facts.

### "3 dedup passes"
Intentional design: LLM semantic dedup catches paraphrases, text similarity dedup catches near-duplicates. Different purposes.

### "Sentence splitting breaks on abbreviations"
Audit misread the code. We use window matching for extraction, not sentence splitting. Sentence splitting is only in quality filtering, which is less critical.

---

## Possibly Overstated by Audits

### "Citation markers can point to wrong facts"
The mapping is sequential WITHIN each theme, not globally. Each theme gets its own fact list [1], [2], [3]. This should work correctly.

### "max_retries=7 is excessive"
7 retries with exponential backoff = ~3 minutes max wait. For a $50 research run, 3 minutes of retries is reasonable. Not critical to change.

---

## Summary

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| NOW | asyncio.gather exceptions | 5 lines | Prevents crashes |
| NOW | Division by zero guard | 2 lines | Prevents crashes |
| NOW | LLM timeout (20s) | 10 lines | Prevents hangs |
| SOON | Span verification normalization | 20 lines | Fixes false rejections |
| SOON | BATCH_SIZE increase | Config change | 80% cost reduction |
| DEFER | Everything else | - | Working as designed |
