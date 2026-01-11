# STATE.md

## What We're Building

Deep Research Agent — AI-powered research agent that searches the web, gathers sources, and generates verified research reports with anti-hallucination verification.

## Current Status

**Simplified: High trust mode removed. Source quality now a simple boolean flag.**

| Component | Status |
|-----------|--------|
| Pipeline v2 | ✅ Working - pointer extraction + safeguarded synthesis |
| Source quality | ✅ `prefer_authoritative_sources` flag (default true) |
| Render module | ✅ CSS FIXED - two-column layout, footnotes styling |
| Retry logic | ✅ ADDED - exponential backoff on rate limits |
| Upstream sandboxes | ✅ Built - researcher, search, brief |
| Extraction quality | ✅ IMPROVED - avg 24 words, zero > 50 words, zero artifacts |
| Citation quality | ✅ TESTED - 100% citation rate in sandbox |
| Dedup accuracy | ✅ UPGRADED - LLM semantic dedup, 0% false positives |
| Arrangement | ✅ TESTED - correct grouping, 56% exclusion of junk |

## Quality Audit: Complete

| Area | Items | Status |
|------|-------|--------|
| Extraction | 3 | ✅ Fixed (sandbox passing) |
| Deduplication | 3 | ✅ LLM-based (0% FP, 73% recall) |
| Arrangement | 3 | ✅ Sandbox passing |
| Synthesis/Citation | 3 | ✅ Sandbox passing (100% rate) |
| Source Quality | 3 | ✅ Implemented (per-domain + prompt guidance) |

### Sandboxes

| Sandbox | Status | Metrics |
|---------|--------|---------|
| `prompt_sandbox.py` | ✅ Done | word count, artifacts, headers |
| `dedup_sandbox.py` | ✅ Done | LLM semantic matching, 0% FP |
| `citation_sandbox.py` | ✅ Done | 100% citation rate |
| `arrangement_sandbox.py` | ✅ Done | coverage, exclusion, balance |
| `quality_sandbox.py` | ✅ Done | per-domain limit, prompt placeholders |

## Features Implemented This Session

### Extraction Quality Improvements

**Problem:** Facts were too long (avg 40+ words), contained headers/artifacts, and product intros.

**Solution:** Multi-layer invariant-safe filtering:

| Layer | Method | Result |
|-------|--------|--------|
| Prompt | Single-sentence targeting, explicit bad examples | LLM points to one sentence |
| Code | Word limit (50), header patterns, question check | Reject garbage extractions |
| LLM Filter | GPT-4.1 KEEP/REJECT on extracted facts | Final quality gate |

**Metrics (before → after):**
- Avg words: 40.6 → 24.6
- Over 50 words: 11 → 0
- Artifacts: 1 → 0
- Headers: 9 → 0

**New files:**
- `scripts/prompt_sandbox.py` - Autonomous iteration testing (3 sources, ~30s/run)
- Updated `CLAUDE.md` with invariant safeguard section

### LLM-based Deduplication (NEW)

**Problem:** Jaccard similarity can't handle semantic duplicates or entity distinction.
- "200ms latency" vs "180ms latency" → incorrectly marked as duplicate (high word overlap)
- "Claude 95%" vs "GPT-4 95%" → incorrectly marked as duplicate (same metric, different subject)
- "under 500ms" vs "below 500 milliseconds" → incorrectly marked as different (same meaning)

**Solution:** Replaced Jaccard with LLM semantic dedup.

| Metric | Jaccard | LLM |
|--------|---------|-----|
| False Positive Rate | 55% | 0% |
| Recall | 73% | 73% |
| Precision | 45% | 100% |

**Implementation:**
- `deduplicate_extractions_llm()` in `pipeline_v2.py`
- Prompt explicitly handles number/entity differences
- Batch processing (50 facts/call)

**New files:**
- `scripts/dedup_sandbox.py` - Tests against 20 labeled pairs
- `tests/fixtures/dedup_labeled_pairs.json` - Ground truth pairs

### Source Quality Configuration

**Problem:** Need balance between authoritative sources (official docs, papers) and diverse sources (blogs, user opinions). Different queries need different balances.

**Solution:** Added `prefer_authoritative_sources` boolean (default: True)

| Value | Behavior | Use Case |
|-------|----------|----------|
| `True` | Prompts encourage official docs, papers, established sources | Standard research, technical queries |
| `False` | Include diverse sources like blogs, forums, user opinions | Niche topics, "how do people use X" |

### Render Output (Simplified)

- AI writes prose with `[N]` citation markers
- Uncited sentences marked with `[u]` and styled italic
- Verified facts shown as footnotes grouped by theme
- Analysis/Conclusion sections hidden (code preserved)
- Sources & Evidence section serves as "show your work" for raw facts

### Pipeline Parallelization

- Extraction batches now run in parallel with `asyncio.gather()`
- Theme synthesis runs in parallel
- Result: 96 sources in 320s (vs estimated 1180s sequential)

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

## Files Changed This Session

- `src/open_deep_research/pipeline_v2.py`
  - Added `trust_level` param to `run_pipeline_v2()`
  - Skip synthesis/assembly for high trust mode
  - Parallel execution with `asyncio.gather()`

- `src/open_deep_research/render.py`
  - Added `render_high_trust()` function with card layout
  - Added `render_prose_with_citations()` for `[u]` markers
  - Hidden Analysis/Conclusion sections in med trust mode

- `scripts/audit_pipeline.py`
  - Added `--trust` flag (`--trust=high` or `--trust med`)

- `tests/fixtures/gold_queries/voice_agent_eval.json`
  - Saved fixture: 96 sources for voice agent simulation testing

## Current Work

### Simplification Complete

**Changes Made:**
1. ✅ Removed high trust mode entirely (redundant with Sources & Evidence section)
2. ✅ Replaced `trust_level` with `prefer_authoritative_sources: bool` (default True)
3. ✅ Source quality guidance now controlled by boolean flag
4. ✅ Fixed CSS: two-column layout, footnotes styling (serif headers, tighter spacing)
5. ✅ Content balance: columns now balanced by content length, not section count

**Implementation:**
- `configuration.py` - `prefer_authoritative_sources: bool = True`
- 4 prompts inject quality guidance when True: researcher, supervisor, arranger, synthesis
- Per-domain limit (`max_sources_per_domain: 3`) always on

## Next Steps

- (Future) Improve citation quality - ensure LLM cites more consistently
- Test with a fresh query to verify full pipeline

## Already Tried (Don't Repeat)

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Regex blocklist (50+ patterns) | Fragile | Whack-a-mole, overfitting |
| Keyword blocklist for garbage | Incomplete | Not generalizable |
| Chunking for "thoroughness" | Wasteful | Same coverage, 3x cost |
| Per-source dedup limit | Killed coverage | Threw away 95% of facts |
| Jaccard dedup with thresholds | 55% FP rate | Can't understand semantics |
| Jaccard + number protection | 0% FP, 54% recall | Still misses paraphrases |

## Last Updated

2026-01-11 — Simplified: removed high trust mode, replaced with `prefer_authoritative_sources` boolean. CSS fixes complete.
