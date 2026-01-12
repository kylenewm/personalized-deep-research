# STATE.md

## What We're Building

Deep Research Agent — AI-powered research agent that searches the web, gathers sources, and generates verified research reports with anti-hallucination verification.

## Current Status

**QUALITY ISSUES FOUND: Three architectural gaps identified**

| Component | Status |
|-----------|--------|
| Pipeline v2 | ✅ Working - pointer extraction + safeguarded synthesis |
| Source quality | ⚠️ SOFT ONLY - `prefer_authoritative_sources` is prompt hint, no hard filtering |
| Render module | ✅ FIXED - proper template, citations, footnotes layout |
| Synthesis | ✅ FIXED - plain prose output, no JSON artifacts |
| Citation marking | ✅ FIXED - correctly identifies cited vs uncited sentences |
| Retry logic | ✅ ADDED - exponential backoff on rate limits |
| Extraction quality | ✅ IMPROVED - avg 24 words, zero > 50 words, zero artifacts |
| Dedup accuracy | ⚠️ BATCH-LIMITED - cross-batch duplicates leak through |
| Arrangement | ✅ TESTED - correct grouping, 56% exclusion of junk |
| Query specificity | ⚠️ DILUTED - supervisor loses query nuance |

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

- `src/open_deep_research/nodes/safeguarded_report.py`
  - Changed import from `render_hybrid_report` to `render_report` from render.py
  - Line 136: now calls `render_report(report)` for proper template rendering

- `src/open_deep_research/render.py`
  - Fixed `render_prose_with_citations()` regex - now checks if sentence CONTAINS citation
  - Fixed footnotes layout - moved outside `.columns` div
  - Simplified `.footnotes-section` CSS - removed broken `calc(50vw)` hack

- `src/open_deep_research/pipeline_v2.py`
  - Removed JSON output requirement from `THEME_SYNTHESIS_PROMPT`
  - Now outputs plain prose only with `[N]` citations inline
  - Simplified response parsing: `prose = response.strip()`

- `tests/fixtures/gold_queries/latest_research.json`
  - Voice agent orchestration query fixture (42 sources)

## Current Work

### E2E Test Run: Agentic Orchestration (2026-01-12)

**Query:** "What are the top agentic orchestration methods in 2026 for building long contextual systems?"

**Results:**
- 70 sources collected
- 212 verified extractions → 179 after dedup → 172 after cleanup
- 63 facts kept in 5 themes
- Report rendered successfully

**Saved:**
- `tests/fixtures/gold_queries/latest_research.json` - full data
- `orchestration_2026_report.html` - rendered report

### Quality Issues Found (2026-01-12)

**Issue 1: Duplicate facts leaked through dedup**
- Facts [11], [14], [16] are identical ("LangGraph 2.2x faster than CrewAI")
- Root cause: `deduplicate_extractions_llm()` processes in batches of 50
- Cross-batch duplicates are never compared
- Location: `pipeline_v2.py:492-550`

**Issue 2: Medium blogs got through despite `prefer_authoritative_sources`**
- Root cause: Flag is **soft prompt guidance only**, not hard filtering
- `blocked_domains` exists but only blocks YouTube/Reddit/social media
- No domain authority scoring at search time
- Location: `pipeline_v2.py:673-764`

**Issue 3: "Long contextual systems" depth missing**
- Query asked specifically about long-context techniques
- Report focused on generic framework comparisons
- Root cause: Supervisor's `research_topic` diluted query specificity
- Location: `nodes/supervisor.py:142-168`

### Architectural Gaps Summary

| Issue | Root Cause | Fix Complexity |
|-------|------------|----------------|
| Duplicate facts | Batch dedup (50/batch) - no global pass | Medium |
| Low-quality sources | `prefer_authoritative_sources` is prompt-only | Medium |
| Lost query specificity | Supervisor dilutes research_topic | Hard |

### Previous E2E Test (2026-01-11)

**Query:** Multi-agent orchestration frameworks for AI coding assistants (2026)

**Results:**
- 58 sources collected
- 149 verified extractions → 129 after dedup → 120 after cleanup
- 78 facts kept in 5 themes
- Report rendered successfully

**Saved for downstream work:**
- `tests/fixtures/gold_queries/agentic_coding_2026.json` - full data (sources, hybrid_report)
- `agentic_coding_report.html` - rendered report

### Known Issues (Not Fixing - Would Be Over-Engineering)

**Extraction quality:** Some garbage still slips through (~5%):
- Marketing fluff: "Domo transforms the way..."
- Markdown artifacts: `### Header` mixed into text
- First-person opinions: "I find that..."
- Vague claims with no metrics

**Why not fixing:** Prompt already tells LLM to reject these. Adding regex filters is whack-a-mole (see "Already Tried"). Accept some noise.

### Render Pipeline Fix (COMPLETED - 4 Issues Fixed)

1. Wrong render function → now uses `render_report()` from render.py
2. Citation regex → now checks if sentence CONTAINS `[N]` anywhere
3. Footnotes layout → moved outside `.columns` div
4. JSON artifacts → synthesis outputs plain prose only

### Documentation Updates

- Added to CLAUDE.md: pipeline bypass flags (`review_mode: 'none'`)
- Added to CLAUDE.md: always ask about saving data before running reports

## Next Steps

1. **Fix duplicate leak** - Add global dedup pass after batch dedup
2. **Add source scoring** - Domain authority tier (official > papers > news > blogs)
3. **Preserve query specificity** - Improve supervisor prompt to maintain nuance
4. **Re-run targeted query** - Test "long context memory architectures" specifically

## Already Tried (Don't Repeat)

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Regex blocklist (50+ patterns) | Fragile | Whack-a-mole, overfitting |
| Keyword blocklist for garbage | Incomplete | Not generalizable |
| Chunking for "thoroughness" | Wasteful | Same coverage, 3x cost |
| Per-source dedup limit | Killed coverage | Threw away 95% of facts |
| Jaccard dedup with thresholds | 55% FP rate | Can't understand semantics |
| Jaccard + number protection | 0% FP, 54% recall | Still misses paraphrases |
| JSON output from synthesis | Leaked artifacts | LLM outputs JSON structure in prose |
| Regex cleanup of JSON | Fragile | Whack-a-mole, won't scale |

## Last Updated

2026-01-12 — Quality audit on orchestration query. Found 3 architectural gaps: batch dedup leak, soft-only source filtering, query specificity dilution. Next: fix dedup, add source scoring.
