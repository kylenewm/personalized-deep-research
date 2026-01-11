# STATE.md

## What We're Building

Deep Research Agent — AI-powered research agent that searches the web, gathers sources, and generates verified research reports with anti-hallucination verification.

## Current Status

**Source quality guidance implemented. All sandboxes passing.**

| Component | Status |
|-----------|--------|
| Pipeline v2 | ✅ Working - pointer extraction + safeguarded synthesis |
| Trust modes | ✅ ADDED - `high` (facts only) and `med` (prose + citations) |
| Source quality | ✅ ADDED - per-domain limit + prompt guidance for trust=high |
| Render module | 🔧 CSS ISSUES - two-column layout broken, needs fixes |
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

### Trust-Level Modes

**Problem:** Tradeoff between readability (AI prose) and trust (verified facts only). Different use cases need different balances.

**Solution:** Added `trust_level` config with two modes:

| Mode | Trust | Readability | Use Case |
|------|-------|-------------|----------|
| `high` | Zero hallucination risk | Lower (facts only) | Legal, compliance, research |
| `med` | Some risk (marked with [u]) | Higher (prose + facts) | Overview, presentation |

### High Trust Mode (facts only)

- Skips synthesis and assembly LLM calls entirely
- Renders verified facts in card-based layout per theme
- Progressive disclosure: "Show N more" for long lists
- Zero AI prose = zero hallucination risk

### Med Trust Mode (prose + citations)

- AI writes prose with `[N]` citation markers
- Uncited sentences marked with `[u]` and styled italic
- Verified facts shown as footnotes grouped by theme
- Analysis/Conclusion sections hidden (code preserved)

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

### Source Quality Guidance (COMPLETE)

**Goal:** Add source quality guidance tied to `trust_level`. When trust=high, prompts encourage preferring authoritative sources.

**Approach:**
- Tie to existing `trust_level` (no new config flags)
- Add prompt guidance to 4 prompts (researcher, supervisor, arranger, synthesis)
- Add per-domain limit (max 3 per domain) - always on

**Implementation:**
1. ✅ Plan approved
2. ✅ Added `max_sources_per_domain: int = 3` to configuration.py
3. ✅ Implemented per-domain limit in utils.py:tavily_search()
4. ✅ Added quality guidance to 4 prompts (conditional on trust=high):
   - `research_system_prompt` - prefer authoritative sources
   - `lead_researcher_prompt` - quality check after research
   - `ARRANGER_PROMPT` - prefer facts from authoritative sources
   - `THEME_SYNTHESIS_PROMPT` - cite authoritative sources first
5. ✅ Created `quality_sandbox.py` - all tests passing

## Next Steps

- (Future) Improve citation quality - LLM not citing enough in med trust mode
- (Future) Implement `trust_level: "low"` for more coverage

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

2026-01-11 — Source quality guidance implemented (per-domain limit + prompt guidance for trust=high). All sandboxes passing.
