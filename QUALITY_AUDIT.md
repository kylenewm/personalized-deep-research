# Deep Research Quality Audit

> Systematic review of all pipeline stages with 3 optimization items each.
> Total: 15 items across 5 major areas.

---

## How to Use This Document

For each item:
1. **Current State** - What the code does now
2. **Problem** - Why output quality suffers
3. **Measurement** - How to quantify the problem
4. **Fix Approach** - Invariant-safe solution

---

## Area 1: Search & Source Quality

### 1.1 Source Authority/Quality Scoring

**Current State:**
- Tavily returns sources ranked by relevance
- No distinction between authoritative sources (docs, research) vs low-quality (SEO blogs, aggregators)
- Domain blocklist exists but is reactive (add domains after seeing bad output)

**Problem:**
- Report mixes peer-reviewed research with affiliate marketing content
- Readers can't distinguish trust level of different facts
- Some queries dominated by SEO-heavy sites

**Measurement:**
```python
# Per-report metrics
source_authority = {
    "official_docs": 0,      # docs.*, github.com, arxiv.org
    "news_major": 0,         # reuters, nytimes, wsj
    "blogs_quality": 0,      # known tech blogs
    "unknown": 0,            # everything else
}
# Target: official_docs + news_major > 40%
```

**Fix Approach:**
- Add domain scoring in `utils.py:tavily_search()`
- Score sources 1-5 based on domain
- Pass score through to rendering (show trust badges)
- Log distribution per report

**Files:** `src/open_deep_research/utils.py`

---

### 1.2 Source Freshness

**Current State:**
- No filtering by publication date
- Old content ranked same as new
- For fast-moving topics (AI, crypto, politics), old = wrong

**Problem:**
- Report on "GPT-4 capabilities" might cite 2022 articles about GPT-3
- No way to prefer recent sources for time-sensitive queries

**Measurement:**
```python
# Parse dates from Tavily results
freshness = {
    "last_7_days": 0,
    "last_30_days": 0,
    "last_year": 0,
    "older": 0,
    "unknown": 0,
}
# For tech queries: target last_year > 70%
```

**Fix Approach:**
- Extract publish date from Tavily metadata (if available)
- Add `freshness_weight` parameter to search
- For queries with time-sensitivity keywords ("latest", "2024", "new"), boost recent
- Log freshness distribution

**Files:** `src/open_deep_research/utils.py`

---

### 1.3 Source Diversity

**Current State:**
- No limit on sources per domain
- One authoritative site could dominate (10 pages from same blog)
- Creates echo chamber effect

**Problem:**
- Report reflects single perspective
- If that source is wrong, entire report is wrong
- Readers expect multiple independent sources

**Measurement:**
```python
unique_domains = len(set(urlparse(s.url).netloc for s in sources))
total_sources = len(sources)
diversity_ratio = unique_domains / total_sources
# Target: diversity_ratio > 0.5
```

**Fix Approach:**
- Add per-domain limit (max 3 sources per domain)
- Track diversity in pipeline logs
- Warn if diversity_ratio < 0.3

**Files:** `src/open_deep_research/utils.py`, `nodes/researcher.py`

---

## Area 2: Extraction Quality

### 2.1 Keyword-to-Text Matching Accuracy

**Current State:**
- LLM outputs keywords, code finds matching text
- Uses sentence/pairs/triplets with keyword overlap scoring
- Sorting: score desc, length asc

**Problem:**
- Keywords can match multiple locations in source
- Might extract wrong passage (same keywords, different context)
- No validation that extracted text actually supports the claim LLM intended

**Measurement:**
```python
# Manual evaluation (sample 20 extractions)
extraction_accuracy = {
    "correct_passage": 0,     # Extracted text matches LLM intent
    "wrong_passage": 0,       # Same keywords, different fact
    "partial_match": 0,       # Got part of intended content
}
# Target: correct_passage > 85%
```

**Fix Approach:**
- Add context to pointer: `{"keywords": [...], "claim_type": "metric|comparison|feature"}`
- Code can prefer passages matching claim type
- Log when multiple high-score matches exist (ambiguity signal)

**Files:** `src/open_deep_research/pointer_extract.py:find_best_match()`

---

### 2.2 Extraction Completeness

**Current State:**
- Quality filter rejects garbage (good)
- But also rejects some valid content that's slightly long or has minor artifacts

**Problem:**
- Some valuable technical facts get rejected
- Over-aggressive filtering = coverage loss
- No visibility into what's being rejected

**Measurement:**
```python
# Track rejection reasons
rejection_reasons = {
    "too_long": 0,
    "low_alpha": 0,
    "header_pattern": 0,
    "question": 0,
    "nav_pattern": 0,
    "markdown_artifacts": 0,
}
# Calculate: rejected / (rejected + accepted)
# Target: rejection_rate < 40%
```

**Fix Approach:**
- Log all rejections with reason
- Create rejection analysis sandbox
- Tune thresholds per rejection type
- Add "salvage" logic: try to extract clean substring from rejected text

**Files:** `src/open_deep_research/pointer_extract.py:is_quality_extraction()`

---

### 2.3 Context Preservation

**Current State:**
- Single sentences can lose context ("it achieved 99%..." - what is "it"?)
- Triplet extraction adds context but increases length
- Trade-off: context vs conciseness

**Problem:**
- Some facts are incomprehensible without surrounding context
- Reader has to click source to understand what fact is about
- Pronouns, acronyms, relative references all problematic

**Measurement:**
```python
# Detect context-dependent facts
context_issues = {
    "starts_with_pronoun": 0,     # It, They, This, That
    "undefined_acronym": 0,       # Uses acronym not defined in text
    "relative_reference": 0,      # "the model", "the platform"
}
# Target: context_issues / total < 10%
```

**Fix Approach:**
- Detect pronoun starts in extracted text
- If detected, force triplet extraction (more context)
- Alternatively: prompt LLM to include subject in keywords
- Post-process: prepend subject if detected from prior sentence

**Files:** `src/open_deep_research/pointer_extract.py:find_best_match()`

---

## Area 3: Deduplication

### 3.1 Similarity Threshold Accuracy

**Current State:**
- Jaccard similarity at 0.5 threshold
- Word-level tokenization
- Threshold is hardcoded, not validated

**Problem:**
- False positives: "200ms latency" vs "180ms latency" = 0.67 similarity (duplicate!)
- Numbers treated same as words
- Loses substantive variations

**Measurement:**
```python
# Manual evaluation of dedup decisions
dedup_accuracy = {
    "true_positive": 0,    # Correctly marked as duplicate
    "false_positive": 0,   # Different facts marked duplicate
    "false_negative": 0,   # Same fact not caught
}
# Target: false_positive < 5%
```

**Fix Approach:**
- Protect numbers: normalize "200ms" vs "200 ms" but DON'T merge different numbers
- Test thresholds: 0.3, 0.4, 0.5, 0.6, 0.7
- Expose threshold as config
- Use semantic similarity (embeddings) for borderline cases

**Files:** `src/open_deep_research/pipeline_v2.py:deduplicate_extractions()`

---

### 3.2 Cross-Source vs Same-Source Dedup

**Current State:**
- Same dedup logic for within-source and cross-source
- No distinction between "same fact from 2 sources" vs "variation from same source"

**Problem:**
- Multiple sources confirming same fact = good (shows consensus)
- Same source saying same thing twice = bad (redundant)
- Current logic treats both the same

**Measurement:**
```python
dedup_types = {
    "same_source_duplicate": 0,      # Same URL, duplicate text
    "cross_source_duplicate": 0,     # Different URLs, same fact
    "cross_source_variation": 0,     # Different URLs, similar with new info
}
```

**Fix Approach:**
- Apply stricter threshold for same-source (0.6)
- Apply looser threshold for cross-source (0.4)
- Keep one representative per "fact cluster"
- Annotate facts with "confirmed by N sources"

**Files:** `src/open_deep_research/pipeline_v2.py:deduplicate_extractions()`

---

### 3.3 Semantic vs Lexical Dedup

**Current State:**
- Pure lexical (Jaccard on words)
- Misses semantic duplicates with different wording

**Problem:**
- "Claude achieved 95% accuracy" vs "Anthropic's model scored 95%" = different words, same fact
- Both appear in report as separate facts
- Reader sees redundancy

**Measurement:**
```python
# Use embeddings to find semantic duplicates missed by Jaccard
semantic_check = []
for pair in all_pairs:
    jaccard = compute_jaccard(pair)
    cosine = compute_embedding_similarity(pair)
    if jaccard < 0.4 and cosine > 0.85:
        semantic_check.append(pair)  # Missed duplicate
# Target: len(semantic_check) / total_pairs < 5%
```

**Fix Approach:**
- Two-pass dedup:
  1. Jaccard for obvious duplicates (fast)
  2. Embedding similarity for remaining (slower, more accurate)
- Cache embeddings per extraction
- Only run semantic pass if >50 facts (cost optimization)

**Files:** `src/open_deep_research/pipeline_v2.py:deduplicate_extractions()`

---

## Area 4: Arrangement & Grouping

### 4.1 Theme Coherence

**Current State:**
- LLM groups facts into themes
- No validation that facts actually match theme
- Theme names are LLM-generated

**Problem:**
- Theme "Security Features" might contain pricing facts
- LLM may create overlapping themes
- Inconsistent granularity (one theme has 2 facts, another has 20)

**Measurement:**
```python
# Per-theme coherence check
coherence = {}
for theme in themes:
    # Use LLM judge: "Does this fact belong to this theme?"
    matches = sum(1 for f in theme.facts if judge(f, theme.name))
    coherence[theme.name] = matches / len(theme.facts)
# Target: min(coherence.values()) > 0.8
```

**Fix Approach:**
- Add coherence check after arrangement
- If fact doesn't match theme, move to "Other" or reassign
- Limit themes to 4-6 with balanced fact counts
- Prompt: "Each theme should have 3-8 facts"

**Files:** `src/open_deep_research/pipeline_v2.py:arrange_facts()`

---

### 4.2 Fact Exclusion Rate

**Current State:**
- LLM drops "weak" facts during arrangement
- Exclusion criteria are in prompt, not enforced by code
- Variable exclusion rate (10-60% depending on LLM mood)

**Problem:**
- Some runs drop 60% of facts, others 10%
- No visibility into what's being dropped
- Valuable facts might be excluded as "off-topic"

**Measurement:**
```python
exclusion_metrics = {
    "total_input": len(verified_facts),
    "total_kept": sum(len(g.fact_ids) for g in groups),
    "total_excluded": len(excluded_ids),
    "exclusion_rate": excluded / input,
}
# Target: exclusion_rate between 20-40%
```

**Fix Approach:**
- Log exclusion reasons from LLM
- Set bounds: "Keep at least 50% of facts"
- If exclusion_rate > 50%, re-prompt with stricter instructions
- Manual review: are excluded facts actually low-quality?

**Files:** `src/open_deep_research/pipeline_v2.py:arrange_facts()`

---

### 4.3 Theme Balance

**Current State:**
- No constraints on facts per theme
- One theme might have 15 facts, another has 1

**Problem:**
- Imbalanced themes look weird in report
- Single-fact themes add noise
- Very large themes are hard to synthesize well

**Measurement:**
```python
theme_sizes = [len(g.fact_ids) for g in groups]
balance_metrics = {
    "min_size": min(theme_sizes),
    "max_size": max(theme_sizes),
    "std_dev": statistics.stdev(theme_sizes),
    "single_fact_themes": sum(1 for s in theme_sizes if s == 1),
}
# Target: min_size >= 2, max_size <= 10, std_dev < 3
```

**Fix Approach:**
- Prompt: "Each theme must have 2-8 facts"
- Merge single-fact themes into "Additional Findings"
- Split themes with >10 facts
- Balance check after arrangement, re-prompt if unbalanced

**Files:** `src/open_deep_research/pipeline_v2.py:arrange_facts()`

---

## Area 5: Synthesis & Citation

### 5.1 Citation Rate

**Current State:**
- Prompt says "cite at least 80% of facts"
- No code enforcement
- Actual rate often 30-50%

**Problem:**
- Uncited facts appear in footnotes but not in prose
- Reader doesn't know which facts support which claims
- Trust model breaks: prose makes claims, facts are orphaned

**Measurement:**
```python
citation_metrics = {
    "facts_provided": len(facts),
    "facts_cited": len(set(cited_ids)),
    "citation_rate": cited / provided,
    "prose_sentences": count_sentences(prose),
    "sentences_with_citation": count_with_citation(prose),
}
# Target: citation_rate > 0.8
```

**Fix Approach:**
- After synthesis, check citation_rate
- If < 0.8, re-prompt: "You only cited N/M facts. Revise to cite at least 80%"
- Max 2 re-prompts before accepting
- Log citation rates for monitoring

**Files:** `src/open_deep_research/pipeline_v2.py:synthesize_theme()`

---

### 5.2 Citation Alignment

**Current State:**
- LLM writes `[1]`, `[2]` markers
- Code extracts markers with regex
- Maps marker to fact by position

**Problem:**
- If LLM skips a number (`[1][3]` instead of `[1][2][3]`), mapping breaks
- Marker `[2]` might link to wrong fact
- No validation that cited fact supports the sentence

**Measurement:**
```python
alignment_check = []
for sentence, cited_fact in pairs:
    # Does this fact actually support this sentence?
    support_score = check_support(sentence, cited_fact)
    alignment_check.append(support_score)
alignment_accuracy = sum(alignment_check) / len(alignment_check)
# Target: alignment_accuracy > 0.9
```

**Fix Approach:**
- Validate after synthesis: for each `[N]`, check fact_text appears in semantic context
- If misaligned, log warning (don't auto-fix, as that requires LLM rewrite)
- Add to prompt: "Use sequential numbers [1][2][3], do not skip"
- Parse response to detect skipped numbers, flag if detected

**Files:** `src/open_deep_research/pipeline_v2.py:synthesize_theme()`

---

### 5.3 Uncited Sentence Marking

**Current State:**
- Sentences without citations get `[u]` marker
- Rendered in italic gray
- Intended to show "this is AI opinion"

**Problem:**
- Sentence-level marking has false positives
- "The platform offers real-time monitoring[1] and alerting." - whole sentence marked even though first part is cited
- Confuses readers

**Measurement:**
```python
marking_accuracy = {
    "correctly_marked_u": 0,      # Truly AI-only content
    "incorrectly_marked_u": 0,    # Has citation but marked [u]
    "correctly_unmarked": 0,      # Has citation, not marked
}
# Target: incorrectly_marked_u < 5%
```

**Fix Approach:**
- Mark at clause level, not sentence level
- Split on conjunctions: "X[1] and Y" → only Y gets [u]
- Or: only mark sentences with zero citations AND no fact content
- Alternative: remove [u] marking entirely, trust citation system

**Files:** `src/open_deep_research/render.py:render_prose_with_citations()`

---

## Summary: 15 Items by Priority

| # | Area | Item | Priority | Impact |
|---|------|------|----------|--------|
| 1 | Synthesis | Citation Rate | CRITICAL | Trust |
| 2 | Synthesis | Citation Alignment | CRITICAL | Accuracy |
| 3 | Dedup | Similarity Threshold | HIGH | Coverage |
| 4 | Extraction | Matching Accuracy | HIGH | Quality |
| 5 | Synthesis | Uncited Marking | HIGH | UX |
| 6 | Arrangement | Theme Coherence | HIGH | Quality |
| 7 | Dedup | Semantic Dedup | MEDIUM | Coverage |
| 8 | Extraction | Context Preservation | MEDIUM | Readability |
| 9 | Arrangement | Exclusion Rate | MEDIUM | Coverage |
| 10 | Source | Authority Scoring | MEDIUM | Trust |
| 11 | Arrangement | Theme Balance | MEDIUM | UX |
| 12 | Extraction | Completeness | MEDIUM | Coverage |
| 13 | Dedup | Cross vs Same Source | LOW | Quality |
| 14 | Source | Freshness | LOW | Accuracy |
| 15 | Source | Diversity | LOW | Quality |

---

## Next Steps

1. Create sandboxes for top 5 items
2. Define metrics and collect baseline data
3. Iterate on fixes
4. Re-measure to confirm improvement
5. Update ARCHITECTURE.md with changes

---

## Related Files

| File | Areas |
|------|-------|
| `pointer_extract.py` | Extraction (2.1-2.3) |
| `pipeline_v2.py` | Dedup (3.1-3.3), Arrangement (4.1-4.3), Synthesis (5.1-5.2) |
| `render.py` | Synthesis (5.3) |
| `utils.py` | Source (1.1-1.3) |

---

*Last updated: 2026-01-10*
