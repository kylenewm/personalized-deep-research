# Complexity Audit: Over-Engineering in Codebase

**Date:** 2026-01-13
**Goal:** Find code with excessive complexity for little added benefit

---

## CRITICAL: High Complexity, Low Value

### 1. Triple Deduplication Functions
**Location:** `pipeline_v2.py:428-561`

Three separate dedup functions doing nearly the same thing:
- `deduplicate_extractions()` - Basic Jaccard (line 428)
- `deduplicate_with_diversity()` - Same + per-source tracking (line 474)
- `deduplicate_extractions_llm()` - LLM-based (line 623)

**Problem:** 90% identical code paths with different thresholds.

**Simpler:** Single `deduplicate(method="jaccard"|"llm", per_source=False)` function.

---

### 2. Artifact System Never Used
**Location:** `artifacts.py` (entire 245-line module), `pipeline_v2.py:1601-1777`

Complex infrastructure:
- RunArtifacts, SourceArtifact, ExtractionArtifact, PointerArtifact classes
- Checkpoints at every stage (pre_dedup, post_dedup, pre_arrangement, post_arrangement)
- Prompt versioning with SHA256 hashing
- 150+ lines of serialization

**Problem:** Artifacts are SAVED but never LOADED. No evidence of consumption in tests or production.

**Simpler:** Delete entirely, or replace with 5-line JSON logging if debugging needed.

---

### 3. Four-Layer Extraction Matching
**Location:** `pointer_extract.py:258-584`

Four matching strategies stacked:
1. Micro-quote exact matching (lines 434-481) - 50 lines
2. Tightest keyword window (lines 258-354) - 100 lines
3. Sentence-based fallback (lines 523-563)
4. Sentence pair/triplet attempts (lines 544-562)

**Problem:** Sentence pairs/triplets solve ~2% of cases but add 20% code complexity.

**Simpler:** Keep only micro-quote + keyword window. Delete sentence triplet logic.

---

## HIGH: Significant Over-Engineering

### 4. Token Limit Detection (120 lines → 3 lines)
**Location:** `utils.py:1342-1506`

- `is_token_limit_exceeded()` dispatcher
- `_check_openai_token_limit()` - 30 lines
- `_check_anthropic_token_limit()` - 20 lines
- `_check_gemini_token_limit()` - 25 lines
- `MODEL_TOKEN_LIMITS` dict with 40+ entries (will go stale)

**Problem:** 95% of logic is just `'token' in error.lower()`

**Simpler:**
```python
def is_token_limit_exceeded(exc):
    return any(x in str(exc).lower() for x in ['token', 'context', 'length', 'prompt is too long'])
```

---

### 5. Search Error Handling Layers
**Location:** `utils.py:410-550`

Multi-layer redundant error handling:
- `gather(..., return_exceptions=True)` (line 435)
- Individual exception checks (lines 443-450)
- ANOTHER Extract API check (line 476)
- Multiple try/except blocks doing same fallback (lines 478-544)
- Truncated source refetch (lines 529-544)

**Problem:** All paths converge to "use raw content if extract fails."

**Simpler:** Single try/except: try Extract API, fall back to raw content. Done.

---

### 6. MCP Auth Wrapper
**Location:** `utils.py:1062-1124`

63 lines wrapping one error code:
- `_find_mcp_error_in_exception_chain()` - Recursive search
- `authentication_wrapper()` - Async wrapper
- Handles only error code -32003

**Simpler:** Check `error.code == -32003` inline at call site.

---

### 7. Citation Validation Functions (Warnings Only)
**Location:** `pipeline_v2.py:1060-1148`

Two validation functions (90 lines total):
- `validate_no_new_facts()` - checks numbers not in facts
- `validate_section_citations()` - checks citation-fact matching

**Problem:** These produce WARNINGS only. Don't block or fix anything. Most violations are false positives.

**Simpler:** Delete both. If citations are wrong, the report is already wrong.

---

### 8. Dual Render Systems
**Location:** `render.py` (350+ lines), `pipeline_v2.py:1330-1405`

- `render.py` - Modern Jinja-style templates
- `pipeline_v2.py` - Old inline HTML (`render_hybrid_report()`)
- `render_html()` tries render.py, falls back to inline

**Problem:** Two systems, unclear which is used, 100+ lines duplicated.

**Simpler:** Delete inline renderer, use only render.py (or vice versa).

---

## MEDIUM: Moderate Complexity

### 9. Configuration Bloat
**Location:** `configuration.py` (758 lines, 60+ fields)

- Many disabled by default: `use_council`, `use_findings_council`, `use_claim_verification`
- Preset system overriding multiple fields
- Fields that don't appear used: `council_max_revisions`, `findings_max_revisions`

**Simpler:** Keep only ACTIVE features. Delete disabled features entirely.

---

### 10. Wrapper Getter Methods
**Location:** `configuration.py:743-753`

```python
def get_effective_max_researcher_iterations(self) -> int:
    return 2 if self.test_mode else self.max_researcher_iterations
```

Three such methods (30 lines) used ~3 times total.

**Simpler:** Inline the ternary at call sites.

---

### 11. Multiple Similarity Systems
**Location:** `pipeline_v2.py:360-426`

Three text comparison approaches:
- `extract_numbers()` - Custom regex
- `compute_text_similarity()` - Jaccard with number protection
- `normalize_for_comparison()` - Markdown stripping

**Simpler:** Single `compare_texts(t1, t2, normalize=True)` function.

---

### 12. Over-Abstracted Data Classes
**Location:** `pipeline_v2.py:77-136`, `pointer_extract.py:18-60`

Classes that are just dicts with names:
- `Citation`, `CuratedFacts`, `ThemeGroup` - 3-5 fields each, no validation or behavior

**Simpler:** Use TypedDict instead of @dataclass for pure data containers.

---

### 13. Dual Verification Functions
**Location:** `pipeline_v2.py:688-754`, `pointer_extract.py:707-788`

- `verify_and_apply_cleanup()` checks if cleaned is substring
- `verify_span()` does the same check

**Simpler:** Single verification function used in both places.

---

## MINOR: Small Inefficiencies

### 14. Defensive Import Pattern (5 occurrences)
**Location:** `pipeline_v2.py:18-70`

```python
try:
    from .pointer_extract import ...
except ImportError:
    from pointer_extract import ...
```

**Problem:** Package structure should be fixed, making these unnecessary.

---

### 15. Unused Config Options
- `claim_pre_check` - enabled but never called
- `council_min_consensus` - councils disabled by default
- `max_claims_to_verify` - claim verification disabled
- `prefer_authoritative_sources` - always True, never False

---

### 16. Prompt Template Redundancy
**Location:** Various prompts (50+ lines each)

Same instructions repeated across prompts:
- "be strict"
- "drop vague content"
- "verify citations"

**Simpler:** PromptBuilder with shared rules + prompt-specific parts.

---

## Summary: Recommended Deletions

| Item | Lines | Risk |
|------|-------|------|
| artifacts.py (entire module) | ~250 | Zero - never loaded |
| Citation validation functions | ~90 | Zero - warnings only |
| Inline render_hybrid_report() | ~80 | Low - use render.py |
| Sentence triplet matching | ~40 | Low - 2% case coverage |
| Token limit verbose checks | ~100 | Zero - simple replacement works |
| MCP auth wrapper | ~60 | Low - inline check works |
| Unused config options | ~50 | Zero - disabled anyway |

**Total potential reduction:** 600-700 lines with zero functional change.

---

## Recommended Consolidations

| Current | Lines | Replacement | New Lines |
|---------|-------|-------------|-----------|
| 3 dedup functions | ~180 | 1 function with options | ~80 |
| 3 similarity helpers | ~70 | 1 unified function | ~30 |
| 2 verification functions | ~100 | 1 shared function | ~50 |
| 3 config getters | ~30 | Inline ternaries | 0 |

**Additional reduction:** ~220 lines.

---

## Priority Order

1. **Delete artifacts.py** - Zero risk, 250 lines
2. **Delete validation warnings** - Zero risk, 90 lines
3. **Consolidate dedup functions** - Medium effort, 100 lines saved
4. **Simplify token limit check** - Low effort, 100 lines saved
5. **Pick one render system** - Medium effort, 80 lines saved
6. **Remove unused config** - Low effort, cleaner code
