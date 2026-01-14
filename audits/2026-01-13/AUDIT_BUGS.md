# Bugs and Error Handling Audit

**Date:** 2026-01-13
**Auditor:** Haiku subagent
**Focus:** Code that will crash or lose data in production

## Executive Summary

Multiple categories of bugs and missing error handling that will cause crashes or silent data loss under production load.

---

## CRITICAL (Will Crash)

### C1: Division by Zero in `find_tightest_keyword_window()`
**File:** `pointer_extract.py:347`

```python
coverage_ratio = len(matched) / len(keywords)
```

If `keywords` list is empty after filtering (whitespace-only keywords), this divides by zero.

**Trigger:** LLM outputs pointer with keywords like `["  ", "  "]`

### C2: Index Out of Bounds in Span Offset
**File:** `pointer_extract.py:640-645`

`source_content.find(cleaned)` can return -1 if cleaned text not found. This -1 is used as span_start without validation.

### C3: asyncio.gather Crashes on Any Failure
**File:** `pipeline_v2.py:1289`

```python
exec_summary, analysis, conclusion = await asyncio.gather(
    generate_executive_summary(...),
    generate_analysis(...),
    generate_conclusion(...)
)
```

No `return_exceptions=True`. If ANY task fails, entire pipeline crashes and loses all results.

### C4: Missing Validation on Arranger LLM Response
**File:** `pipeline_v2.py:830-878`

If LLM returns `{"groups": [{"fact_ids": "1,2,3"}]}` (string instead of list), code iterates characters not numbers.

### C5: Checkpoint Save Can Fail Silently
**File:** `pipeline_v2.py:1758-1759`

If checkpoint directory is read-only, `mkdir()` doesn't raise but `open()` fails with unhandled PermissionError.

---

## HIGH (Data Loss)

### H1: Race Condition in Parallel Extraction
**File:** `pipeline_v2.py:339-342`

If any batch fails in `asyncio.gather()`, all completed batches' results are lost. No partial result aggregation.

### H2: Cleanup Rejection Loses Data Silently
**File:** `pipeline_v2.py:744-746`

When `verify_and_apply_cleanup()` returns None (garbage), extraction is dropped with no logging. Valid facts can disappear.

### H3: Missing Null Check on `.pointer` Attribute
**File:** `pipeline_v2.py:980`

```python
lines.append(f"    Source: {ext.pointer.context}")
```

If `pointer=None` in error cases, this crashes.

### H4: No Timeout on LLM Calls
**File:** Throughout pipeline

`retry_with_backoff()` has no overall timeout. If LLM hangs, pipeline waits forever.

---

## MEDIUM (Incorrect Output)

### M1: Empty Theme Handling
**File:** `pipeline_v2.py:967-982`

If theme has zero facts after filtering, synthesis prompt receives empty facts → LLM synthesizes nonsense.

### M2: Off-By-One in Dedup Index Mapping
**File:** `pipeline_v2.py:667-674`

LLM outputs 1-indexed fact numbers but calculation assumes batch-relative indexing. Can corrupt fact assignments.

### M3: Citation Extraction Doesn't Validate Bounds
**File:** `pipeline_v2.py:1036-1048`

If LLM outputs `[12]` but only 5 facts exist, citation is silently dropped from list but remains in prose.

---

## Recommended Priority Fixes

**Week 1 (Critical):**
1. C1 - Division by zero guard
2. C3 - asyncio.gather with return_exceptions=True
3. H4 - Add timeouts to async operations

**Week 2 (High):**
4. C2 - Index bounds checking
5. H2 - Log cleanup rejections

**Week 3 (Defensive):**
6. Circuit breaker for rate limiting
7. Structured logging
