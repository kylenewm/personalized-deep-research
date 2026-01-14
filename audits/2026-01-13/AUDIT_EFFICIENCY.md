# Efficiency Audit

**Date:** 2026-01-13
**Auditor:** Haiku subagent
**Potential Savings:** 30-50% of pipeline cost

## Executive Summary

The pipeline is designed for precision over cost, but leaves 30-50% of budget on the table through sequential operations that could be batched or parallelized.

---

## CRITICAL (Major Cost Impact)

### 1. BATCH_SIZE = 1 for Extraction
**File:** `pipeline_v2.py:142`

```python
BATCH_SIZE = 1  # Process one source at a time for thoroughness
```

- 50 sources = 50 LLM calls when 5-10 batches could handle them
- **Cost multiplier:** 5-10x excess on extraction phase

### 2. Deduplication LLM Called Sequentially in Batches
**File:** `pipeline_v2.py:621-679`

Processes facts in batches of 50, making separate LLM calls for each batch.
- 200 facts = 4 LLM calls instead of 1
- **Cost multiplier:** 3-10x excess on dedup

### 3. Cleanup Extractions: Sequential, No Parallelization
**File:** `pipeline_v2.py:686-752`

Processes facts in batches of 20 sequentially.
- 100 facts = 5 LLM calls when could be 1-2 parallel
- **Cost multiplier:** 2.5-5x excess

---

## HIGH IMPACT

### 4. No Source Deduplication Before Extraction
**File:** `supervisor.py:246-266`

No content-level deduplication of similar sources before extraction.
- Extracting from 150 near-duplicate sources wastes 50%+ of calls

### 5. No Smart Source Limiting or Prioritization
**File:** `supervisor.py:243-275`

Sources accumulated up to max (200) with no ranking/priority.
- First-come-first-served, worse sources processed same as better ones

### 6. max_retries=7 for Dedup (Excessive)
**File:** `pipeline_v2.py:661-662`

```python
max_retries=7,  # Can wait up to 6+ minutes
base_delay=3.0
```

Standard is max_retries=3.

---

## Already Good (Parallelized)

- Theme synthesis: `asyncio.gather()` at line 1694-1697
- Report assembly: 3 parallel calls at line 1289

---

## Optimization Summary

| Fix | Current | Optimized | Savings |
|-----|---------|-----------|---------|
| BATCH_SIZE 1→10 | 50 calls | 5 calls | 80% |
| Dedup batching | 4-10 calls | 1-2 calls | 70% |
| Cleanup parallel | 5 sequential | 1-2 parallel | 60% time |
| Source pre-rank | 200 sources | 50 sources | 60% |

**Total potential savings: 30-50% of pipeline cost**
