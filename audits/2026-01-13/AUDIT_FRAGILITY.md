# Fragility Audit

**Date:** 2026-01-13
**Auditor:** Haiku subagent
**Focus:** Brittle methods that will break on edge cases

## Executive Summary

The system works on curated examples but will fail systematically on real-world data where edge cases are the norm, not exceptions.

---

## WILL DEFINITELY BREAK

### 1. Sentence Splitting on Abbreviations
**File:** `pointer_extract.py:519-525`

Splits on `(?<=[.!?])\s+` but breaks on:
- "Dr. Smith found..." → splits as "Dr" + "Smith found"
- "U.S. markets rose" → splits incorrectly
- "e.g., research showed" → broken

**Note:** This is for quality filtering, not span extraction. May be less critical than stated.

### 2. Number Extraction Without Units Context
**File:** `pipeline_v2.py:358-373`

Extracts numbers but conflates:
- "20MB" vs "20M" vs "20m" (megabyte vs million vs minute)
- "Version 2.1" vs actual metric "2.1%"

### 3. Quality Filter Alpha Ratio Too Aggressive
**File:** `pointer_extract.py:207-212`

`alpha_ratio < 0.35` rejects:
- Financial data: `$0.30/1K | $15/1M | 95%`
- Ticker symbols: `AAPL/MSFT/GOOGL prices fell`

Metric-heavy content (common in research) often <35% alpha.

### 4. Similarity Threshold Magic Numbers
**File:** `pipeline_v2.py:428, 475, 552`

Hard-coded thresholds not validated:
- `0.5` (default dedup) - why 50%?
- `0.7` (intra-source diversity)
- `0.85` (cross-batch dedup)

---

## HIGH FRAGILITY

### 5. Deduplication Based on Word Jaccard
**File:** `pipeline_v2.py:376-411`

Fails on:
- "10,000 concurrent calls" vs "10000 concurrent calls" → different tokens
- "sub-200ms latency" vs "under 200 milliseconds" → low Jaccard
- "NOT available" vs "Available" → opposite meaning, high word overlap

### 6. Keyword Presence Check Ignores Synonyms
**File:** `pipeline_v2.py:483-488`

`if kw in content_lower` is substring, not semantic:
- Looking for "fast" but content has "rapid", "quick"
- Looking for "CEO" but content has "Chief Executive"

---

## MEDIUM FRAGILITY

### 7. Micro-Quote Length Assumption
**File:** `pointer_extract.py:430`

Hardcoded 10-character minimum. Some valid facts are shorter.

### 8. HTML Entity Handling Incomplete
**File:** `pointer_extract.py:70`

Only strips angle brackets, misses:
- `&nbsp;`, `&#169;`, `&lt;`, `&gt;`
- CSS in `<style>`, JavaScript in `<script>`

---

## Suggested Hardening

1. Use spaCy/NLTK for sentence splitting (handles abbreviations)
2. Normalize numbers before dedup (`10,000` → `10000`)
3. Make thresholds configurable, not magic numbers
4. Add semantic keyword expansion (LLM call per run)
