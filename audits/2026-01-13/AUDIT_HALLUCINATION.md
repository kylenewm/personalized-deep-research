# Hallucination Risk Audit

**Date:** 2026-01-13
**Auditor:** Haiku subagent
**Risk Level:** MEDIUM-HIGH

## Executive Summary

The pipeline implements strong structural guardrails (Invariant I1: "LLM points, code extracts verbatim"). However, there are gaps where LLM-generated content escapes verification and enters the final report. The core extraction pipeline is solid, but synthesis layers have unvalidated hallucination injection points.

---

## CRITICAL

### C1: Executive Summary + Analysis + Conclusion Are Uncited
**File:** `pipeline_v2.py:1153-1208`

LLM writes 3 sections (Executive Summary, Analysis, Conclusion) based ONLY on summary snippets, not full fact text. LLM can:
- Synthesize new claims not in facts
- Mix facts between sources
- Invent comparative claims ("X is better than Y")
- Hallucinate metrics

```python
# Line 1238-1239: Summary passes ONLY first 3 facts per theme, truncated to 150 chars
for f in s.facts[:3]:
    summary_parts.append(f"  - {f.extracted_text[:150]}...")
```

### C2: Synthesis Violations Are Logged But Not Blocking
**File:** `pipeline_v2.py:1700-1722`

When `validate_no_new_facts()` or `validate_section_citations()` find violations, they're logged as warnings only (non-blocking).

```python
if synthesis_violations:
    # Log warnings but don't block - these are diagnostic for now
    logger.warning(f"Synthesis validation warnings ({len(synthesis_violations)} issues):")
```

### C3: Synthesis Prompts Allow Arbitrary "Synthesis"
**File:** `pipeline_v2.py:1166-1196`

ANALYSIS_PROMPT explicitly invites interpretation:
```
2. DRAW INSIGHTS: What conclusions can we draw from these specific facts?
- ADD VALUE by connecting, comparing, contextualizing
```

### C4: Theme Synthesis Prose Can Introduce New Facts
**File:** `pipeline_v2.py:921-958`

Each themed section starts with LLM-written prose. Anchor validation is weak - if ONE anchor matches, validation passes even if other claims are fabricated.

---

## HIGH RISK

### H1: Arranger Can Group Unrelated Facts Into Same Theme
**File:** `pipeline_v2.py:759-807`

LLM groups verified facts into themes. If arranger misgroups facts, synthesis will connect them with fake causation.

### H2: Citation Markers Can Point to Wrong Facts
**File:** `pipeline_v2.py:1034-1048`

Theme synthesis extracts `[N]` citations from LLM prose, maps to facts by sequential position. Fragile if facts reordered.

### H3: Cleanup Can Semantically Alter Meaning
**File:** `pointer_extract.py:765-811`

LLM "cleans" extracted facts, then code verifies it's a substring. But cleanup can semantically alter meaning while technically being a substring.

### H4: Executive Summary Sees Only Theme Names
**File:** `pipeline_v2.py:1153-1163`

Executive summary sees `sections_overview` which only has theme NAMES and fact COUNTS, not fact content.

---

## Currently Mitigated (Good)

- **Pointer Extraction:** LLM outputs keywords/pointers, code extracts verbatim text
- **Micro-Quote Matching:** Requires exact 8-15 word phrase from source
- **Span Verification:** All extractions record character positions
- **Cleanup Guards:** Negation/number removal prevented
- **Quality Filter:** Rejects tables, navigation, metadata

---

## Recommendations

1. Block report generation if validation violations found
2. Analysis section needs citation markers or removal
3. Executive summary should see full fact text, not truncation
4. Make validation blocking, not just logging
