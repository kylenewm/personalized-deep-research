# Deep Research V0 - Trust Pipeline Refactoring Summary

## Overview

This document summarizes the refactoring work done on the Deep Research agent to implement an anti-hallucination "Trust Pipeline" that verifies quotes exist in source material before including them in reports.

## What Was Built

### The Problem
LLMs can hallucinate quotes and citations. The original monolithic code had no verification that quotes actually existed in source documents.

### The Solution: Trust Pipeline
A deterministic verification layer that:
1. **Extracts** candidate quotes from raw HTML sources
2. **Verifies** each quote exists in the source (substring + fuzzy matching)
3. **Selects** only verified quotes for the final report (no invention allowed)

---

## Refactoring Steps Completed

### S01: Split Monolith
**What:** Broke `deep_researcher.py` (1248 lines) into modular node files.

**Files Created:**
- `src/open_deep_research/graph.py` - Main graph construction
- `src/open_deep_research/models.py` - Pydantic models
- `src/open_deep_research/nodes/clarify.py` - User clarification
- `src/open_deep_research/nodes/brief.py` - Research brief + validation
- `src/open_deep_research/nodes/supervisor.py` - Supervisor subgraph
- `src/open_deep_research/nodes/researcher.py` - Researcher subgraph
- `src/open_deep_research/nodes/findings.py` - Fact-check findings
- `src/open_deep_research/nodes/report.py` - Final report generation
- `src/open_deep_research/nodes/verify.py` - Claim verification

---

### S02: Trust Store & Gating
**What:** Added fail-fast check for LangGraph Store availability.

**Files Modified:**
- `src/open_deep_research/state.py` - Added `verified_disabled: bool` field
- `src/open_deep_research/nodes/store.py` - NEW: `check_store` gating node
- `src/open_deep_research/graph.py` - Wired `check_store` as entry point

**How It Works:**
```
START → check_store → (sets verified_disabled if Store unavailable) → ...
```

If Store is unavailable, `verified_disabled=True` and the Verified Findings section is skipped.

---

### S03: Evidence Extraction
**What:** Deterministic extraction of candidate quotes from HTML sources.

**Files Created:**
- `src/open_deep_research/logic/sanitize.py` - HTML sanitization
- `src/open_deep_research/nodes/extract.py` - Evidence extraction node

**Key Functions:**
```python
sanitize_for_quotes(html)     # Strip tags, preserve paragraph structure
extract_paragraphs(text)      # Filter to 15-60 words with noun/number
extract_evidence(state, config)  # Main node function
```

**State Added:**
```python
class EvidenceSnippet(TypedDict):
    snippet_id: str      # Hash of source_id + quote
    source_id: str       # Source URL
    source_title: str    # Title
    quote: str           # Verbatim text (15-60 words)
    status: str          # PENDING | PASS | FAIL | SKIP
```

---

### S04: Verification Logic
**What:** Deterministic verification that quotes exist in sources.

**Files Modified:**
- `src/open_deep_research/nodes/verify.py` - Added `verify_evidence` function

**Verification Algorithm:**
```python
def verify_quote_in_source(quote, source_content):
    # Check 1: Exact substring match
    if quote in clean_source:
        return "PASS"

    # Check 2: Fuzzy match (Jaccard similarity > 0.8)
    if jaccard_similarity(quote, window) >= 0.8:
        return "PASS"

    return "FAIL"
```

**Key Point:** NO LLM calls - purely deterministic string matching.

---

### S05: Wire Graph & Report Selector
**What:** Connected all nodes and added "Selector Mode" for reports.

**Files Modified:**
- `src/open_deep_research/graph.py` - Wired extract → verify → report
- `src/open_deep_research/nodes/report.py` - Added Selector Mode
- `src/open_deep_research/prompts.py` - Added verified findings prompts

**Graph Flow:**
```
check_store → clarify → brief → validate_brief → supervisor
           → validate_findings → extract_evidence → verify_evidence
           → final_report_generation → verify_claims → END
```

**Selector Mode Prompt:**
The LLM must SELECT from verified quotes, not generate new ones:
```
SELECTOR MODE RULES:
1. Select 3-5 quotes from AVAILABLE_VERIFIED_QUOTES
2. Copy each quote EXACTLY - no paraphrasing
3. Format as: * "[Quote]" - [Source Title](URL)
```

---

### S06: E2E Verification
**What:** Created test suite to verify the Trust Pipeline works.

**Files Created:**
- `tests/__init__.py`
- `tests/e2e.py` - 28 tests covering all components

**Test Categories:**
| Class | Tests | What It Tests |
|-------|-------|---------------|
| TestSanitization | 7 | HTML sanitizer (S03) |
| TestVerification | 8 | Jaccard/substring matching (S04) |
| TestSelectorMode | 4 | Quote formatting (S05) |
| TestTrustPipelineIntegration | 4 | Full pipeline flow |
| TestAcceptanceCriteria | 3 | S06 requirements |

---

## System Invariants (Must Be Preserved)

1. **I1** - Verified Findings use Selector pattern only (no invention)
2. **I2** - Verification is deterministic (code-based, not LLM)
3. **I3** - LangGraph Store is canonical source for raw content
4. **I4** - Fail-safe gating when Store unavailable
5. **I5** - CLI-only scope (no web server)
6. **I6** - No secrets in repo
7. **I7** - No sweeping refactors (scoped changes only)
8. **I8** - Deterministic, scoped file writes

---

## File Structure After Refactoring

```
src/open_deep_research/
├── __init__.py              # Package exports
├── graph.py                 # Main graph construction (S01, S05)
├── models.py                # Pydantic models (S01)
├── configuration.py         # Config management (existing)
├── council.py               # Multi-stage review (existing)
├── prompts.py               # Prompts + Selector Mode (S05)
├── state.py                 # State definitions (S02, S03)
├── utils.py                 # Utilities (existing)
├── verification.py          # Embedding verification (existing)
├── logic/
│   ├── __init__.py
│   └── sanitize.py          # HTML sanitization (S03)
├── nodes/
│   ├── __init__.py
│   ├── clarify.py           # User clarification (S01)
│   ├── brief.py             # Research brief (S01)
│   ├── extract.py           # Evidence extraction (S03)
│   ├── findings.py          # Fact-check findings (S01)
│   ├── report.py            # Report generation + Selector Mode (S01, S05)
│   ├── researcher.py        # Researcher subgraph (S01)
│   ├── store.py             # Store gating (S02)
│   ├── supervisor.py        # Supervisor subgraph (S01)
│   └── verify.py            # Evidence + claim verification (S01, S04)
├── prompts/
│   └── __init__.py
└── tools/
    └── __init__.py

tests/
├── __init__.py
└── e2e.py                   # E2E tests (S06)
```

---

## How to Run

### Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install
pip install -e .
```

### Run Tests
```bash
pytest tests/e2e.py -v
```

### Run the Agent
```bash
# Set API keys
export OPENAI_API_KEY=your_key
export TAVILY_API_KEY=your_key

# Run with LangGraph
langgraph dev
```

---

## Key Code to Review

### 1. Evidence Extraction (`nodes/extract.py`)
```python
async def extract_evidence(state, config):
    # Get sources from state
    # For each source: sanitize HTML → extract paragraphs → create snippets
    # Return snippets with status=PENDING
```

### 2. Evidence Verification (`nodes/verify.py`)
```python
def verify_quote_in_source(quote, source_content, fuzzy_threshold=0.8):
    # Strict: exact substring match → PASS
    # Fuzzy: Jaccard similarity > 0.8 → PASS
    # Else → FAIL

async def verify_evidence(state, config):
    # For each snippet: verify against source
    # Update status to PASS or FAIL
```

### 3. Selector Mode (`nodes/report.py`)
```python
def format_verified_quotes(snippets):
    # Filter to PASS only
    # Format as: '"Quote" - [Title](URL)'

async def final_report_generation(state, config):
    # Build verified_section from evidence_snippets
    # Append selector prompt to report generation
```

### 4. Graph Wiring (`graph.py`)
```python
# Flow: extract_evidence → verify_evidence → final_report_generation
deep_researcher_builder.add_edge("validate_findings", "extract_evidence")
deep_researcher_builder.add_edge("extract_evidence", "verify_evidence")
deep_researcher_builder.add_edge("verify_evidence", "final_report_generation")
```

---

## For ChatGPT Code Review

To have ChatGPT review this code, paste the following files:

1. **Core Logic:**
   - `src/open_deep_research/logic/sanitize.py` (HTML sanitization)
   - `src/open_deep_research/nodes/extract.py` (extraction)
   - `src/open_deep_research/nodes/verify.py` (verification)

2. **Integration:**
   - `src/open_deep_research/graph.py` (wiring)
   - `src/open_deep_research/nodes/report.py` (Selector Mode)

3. **State:**
   - `src/open_deep_research/state.py` (data structures)

4. **Tests:**
   - `tests/e2e.py` (verification tests)

**Review Prompt:**
```
Please review this Trust Pipeline implementation for an anti-hallucination
research agent. Key requirements:

1. Evidence extraction must be deterministic (no LLM)
2. Verification uses substring + Jaccard similarity (threshold 0.8)
3. Report uses "Selector Mode" - only selects from verified quotes
4. If Store unavailable, verified_disabled=True skips verification

Check for:
- Logic bugs in verification algorithm
- Edge cases in HTML sanitization
- Proper state management
- Graph wiring correctness
```
