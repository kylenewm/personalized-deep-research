# Deep Research V1 - Master Plan

> **Structure**:
> - **Part A**: Phase 1 Implementation Plan ✅ COMPLETE
> - **Part A2**: Claim Verification Eval Framework (DO NOW)
> - **Part B**: North Star Reference (FUTURE)

---

# PART A2: Claim Verification Eval Framework

## Goal
Create an evaluation framework that verifies ALL claims in the final report against sources, producing structured metrics as a post-hoc quality check.

## Context

**What exists:**
- `verify_claims` node in `nodes/verify.py` (currently disabled - $0.45/run)
- `verification.py` - full claim extraction + LLM verification logic
- `run_verification.py` - standalone script for verification
- S03/S04 - deterministic quote verification (different from claim verification)

**What user wants:**
- Eval framework to verify all claims in final report
- Secondary check (like verify_claims) as part of eval
- Structured metrics output

## Design

### Two-Layer Verification

| Layer | What | How | When |
|-------|------|-----|------|
| **S03/S04** | Quote verification | Deterministic | During pipeline |
| **Eval** | Claim verification | LLM-based | After report (opt-in) |

### Output Format

```json
{
  "run_id": "20260105_143022",
  "query": "What are best practices for remote work?",

  "claims": {
    "total": 25,
    "supported": 18,
    "partially_supported": 4,
    "unsupported": 2,
    "uncertain": 1,
    "support_rate": 0.88
  },

  "citations": {
    "total": 76,
    "verified": 68,
    "accuracy": 0.89
  },

  "verified_findings": {
    "quotes": 4,
    "all_pass": true,
    "source_diversity": 4
  },

  "per_claim": [
    {
      "claim_id": "c001",
      "claim_text": "Remote work increases productivity by up to 9%",
      "status": "SUPPORTED",
      "confidence": 0.92,
      "source_url": "https://...",
      "evidence_snippet": "...productivity may rise by up to 9%..."
    }
  ],

  "warnings": ["Claim c012 has low confidence (0.45)"]
}
```

---

## Design Decisions (from feedback review)

| Feedback | Decision | Rationale |
|----------|----------|-----------|
| Canonical store contradiction | **Defer to V2** | Current state.source_store works for MVP, Studio tested fine |
| Eval separate from report gen | **AGREE** | Keep as separate node/CLI, not inside final_report_generation() |
| Triage before verification | **Partial** | Prioritize high-risk claims, but verify all for V1 (max 30) |
| Hard retrieval contract | **AGREE** | Already planned: embedding → passages → structured verify |
| I9: Eval non-authoritative | **Modified** | Eval can inform fixes, but requires human checkpoint |

**New Invariant:**
> **I9: Eval-driven changes require human checkpoint.** Eval can identify weak spots and propose fixes, but auto-applying changes to prompts/pipeline requires human approval before commit.

---

## Implementation Steps

### Step 1: Create Evaluation Module

**File:** `src/open_deep_research/evaluation.py`

**Architecture:** Eval is a SEPARATE node/CLI, NOT embedded in report generation.

**Phase 1 - Claim Extraction:**
```python
async def extract_atomic_claims(report: str, llm) -> list[dict]:
    """Extract atomic, verifiable claims from report.

    Prioritize (for triage):
    - Claims with numbers/dates/percentages
    - Uncited factual claims
    - Claims with proper nouns

    Returns: [{"claim_id": "c001", "claim_text": "RAG reduces...", "citation": "[9]", "priority": "high"}]
    """
```

**Phase 2 - Claim Verification (hard retrieval contract):**
```python
async def verify_claim(claim: dict, sources: list, llm) -> dict:
    """Verify single claim against sources.

    Contract:
    1. Retrieve top-k passages via embedding similarity (reuse verification.py)
    2. Pass ONLY those passages to verification prompt
    3. Require structured output

    Returns: {
        "claim_id": "c001",
        "status": "TRUE" | "FALSE" | "UNVERIFIABLE",
        "confidence": 0.92,
        "evidence_snippet": "...",
        "source_url": "..."
    }
    """
```

**Phase 3 - Aggregate Metrics:**
```python
def aggregate_metrics(claim_results: list) -> dict:
    """Aggregate claim-level results into report metrics."""
    return {
        "total_claims": len(claim_results),
        "true_claims": count where status == "TRUE",
        "false_claims": count where status == "FALSE",
        "unverifiable_claims": count where status == "UNVERIFIABLE",
        "hallucination_rate": false / total,  # Target: <2%
        "grounding_rate": true / total,       # Target: >85%
    }
```

**Main entry point:**
```python
async def evaluate_report(state: AgentState, config: EvalConfig) -> EvalResult:
    """Full evaluation pipeline."""
    # 1. Extract atomic claims
    claims = await extract_atomic_claims(state["final_report"], llm)

    # 2. Verify each claim
    results = [await verify_claim(c, state["source_store"], llm) for c in claims]

    # 3. Check citations exist
    citation_check = check_citations(state["final_report"], state["source_store"])

    # 4. Check Verified Findings integrity
    vf_check = check_verified_findings(state)

    # 5. Aggregate
    return EvalResult(
        claims=aggregate_metrics(results),
        citations=citation_check,
        verified_findings=vf_check,
        per_claim=results
    )
```

Reuses existing:
- `verification.py:extract_claims()` - LLM-based claim extraction
- `verification.py:find_relevant_passages()` - embedding-based source matching
- `verification.py:verify_single_claim()` - LLM verification

### Step 2: Add Configuration Flags

**File:** `src/open_deep_research/configuration.py`

```python
run_evaluation: bool = Field(default=False)
verification_model: str = Field(default="openai:gpt-4.1-mini")
```

### Step 3: Optional Pipeline Integration

**File:** `src/open_deep_research/nodes/report.py`

Add optional eval at end of `final_report_generation()`:
```python
if configurable.run_evaluation:
    eval_result = await evaluate_report(state)
    print(f"[EVAL] {eval_result.support_rate:.0%} claims supported")
    return {"final_report": report, "evaluation_result": eval_result}
```

### Step 4: Standalone Eval Script

**File:** `scripts/run_eval.py`

```bash
# Run eval on saved state
python scripts/run_eval.py --state run_state.json --output eval.json

# Run eval from LangSmith thread
python scripts/run_eval.py --thread-id abc123 --output eval.json
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/open_deep_research/evaluation.py` | CREATE |
| `src/open_deep_research/configuration.py` | ADD 2 fields |
| `src/open_deep_research/nodes/report.py` | ADD optional eval call |
| `scripts/run_eval.py` | CREATE |

---

## Success Criteria

- [x] `evaluation.py` with `evaluate_report()` function
- [x] Tri-label claim verification (TRUE/FALSE/UNVERIFIABLE)
- [x] Structured JSON output with claims/citations/VF metrics
- [x] Standalone script `run_eval.py`
- [x] Optional integration via `run_evaluation: true` config flag
- [x] Test on research reports

---

## Implementation Status (Jan 5, 2026)

### Completed

**Files created/modified:**
- `src/open_deep_research/evaluation.py` - Citation-first evaluation framework
- `src/open_deep_research/configuration.py` - Added `run_evaluation` and `evaluation_model` config flags
- `scripts/run_eval.py` - Standalone evaluation script

**Key design: CITATION-FIRST APPROACH**

Instead of expensive embedding search for all claims, we:
1. Parse the Sources section to get `citation_num -> URL` mapping
2. For claims WITH citations `[N]`, verify against the cited source directly
3. For UNCITED claims, flag as high-risk and use embedding fallback
4. Parallelize verification in batches of 5

This is faster, cheaper, and more accurate than blind embedding search.

**Features:**
- `--dry-run` flag shows cost estimate before running
- Parallel verification (batch size configurable)
- Cost tracking (~$0.15-0.20 per eval with gpt-4.1-mini)
- Structured JSON output with per-claim details

### Test Results (AI Safety Report)

**Final results after bug fixes:**

```
Report: 19884 chars | Sources: 141 | Claims: 30

CLAIMS:
  TRUE: 19 (63% grounding)
  FALSE: 10 (33% hallucination)
  UNVERIFIABLE: 1
  UNCITED: 0

QUALITY TARGETS:
  Hallucination <2%: 33% [FAIL]
  Grounding >85%: 63% [FAIL]
  Citation >90%: 63% [FAIL]

Cost: ~$0.16
```

### Bug Fixes Applied

1. **Citation extraction** - Now finds citations by matching claim keywords to paragraphs
2. **Citation parsing** - Fixed regex to handle titles with colons (e.g., "From Labs to Policy: IMDA's Journey")
3. **Sources section exclusion** - Prevents false matches in Sources section

### Analysis of Results

The 33% hallucination rate reveals significant issues:

**Example FALSE claims:**
- c005: Claims about "AEF-1 standardization" not found in cited sources
- c007: Claims about "agent performance vs human benchmarks" not supported
- c015: Claims about "attribution graphs, confession mechanisms" not in sources

**Root causes:**
1. **Synthesis beyond sources** - Report synthesizes/interprets information not explicitly stated
2. **Citation misattribution** - Some claims cite wrong source numbers
3. **Incomplete source content** - source_store may have truncated content

### Actionable Next Steps

1. **Improve report generation prompts** - Enforce stricter grounding to cited sources ✅ DONE
2. **Citation verification during generation** - Check citations before including
3. **Investigate source content quality** - Ensure full content in source_store ✅ DONE (see below)

---

## Prompt Improvements (Jan 5, 2026)

Updated `src/open_deep_research/prompts.py` with stronger grounding rules:

### Changes to `compress_research_system_prompt`:
- Added **VERIFICATION CHECKLIST** that must be applied to every claim
- Added explicit anti-synthesis rule: "DO NOT synthesize information from multiple sources"
- Added "WHEN IN DOUBT, LEAVE IT OUT" principle
- Added requirement: "The cited source MUST actually contain the claimed information"

### Changes to `final_report_generation_prompt`:
- Added **BEFORE WRITING ANY CLAIM, ASK YOURSELF** checklist
- Strengthened anti-synthesis rule
- Added: "DO NOT assign a citation to a claim unless that source actually contains that information"
- Added: "It's better to have a shorter, accurate report than a longer, unverifiable one"

---

## Source Truncation Issue (Jan 5, 2026)

**Problem**: 20/141 sources (14%) have <300 characters - severely truncated.

**Examples of truncated sources:**
- `ari.us/policy-bytes/...` - 223 chars (should be ~3000+)
- `euronews.com/...` - 219 chars
- Various government docs - ~220-245 chars

**Impact**: Claims citing these sources cannot be verified, leading to FALSE ratings.

**Root cause**: Likely in Tavily search results or content summarization step.

**Next step**: Investigate `tavily_search` and `summarize_webpage_prompt` to ensure full content capture

---

## Quality Targets (from research)

| Metric | Target | Source |
|--------|--------|--------|
| Hallucination rate | <2% | Industry standard [Report 1] |
| Grounding rate | >85% | FACTS benchmark [Report 1] |
| Citation accuracy | >90% | Report claims match sources |

---

## Cost Estimate

Using `gpt-4.1-mini`: ~30 claims × ~$0.01 = **~$0.30/eval**

---

## Future: Self-Improvement Loop (Phase 4)

Based on Report 2 (ADAS pattern) and Report 3 (Agent SDK pattern):

```
┌─────────────────────────────────────────────────────────────────┐
│ SELF-IMPROVEMENT LOOP (Future)                                  │
│                                                                 │
│ 1. Run eval on N reports → aggregate metrics                    │
│ 2. Identify weak spots:                                         │
│    - High hallucination claims → trace to prompts               │
│    - Low grounding → check research depth                       │
│    - Citation misses → check source storage                     │
│ 3. Auto-propose fixes:                                          │
│    - Prompt tweaks for report generation                        │
│    - Config changes (more iterations, stricter filtering)       │
│ 4. Human review checkpoint                                      │
│ 5. Apply fixes → re-run eval → compare                          │
└─────────────────────────────────────────────────────────────────┘
```

This creates the "benchmark-driven iteration" pattern from the research.

---

## Future: RALPH-Style Autonomous Loop

Reference: https://github.com/frankbria/ralph-claude-code

**RALPH Pattern for Deep Research:**
```
┌──────────────────────────────────────────────────────────────┐
│ AUTONOMOUS IMPROVEMENT LOOP                                  │
│                                                              │
│ PROMPT.md: "Improve deep research until hallucination <2%"   │
│                                                              │
│ Loop:                                                        │
│   1. Run eval on N reports                                   │
│   2. Check: hallucination_rate < 2%? → EXIT                  │
│   3. Identify worst claims → trace to prompts/code           │
│   4. Apply fix (prompt tweak, config change)                 │
│   5. Re-run eval → compare metrics                           │
│   6. If improved → commit, else → rollback                   │
│   7. Repeat until targets met OR iteration limit             │
│                                                              │
│ Safety: Rate limits, human checkpoints, rollback capability  │
└──────────────────────────────────────────────────────────────┘
```

**Meta: Claude Code improving itself:**
```
# @fix_plan.md for this session
- [ ] Implement evaluation.py
- [ ] Run eval on 3 test reports
- [ ] If hallucination > 2%, identify weak prompts
- [ ] Fix prompts, re-run eval
- [ ] Commit when targets met
```

This allows Claude Code to autonomously improve the deep research pipeline
until quality targets are achieved, with built-in safety limits.

---
---

# PART A: Phase 1 Implementation Plan ✅ COMPLETE

## Goal
Get the basics working reliably: HITL brief review, fix quote quality, better logging.

## Scope
- Set `review_mode: "brief"` as default
- Fix garbage quotes (markdown stripping)
- Add structured console logging
- Run 3-4 test queries to validate

## Files to Modify

| File | Change |
|------|--------|
| `src/open_deep_research/configuration.py` | Set `review_mode: "brief"` default |
| `src/open_deep_research/logic/sanitize.py` | Add markdown stripping |
| `src/open_deep_research/nodes/*.py` | Add structured logging |
| `docs/DEEP_RESEARCH_V1.md` | CREATE - move north star doc here |
| `docs/archive/` | CREATE - archive old docs |

---

## Step 1: Documentation Migration

### 1.1 Create `docs/DEEP_RESEARCH_V1.md`
Copy Part B of this plan (North Star Reference) to permanent location.

### 1.2 Archive old docs
```
docs/archive/
├── DEEP_RESEARCH_TRUST_ARCH.md  (move from docs/)
└── invariants_v0.md              (move from invariants/invariants.md)
```

### 1.3 Update README.md
Point to new `docs/DEEP_RESEARCH_V1.md` for architecture reference.

---

## Step 2: Enable HITL Brief Review

### 2.1 Change default in `configuration.py`

**File:** `src/open_deep_research/configuration.py`

```python
# Change from:
review_mode: str = "none"

# To:
review_mode: str = "brief"
```

### 2.2 Verify interrupt formatting

Check that `validate_brief()` interrupt message renders cleanly:
- Brief text visible
- Council feedback visible
- Options clear (approve / edit)

### 2.3 Test flow
1. Run query
2. Confirm interrupt appears with brief
3. Type "approve" → proceeds
4. Type edited brief → uses edit

---

## Step 3: Fix Quote Extraction Quality

### 3.1 Problem
Quotes contain markdown artifacts like `### 7.` or `**Pro tip**:`

### 3.2 Solution: Add markdown stripping to `sanitize_for_quotes()`

**File:** `src/open_deep_research/logic/sanitize.py`

Add to `sanitize_for_quotes()`:
```python
def sanitize_for_quotes(text: str) -> str:
    """Sanitize text for quote extraction."""

    # ... existing HTML stripping ...

    # NEW: Strip markdown formatting
    # Headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # *italic*
    text = re.sub(r'__(.+?)__', r'\1', text)       # __bold__
    text = re.sub(r'_(.+?)_', r'\1', text)         # _italic_

    # List markers at start of line
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Links: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # ... rest of existing logic ...

    return text
```

### 3.3 Also check `chunk_by_sentences()` in `document_processing.py`
May need similar cleanup for spacy-based extraction path.

---

## Step 4: Add Structured Console Logging

### 4.1 Logging format

```
[STAGE] Message (key metrics)
```

### 4.2 Add to each node

**`nodes/brief.py` - `write_research_brief()`:**
```python
print(f"[BRIEF] Generating research plan...")
# ... after context gathering ...
if context_block:
    print(f"[BRIEF] Context: {len(brief_context.sources_used)} sources, entities: {brief_context.key_entities[:3]}")
# ... after generation ...
print(f"[BRIEF] ✓ Brief generated ({len(response.research_brief)} chars)")
```

**`nodes/brief.py` - `validate_brief()`:**
```python
if configurable.use_council:
    print(f"[COUNCIL] Voting on brief...")
    # ... after vote ...
    print(f"[COUNCIL] Result: {verdict.decision.upper()} ({verdict.consensus_score:.0%} consensus)")
```

**`nodes/supervisor.py` - `supervisor_tools()`:**
```python
print(f"[RESEARCH] Iteration {research_iterations}: {len(conduct_research_calls)} research tasks")
# ... after each researcher completes ...
print(f"[RESEARCH] Researcher: {len(sources)} sources, {len(compressed)} chars compressed")
```

**`nodes/extract.py` - `extract_evidence()`:**
```python
print(f"[EXTRACT] Processing {len(sources)} sources...")
# ... after extraction ...
print(f"[EXTRACT] ✓ {len(all_snippets)} candidates from {unique_sources} sources")
```

**`nodes/verify.py` - `verify_evidence()`:**
```python
print(f"[VERIFY] Verifying {len(snippets)} quotes...")
# ... after verification ...
pass_count = len([s for s in snippets if s["status"] == "PASS"])
print(f"[VERIFY] ✓ {pass_count}/{len(snippets)} PASS ({pass_count/len(snippets)*100:.0f}%)")
print(f"[VERIFY] Source diversity: {unique_pass_sources} sources with PASS quotes")
```

**`nodes/report.py` - `final_report_generation()`:**
```python
print(f"[REPORT] Generating Verified Findings...")
print(f"[REPORT] Selected {num_quotes} quotes from {num_sources} sources")
print(f"[REPORT] Generating main report...")
print(f"[REPORT] ✓ Complete ({len(report)} chars, {citation_count} citations)")
```

---

## Step 5: Validate with Test Queries

### 5.1 Test queries (run in order)

1. **General**: "What are the main challenges and best practices for remote work in 2025?"
2. **AI-specific**: "What are effective methods for managing long context in conversational AI systems?"
3. **Tech**: "What are the latest developments in edge computing for IoT applications?"
4. **Comparative**: "Compare the approaches of OpenAI, Anthropic, and Google to AI safety"

### 5.2 For each query, check:

| Metric | Target | How to Check |
|--------|--------|--------------|
| Brief review interrupt | Shows cleanly | Visual |
| Can approve/edit brief | Works | Try both |
| No markdown in quotes | Clean text | Read Verified Findings |
| Logging shows progress | All stages logged | Console output |
| Source diversity | ≥3 sources in VF | Count in report |
| Report not thin | ≥8000 chars | Check length |

### 5.3 Document issues found
Create `docs/phase1_test_results.md` with:
- Query run
- Issues found
- Metrics observed

---

## Step 6: Commit & Document

### 6.1 Commit changes
```bash
git add -A
git commit -m "Phase 1: Enable HITL, fix quote extraction, add logging"
```

### 6.2 Update README if needed
Note that `review_mode: "brief"` is now default.

---

## Phase 1 Success Criteria

- [ ] `review_mode: "brief"` is default
- [ ] Brief interrupt shows cleanly, approve/edit works
- [ ] No markdown artifacts in Verified Findings quotes
- [ ] Console shows structured progress through all stages
- [ ] 4 test queries run successfully
- [ ] Docs migrated to new structure

---

## What Phase 1 Does NOT Include

These are deferred to Phase 2+:
- CLI flags for review mode
- Evaluation framework / metrics JSON
- Source quality scoring
- Sub-question coverage tracking
- Auto-update loop
- Streaming to external UIs
- Preset profiles (fast/balanced/thorough)

---
---
---

# PART B: North Star Reference (Future Phases)

> **Purpose**: Comprehensive reference for architecture, invariants, and future roadmap.
> **Use**: Look up specific sections when addressing issues. Not an immediate implementation plan.

---

## Table of Contents

1. [System Invariants](#1-system-invariants)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Current Implementation Analysis](#3-current-implementation-analysis)
4. [Evaluation Framework](#4-evaluation-framework)
5. [Roadmap](#5-roadmap)
6. [Key Code Reference](#6-key-code-reference)

---

## 1. System Invariants

These are contracts the system must obey. Violations should fail loudly.

### I1: Verified Findings Integrity
**Contract**: Verified Findings section MUST only contain quotes with `status == PASS`.
**Enforcement**:
- `format_verified_quotes()` filters to PASS only
- `enforce_verified_section()` post-checks LLM didn't modify
- If no PASS quotes, show explicit "No quotes passed verification" message

### I2: Deterministic Verification
**Contract**: Quote verification MUST be deterministic (no LLM).
**Enforcement**:
- `verify_evidence` uses substring match OR Jaccard similarity > 0.8
- Same inputs = same PASS/FAIL results every time

### I3: Source Store as Truth
**Contract**: `state.source_store` is primary source storage, LangGraph Store is fallback.
**Enforcement**:
- Sources flow: `tavily_search` → `compress_research` → `supervisor_tools` → main state
- Extract/verify nodes read from `state.source_store` first, then Store

### I4: Fail-Safe Gating
**Contract**: If Store unavailable, Verified Findings MUST be explicitly disabled.
**Enforcement**:
- `check_store` runs at graph start
- Sets `verified_disabled = True` with reason
- Report shows disabled message, not empty section

### I5: CLI Scope (V0/V1)
**Contract**: CLI-first tool. No web server, no multi-user database.
**Enforcement**: SQLite for local persistence only.

### I6: No Secrets in Repo
**Contract**: No API keys committed.
**Enforcement**: `.env.example` with placeholders only.

### I7: No Sweeping Refactors
**Contract**: Changes scoped to current task only.
**Enforcement**: Code review, this document as reference.

### I8: Deterministic Writes
**Contract**: All file writes reproducible given same inputs.
**Enforcement**:
- `evidence_snippets` uses override semantics (not append)
- Capped at 100 snippets with round-robin diversity

---

## 2. Pipeline Architecture

### 2.1 High-Level Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│   1. check_store    │  ← Fail-fast: Is Store available?
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 2. clarify_with_user│  ← Optional: Ask clarifying questions
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────┐
│ 3. write_research_brief │  ← Generate detailed research plan
│    ├─ gather_brief_context (pre-search for recency)
│    └─ incorporate council feedback (if revising)
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────┐
│  4. validate_brief  │  ← Council vote + HITL checkpoint
│    ├─ Multi-model consensus (GPT-4.1 + Claude)
│    └─ Human review (if review_mode != "none")
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────┐
│  5. research_supervisor │  ← Main research loop (subgraph)
│    ├─ supervisor: Plan & delegate
│    └─ supervisor_tools: Execute ConductResearch
│        └─ researcher_subgraph (parallel)
│            ├─ researcher: Search & think
│            ├─ researcher_tools: Execute searches
│            └─ compress_research: Synthesize findings
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────┐
│ 6. validate_findings│  ← Council 2: Fact-check (advisory)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 7. extract_evidence │  ← S03: Mine quotes from sources
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  8. verify_evidence │  ← S04: PASS/FAIL each quote
└─────────┬───────────┘
          │
          ▼
┌───────────────────────────┐
│ 9. final_report_generation│  ← Two-call approach:
│    ├─ Call 1: Generate Verified Findings (selector-only)
│    └─ Call 2: Generate main report (with VF injected)
└─────────┬─────────────────┘
          │
          ▼
┌─────────────────────┐
│  10. verify_claims  │  ← Health check (optional)
└─────────┬───────────┘
          │
          ▼
      Final Report
```

### 2.2 Briefing Flow (Detailed)

This is the critical path for query → research plan.

```
User Query
    │
    ▼
clarify_with_user
    │
    ├─ need_clarification? → Yes → interrupt() → END (await user)
    │
    └─ No ──────────────────────────────────────────────────────┐
                                                                │
    ▼                                                           │
write_research_brief ◄──────────────────────────────────────────┘
    │
    ├─ Step 1: gather_brief_context (if enabled, first attempt only)
    │    ├─ Generate 3 exploratory search queries
    │    ├─ Run Tavily searches (general + news)
    │    └─ Extract: key_entities, recent_events, key_metrics, summary
    │
    ├─ Step 2: Include previous council feedback (if revising)
    │
    └─ Step 3: Generate ResearchQuestion with structured output
    │
    ▼
validate_brief
    │
    ├─ Step 1: Council Vote (if use_council: true)
    │    ├─ Query all models in parallel
    │    ├─ Each returns: decision, confidence, strengths, weaknesses
    │    ├─ Calculate weighted consensus
    │    └─ Synthesize feedback if not approved
    │
    ├─ Step 2: Human Review (if review_mode != "none")
    │    ├─ interrupt() with brief + council feedback
    │    └─ Human options:
    │         ├─ "approve" → proceed
    │         ├─ edited brief (50+ chars) → use human version
    │         └─ "ignore" → proceed without council suggestions
    │
    └─ Step 3: Route
         ├─ Approved → research_supervisor
         ├─ Human edit → research_supervisor (with edit)
         └─ Council "revise" (no human) → write_research_brief (loop)
```

### 2.3 Research Flow (Detailed)

```
research_supervisor (subgraph entry)
    │
    ▼
supervisor
    │
    ├─ Available tools: ConductResearch, ResearchComplete, think_tool
    │
    ├─ Prompt: lead_researcher_prompt
    │    ├─ "Think like a research manager with limited time"
    │    ├─ "Use think_tool before ConductResearch to plan"
    │    └─ "Stop when you can answer confidently"
    │
    └─ Response with tool_calls
    │
    ▼
supervisor_tools
    │
    ├─ think_tool calls → Record reflection, continue
    │
    ├─ ConductResearch calls → Spawn researcher_subgraph (parallel)
    │    │
    │    └─ researcher_subgraph ────────────────────────────────┐
    │         │                                                 │
    │         ▼                                                 │
    │    researcher                                             │
    │         │                                                 │
    │         ├─ Available tools: tavily_search, think_tool     │
    │         ├─ Prompt: research_system_prompt                 │
    │         │    ├─ "Start with broader searches"             │
    │         │    ├─ "After each search, pause and assess"     │
    │         │    └─ Hard limits: 5 search calls max           │
    │         │                                                 │
    │         └─ Response with tool_calls                       │
    │         │                                                 │
    │         ▼                                                 │
    │    researcher_tools                                       │
    │         │                                                 │
    │         ├─ Execute search tools in parallel               │
    │         ├─ Check iteration limits                         │
    │         └─ Route: continue or compress                    │
    │         │                                                 │
    │         ▼                                                 │
    │    compress_research                                      │
    │         │                                                 │
    │         ├─ Prompt: compress_research_system_prompt        │
    │         ├─ "Clean up findings, preserve ALL relevant info"│
    │         ├─ Extract sources from tool messages             │
    │         └─ Return: compressed_research, raw_notes,        │
    │                    source_store                           │
    │         │                                                 │
    │         └─────────────────────────────────────────────────┘
    │
    ├─ Aggregate results from all researchers
    │    ├─ compressed_research → tool messages
    │    ├─ raw_notes → state.raw_notes
    │    └─ source_store → state.source_store
    │
    ├─ ResearchComplete calls → Exit to validate_findings
    │
    └─ Exit conditions:
         ├─ research_iterations > max_researcher_iterations
         ├─ No tool calls made
         └─ ResearchComplete called
```

### 2.4 Verification Flow (Trust Pipeline)

```
extract_evidence (S03) - DETERMINISTIC
    │
    ├─ Input: state.source_store (or LangGraph Store fallback)
    │
    ├─ For each source:
    │    ├─ If extraction_method == "extract_api":
    │    │    └─ chunk_by_sentences (spacy-based, 10-100 words)
    │    └─ Else (search_raw):
    │         ├─ sanitize_for_quotes (regex HTML strip)
    │         └─ extract_paragraphs (15-60 words)
    │
    ├─ Create EvidenceSnippet for each passage:
    │    ├─ snippet_id: SHA256(source_id + quote)[:16]
    │    ├─ source_id, url, source_title, quote
    │    └─ status: "PENDING"
    │
    ├─ Limit to 100 snippets (round-robin across sources)
    │
    └─ Output: state.evidence_snippets
    │
    ▼
verify_evidence (S04) - DETERMINISTIC
    │
    ├─ For each snippet with status == "PENDING":
    │    │
    │    ├─ Get source content from source_store or Store
    │    │
    │    ├─ Check 1 (Strict): quote in source_content?
    │    │    └─ Yes → status = "PASS"
    │    │
    │    └─ Check 2 (Fuzzy): Jaccard similarity > 0.8?
    │         ├─ Tokenize quote and source windows
    │         ├─ Calculate: |intersection| / |union|
    │         └─ > 0.8 → status = "PASS", else "FAIL"
    │
    └─ Output: Updated state.evidence_snippets with PASS/FAIL
    │
    ▼
final_report_generation
    │
    ├─ Call 1: Generate Verified Findings (selector-only prompt)
    │    ├─ format_verified_quotes(): PASS snippets only, diverse sources
    │    ├─ SELECTOR_ONLY_PROMPT: "Pick 3-5, copy EXACTLY"
    │    └─ Output: verified_md
    │
    ├─ Call 2: Generate main report
    │    ├─ Inject verified_md as "IMMUTABLE - DO NOT EDIT" block
    │    └─ Output: generated_report
    │
    └─ Post-check: enforce_verified_section()
         ├─ Compare existing section to expected
         └─ Replace if LLM modified it
```

---

## 3. Current Implementation Analysis

### 3.1 What Works Well

| Component | Why It Works | Key Code |
|-----------|--------------|----------|
| **Two-call report generation** | Prevents LLM from hallucinating quotes | `report.py:129-164` |
| **Deterministic verification** | Reproducible, no LLM variance | `verify.py` |
| **Council as advisor** | Human has final say, council provides input | `validate_brief()` |
| **Brief context injection** | Grounds research in recent events | `gather_brief_context()` |
| **Round-robin diversity** | Ensures quotes from multiple sources | `format_verified_quotes()` |
| **Parallel researcher execution** | Faster research via asyncio.gather | `supervisor_tools()` |
| **Structured outputs** | Reliable parsing with Pydantic models | `ClarifyWithUser`, `ResearchQuestion` |
| **Progressive truncation** | Handles token limits gracefully | `final_report_generation()` retry loop |

### 3.2 What Needs Improvement

| Issue | Current State | Impact | Priority |
|-------|---------------|--------|----------|
| **"Vibes-based" debugging** | Only LangSmith traces, hard to see step-by-step | Can't diagnose issues systematically | HIGH |
| **HITL only via interrupt** | No CLI flags, Studio limited | User can't easily enable/disable | HIGH |
| **Research "enough" heuristic** | Fixed iteration counts | Over/under-researches | HIGH |
| **Source quality ranking** | All sources treated equally | Noise from low-quality sources | MEDIUM |
| **Quote extraction quality** | Regex + spacy, misses context | "Garbage" quotes with headers | MEDIUM |
| **No streaming progress** | Wait until end to see anything | Poor UX for long runs | MEDIUM |
| **No memory/context** | Each run starts fresh | Can't build on previous research | LOW |
| **Single export format** | PDF only (just added) | Limited output options | LOW |

### 3.3 Specific Problems Identified

#### Problem 1: Verified Findings Source Diversity
**Symptom**: All quotes from same 1-2 sources despite 82 PASS snippets
**Root Cause**: `format_verified_quotes()` round-robin works, but verification itself biases toward sources with cleaner content
**Evidence**: Log shows `[VERIFIED] Selected 6 quotes from 2 unique sources`

#### Problem 2: Quote Extraction Quality
**Symptom**: Quotes contain markdown headers like `### 7.`
**Root Cause**: `sanitize_for_quotes()` doesn't handle markdown in source content
**Fix Needed**: Add markdown stripping to sanitization pipeline

#### Problem 3: Research Depth Control
**Symptom**: Some queries get thin reports, others bloat
**Root Cause**: Fixed `max_researcher_iterations` doesn't adapt to query complexity
**Fix Needed**: LLM-based "enough research" assessment + sub-question coverage check

#### Problem 4: No Step-by-Step Evaluation
**Symptom**: "It feels thin" - can't diagnose where quality dropped
**Root Cause**: No structured metrics per stage
**Fix Needed**: Evaluation framework with stage-by-stage scoring

---

## 4. Evaluation Framework

### 4.1 Design Goals

1. **Per-stage metrics**: Know exactly where quality dropped
2. **Auto-updatable**: Claude can read eval results and propose fixes
3. **Regression testing**: Benchmark queries with expected outputs
4. **LangSmith integration**: Traces linked to structured metrics

### 4.2 Proposed Metrics

#### Stage: Brief Generation
| Metric | Description | Target |
|--------|-------------|--------|
| `brief_clarity_score` | LLM-assessed clarity (1-5) | ≥ 4 |
| `brief_specificity` | Contains specific entities/metrics? | Yes |
| `brief_actionability` | Can researcher start immediately? | Yes |
| `context_injection_used` | Was recent context gathered? | Yes (if enabled) |

#### Stage: Research
| Metric | Description | Target |
|--------|-------------|--------|
| `sources_collected` | Total unique sources | 20-50 |
| `source_quality_avg` | Avg domain reputation + recency + length | ≥ 0.6 |
| `sub_question_coverage` | % of brief sub-questions addressed | ≥ 80% |
| `research_iterations` | Loops before completion | ≤ 6 |
| `search_diversity` | Unique query patterns used | ≥ 5 |

#### Stage: Extraction & Verification
| Metric | Description | Target |
|--------|-------------|--------|
| `snippets_extracted` | Total candidate quotes | 50-150 |
| `snippets_pass_rate` | % that pass verification | ≥ 50% |
| `source_diversity_pass` | Unique sources with PASS quotes | ≥ 5 |
| `quote_quality_avg` | Avg length, no formatting artifacts | ≥ 0.7 |

#### Stage: Report
| Metric | Description | Target |
|--------|-------------|--------|
| `report_length_chars` | Total report length | 8000-20000 |
| `citation_density` | Citations per 500 words | ≥ 3 |
| `verified_findings_count` | Quotes in VF section | 3-5 |
| `section_coverage` | Has intro, findings, sources? | All required |
| `readability_score` | Flesch-Kincaid or similar | Grade 10-14 |

### 4.3 Evaluation Output Format

```json
{
  "run_id": "20260105_143022",
  "query": "What are effective methods for...",
  "total_duration_seconds": 245.3,
  "overall_score": 0.73,
  "stage_scores": {
    "brief": {
      "score": 0.85,
      "metrics": {
        "clarity_score": 4.2,
        "specificity": true,
        "actionability": true,
        "context_injection_used": true
      },
      "issues": []
    },
    "research": {
      "score": 0.68,
      "metrics": {
        "sources_collected": 42,
        "source_quality_avg": 0.54,
        "sub_question_coverage": 0.72,
        "research_iterations": 4
      },
      "issues": ["sub_question_coverage below target", "source_quality_avg below target"]
    },
    "verification": {
      "score": 0.72,
      "metrics": {
        "snippets_extracted": 100,
        "snippets_pass_rate": 0.82,
        "source_diversity_pass": 3,
        "quote_quality_avg": 0.65
      },
      "issues": ["source_diversity_pass below target", "quote_quality_avg below target"]
    },
    "report": {
      "score": 0.78,
      "metrics": {
        "report_length_chars": 14239,
        "citation_density": 4.2,
        "verified_findings_count": 3,
        "section_coverage": true
      },
      "issues": []
    }
  },
  "recommendations": [
    "Increase source quality filtering threshold",
    "Add markdown stripping to quote extraction",
    "Investigate why only 3 sources have PASS quotes"
  ]
}
```

### 4.4 Auto-Update Loop

For running on 50 queries and auto-updating code:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO-UPDATE LOOP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Run benchmark suite (50 queries)                            │
│     └─ Collect evaluation JSON for each                         │
│                                                                  │
│  2. Aggregate results                                           │
│     ├─ Overall pass rate                                        │
│     ├─ Most common issues                                       │
│     └─ Stage-by-stage score distributions                       │
│                                                                  │
│  3. Generate improvement recommendations                        │
│     ├─ Identify lowest-scoring stages                           │
│     ├─ Pattern-match issues to known fixes                      │
│     └─ Prioritize by impact × ease                              │
│                                                                  │
│  4. Apply fixes (Claude Code)                                   │
│     ├─ Update configuration thresholds                          │
│     ├─ Modify extraction/sanitization logic                     │
│     └─ Adjust prompts                                           │
│                                                                  │
│  5. Re-run benchmark, compare                                   │
│     ├─ Score improved? → Commit                                 │
│     └─ Score regressed? → Rollback, try different fix           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Roadmap

### 5.1 Phase 1: Accessibility & Debugging (HIGH PRIORITY)

#### 5.1.1 CLI Commands for HITL
**Goal**: User can enable/disable review modes from CLI

```bash
# Current (implicit via config)
python scripts/test_e2e_quick.py "query"

# Proposed
python -m open_deep_research run "query" --review-mode brief
python -m open_deep_research run "query" --review-mode full
python -m open_deep_research run "query" --review-mode none
python -m open_deep_research run "query" --no-council
```

**Implementation**:
- Add `cli.py` with argparse/click
- Map CLI flags to Configuration overrides
- Entry point in `pyproject.toml`

#### 5.1.2 LangGraph Studio HITL
**Goal**: Review brief and report in Studio UI

**Current State**: interrupt() works but UX is rough
**Fix**: Ensure interrupt messages are well-formatted for Studio rendering

#### 5.1.3 Evaluation Framework
**Goal**: Structured metrics per run, readable by Claude

**Implementation**:
- Add `evaluation.py` module
- Generate `run_evaluation.json` alongside `run_report.md`
- Add `--eval` flag to CLI to enable detailed metrics
- Integrate with LangSmith tags for trace linking

### 5.2 Phase 2: Research Quality (HIGH PRIORITY)

#### 5.2.1 "Enough Research" Assessment
**Goal**: Model knows when to stop researching

**Current**: Fixed iteration count
**Proposed**: Multi-signal assessment
```python
def assess_research_completeness(state, config) -> dict:
    """
    Returns:
        {
            "is_complete": bool,
            "confidence": float,
            "coverage": {
                "sub_questions_addressed": ["q1", "q2"],
                "sub_questions_missing": ["q3"],
                "coverage_pct": 0.67
            },
            "quality": {
                "source_count": 25,
                "high_quality_sources": 18,
                "unique_perspectives": 4
            },
            "recommendation": "continue" | "complete"
        }
    """
```

**Signals**:
1. Sub-question coverage (parse brief, check notes)
2. Source quality score (domain, recency, length)
3. LLM self-assessment ("Do I have enough to answer comprehensively?")
4. Diminishing returns (last N searches yielded similar info)

#### 5.2.2 Source Quality Ranking
**Goal**: Prioritize high-quality sources

**Scoring factors**:
- Domain reputation (whitelist of authoritative domains)
- Recency (exponential decay, half-life = 90 days)
- Content length (penalize thin content < 500 chars)
- Citation by other sources (if detectable)

**Implementation**:
```python
def score_source(source: dict) -> float:
    domain_score = get_domain_reputation(source["url"])  # 0-1
    recency_score = calculate_recency(source["timestamp"])  # 0-1
    length_score = min(len(source["content"]) / 2000, 1.0)  # 0-1

    return (domain_score * 0.4 + recency_score * 0.3 + length_score * 0.3)
```

#### 5.2.3 Quote Extraction Cleanup
**Goal**: No formatting artifacts in quotes

**Fixes needed**:
- Strip markdown headers (`#`, `##`, etc.)
- Strip list markers (`-`, `*`, `1.`)
- Normalize whitespace
- Filter quotes that are clearly navigation/boilerplate

### 5.3 Phase 3: User Experience (MEDIUM PRIORITY)

#### 5.3.1 Streaming Output
**Goal**: Show progress as research happens

**Implementation options**:
1. **Console streaming**: Print stage entry/exit, source counts
2. **Progress callback**: Hook for external UIs
3. **LangGraph streaming**: Use built-in streaming support

**Minimum viable**:
```
[BRIEF] Generating research plan...
[BRIEF] ✓ Brief generated (context from 12 sources)
[RESEARCH] Starting research loop...
[RESEARCH] Researcher 1: Searching "AI context management"...
[RESEARCH] Researcher 1: Found 8 sources
[RESEARCH] Researcher 2: Searching "sales call AI memory"...
[RESEARCH] ✓ Research complete (42 sources, 3 iterations)
[VERIFY] Extracting quotes...
[VERIFY] ✓ 82/100 quotes verified
[REPORT] Generating final report...
[REPORT] ✓ Report complete (14,239 chars)
```

#### 5.3.2 Preset Profiles
**Goal**: Quick settings without config diving

```python
PRESETS = {
    "fast": {
        "max_researcher_iterations": 2,
        "max_react_tool_calls": 3,
        "use_council": False,
        "use_findings_council": False,
    },
    "balanced": {
        "max_researcher_iterations": 4,
        "max_react_tool_calls": 6,
        "use_council": True,
        "council_max_revisions": 1,
    },
    "thorough": {
        "max_researcher_iterations": 8,
        "max_react_tool_calls": 10,
        "use_council": True,
        "council_max_revisions": 3,
        "use_claim_verification": True,
    }
}
```

**Usage**:
```bash
python -m open_deep_research run "query" --preset fast
python -m open_deep_research run "query" --preset thorough
```

### 5.4 Phase 4: Advanced Features (LOW PRIORITY)

#### 5.4.1 Memory/Context
**Goal**: Build on previous research

**Options**:
1. **Session memory**: Store sources/findings in SQLite, query by topic
2. **Vector store**: Embed past research, retrieve relevant context
3. **Explicit links**: User can reference previous runs by ID

#### 5.4.2 Multi-Format Export
**Goal**: PDF, DOCX, HTML, Markdown

**Current**: PDF (WeasyPrint), Markdown
**Add**:
- DOCX via python-docx
- HTML (already have, just expose)

---

## 6. Key Code Reference

### 6.1 Briefing Flow

#### `clarify_with_user` (clarify.py or graph.py)
```python
async def clarify_with_user(state: AgentState, config: RunnableConfig):
    """Ask clarifying questions if needed."""
    configurable = Configuration.from_runnable_config(config)

    # Skip if disabled
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")

    # Structured output for clarification decision
    response = await clarification_model.ainvoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages),
            date=get_today_str()
        ))
    ])

    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})
```

#### `gather_brief_context` (utils.py)
```python
async def gather_brief_context(user_messages, config, max_queries=3, max_results=5, days=90):
    """Pre-search for recent context before brief generation.

    Steps:
    1. Generate exploratory search queries from user request
    2. Run Tavily searches (general + news)
    3. Extract: key_entities, recent_events, key_metrics, summary

    Returns: BriefContext dataclass
    """
    # Step 1: Generate queries
    query_response = await query_model.ainvoke([
        HumanMessage(content=generate_context_queries_prompt.format(
            num_queries=max_queries,
            user_messages=user_messages,
            date=get_today_str()
        ))
    ])
    queries = json.loads(query_response.content.strip())

    # Step 2: Execute searches
    general_results = await tavily_search_async(queries, max_results, topic="general", days=days)
    news_results = await tavily_search_async(queries[:2], max_results=3, topic="news", days=min(days, 30))

    # Step 3: Extract structured context
    extract_response = await extract_model.ainvoke([
        HumanMessage(content=extract_brief_context_prompt.format(
            days=days,
            search_results=formatted_results,
            user_query=user_messages
        ))
    ])

    return BriefContext(
        key_entities=extracted.get("key_entities", []),
        recent_events=extracted.get("recent_events", []),
        key_metrics=extracted.get("key_metrics", []),
        context_summary=extracted.get("context_summary", ""),
        sources_used=sources_used
    )
```

#### `council_vote_on_brief` (council.py)
```python
async def council_vote_on_brief(brief: str, config: CouncilConfig, runnable_config):
    """Multi-model consensus on research brief.

    Steps:
    1. Query all council models in parallel
    2. Each returns: decision, confidence, strengths, weaknesses
    3. Calculate weighted consensus
    4. Synthesize feedback if not approved
    """
    # Parallel voting
    vote_tasks = [get_single_vote(model, brief, runnable_config) for model in config.models]
    votes = await asyncio.gather(*vote_tasks)

    # Weighted consensus
    weighted_votes = {"approve": 0.0, "revise": 0.0, "reject": 0.0}
    for vote in votes:
        weight = vote.confidence if vote.confidence >= 0.6 else vote.confidence * 0.5
        weighted_votes[vote.decision] += weight

    # Decision logic
    if weighted_votes["approve"] >= config.min_consensus_for_approve:
        decision = "approve"
    else:
        decision = "revise"

    verdict = CouncilVerdict(decision=decision, consensus_score=..., votes=votes)

    # Synthesize feedback
    if verdict.decision != "approve":
        verdict.synthesized_feedback = await synthesize_feedback(votes, synthesis_model)

    return verdict
```

### 6.2 Research Flow

#### `supervisor` (supervisor.py or graph.py)
```python
async def supervisor(state: SupervisorState, config: RunnableConfig):
    """Lead researcher that plans and delegates."""
    # Available tools
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # System prompt emphasizes:
    # - "Think like a research manager with limited time"
    # - "Use think_tool before ConductResearch to plan"
    # - "Stop when you can answer confidently"

    response = await research_model.bind_tools(lead_researcher_tools).ainvoke(supervisor_messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )
```

#### `compress_research` (researcher.py or graph.py)
```python
async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Synthesize research findings, extract sources."""
    # Key prompt rules:
    # - "Clean up findings, preserve ALL relevant information"
    # - "DO NOT invent, fabricate, or extrapolate any facts"
    # - "Every claim MUST have a citation"

    response = await synthesizer_model.ainvoke([
        SystemMessage(content=compress_research_system_prompt),
        *researcher_messages,
        HumanMessage(content=compress_research_simple_human_message)
    ])

    # Extract sources from tool messages (critical for verification)
    # Pattern: --- SOURCE N: {title} ---\nURL: {url}\n\nSUMMARY:\n{content}

    return {
        "compressed_research": response.content,
        "raw_notes": [raw_notes_content],
        "source_store": extracted_sources  # For verification pipeline
    }
```

### 6.3 Verification Flow

#### `extract_evidence` (extract.py)
```python
async def extract_evidence(state: AgentState, config: RunnableConfig):
    """DETERMINISTIC: Extract candidate quotes from sources."""
    if state.get("verified_disabled", False):
        return {"evidence_snippets": []}

    sources = state.get("source_store", []) or await get_stored_sources(config)
    all_snippets = []

    for source in sources:
        if source.get("extraction_method") == "extract_api":
            # Clean content: spacy-based chunking
            passages = chunk_by_sentences(content, min_words=10, max_words=100)
        else:
            # Raw HTML: regex sanitization + paragraph extraction
            clean_text = sanitize_for_quotes(content)
            passages = extract_paragraphs(clean_text, min_words=15, max_words=60)

        for passage in passages:
            snippet = {
                "snippet_id": generate_snippet_id(source_url, passage),
                "source_id": source_url,
                "url": source_url,
                "source_title": source_title,
                "quote": passage,
                "status": "PENDING"
            }
            all_snippets.append(snippet)

    # Limit with round-robin diversity
    if len(all_snippets) > 100:
        all_snippets = round_robin_select(all_snippets, max_snippets=100)

    return {"evidence_snippets": all_snippets}
```

#### `verify_evidence` (verify.py)
```python
async def verify_evidence(state: AgentState, config: RunnableConfig):
    """DETERMINISTIC: Verify quotes exist in sources."""
    snippets = state.get("evidence_snippets", [])
    sources = state.get("source_store", [])

    for snippet in snippets:
        if snippet["status"] != "PENDING":
            continue

        source_content = get_source_content(snippet["source_id"], sources)
        quote = snippet["quote"]

        # Check 1: Strict substring match
        if quote in source_content:
            snippet["status"] = "PASS"
            continue

        # Check 2: Fuzzy Jaccard similarity
        quote_tokens = set(tokenize(quote))
        for window in sliding_windows(source_content, len(quote) * 2):
            window_tokens = set(tokenize(window))
            jaccard = len(quote_tokens & window_tokens) / len(quote_tokens | window_tokens)
            if jaccard > 0.8:
                snippet["status"] = "PASS"
                break
        else:
            snippet["status"] = "FAIL"

    return {"evidence_snippets": snippets}
```

### 6.4 Report Generation

#### `format_verified_quotes` (report.py)
```python
def format_verified_quotes(snippets, max_quotes=20, max_per_source=3):
    """Format PASS snippets with source diversity."""
    verified = [s for s in snippets if s.get("status") == "PASS"]

    # Group by source for diversity
    by_source = defaultdict(list)
    for snippet in verified:
        by_source[snippet["url"]].append(snippet)

    # Round-robin selection
    diverse_quotes = []
    source_urls = list(by_source.keys())
    round_num = 0

    while len(diverse_quotes) < max_quotes and round_num < max_per_source:
        for url in source_urls:
            if round_num < len(by_source[url]) and len(diverse_quotes) < max_quotes:
                diverse_quotes.append(by_source[url][round_num])
        round_num += 1

    # Format for prompt
    return "\n".join([
        f'{i}. "{s["quote"]}" - [{s["source_title"]}]({s["url"]})'
        for i, s in enumerate(diverse_quotes, 1)
    ])
```

#### `enforce_verified_section` (report.py)
```python
def enforce_verified_section(report: str, verified_md: str) -> str:
    """Post-check: Ensure LLM didn't modify Verified Findings."""
    pattern = r'(## Verified Findings\s*\n[\s\S]*?)(?=\n## |\Z)'

    match = re.search(pattern, report)
    if not match:
        # Section missing - append it
        return report + "\n\n" + verified_md

    existing = re.sub(r'\s+', ' ', match.group(1).strip())
    expected = re.sub(r'\s+', ' ', verified_md.strip())

    if existing != expected:
        # LLM modified it - replace
        report = re.sub(pattern, verified_md + "\n", report)

    return report
```

---

## 7. Migration Plan

### Step 1: Documentation Consolidation
- [ ] Create this document as `docs/DEEP_RESEARCH_V1.md`
- [ ] Archive `docs/DEEP_RESEARCH_TRUST_ARCH.md` to `docs/archive/`
- [ ] Archive `invariants/invariants.md` to `docs/archive/`
- [ ] Update README.md to point to new docs

### Step 2: CLI Implementation
- [ ] Create `src/open_deep_research/cli.py`
- [ ] Add entry point in `pyproject.toml`
- [ ] Implement: `run`, `--review-mode`, `--preset`, `--eval`

### Step 3: Evaluation Framework
- [ ] Create `src/open_deep_research/evaluation.py`
- [ ] Define metric collectors per stage
- [ ] Generate `run_evaluation.json` output
- [ ] Add `--eval` flag to CLI

### Step 4: Research Quality Improvements
- [ ] Implement `assess_research_completeness()`
- [ ] Add source quality scoring
- [ ] Fix quote extraction (markdown stripping)

### Step 5: Streaming & UX
- [ ] Add console progress output
- [ ] Implement preset profiles
- [ ] Test LangGraph Studio HITL

---

## Appendix A: Configuration Reference

```python
@dataclass
class Configuration:
    # Models
    research_model: str = "openai:gpt-4.1"
    summarization_model: str = "openai:gpt-4.1-mini"
    compression_model: str = "openai:gpt-4.1"
    final_report_model: str = "openai:gpt-4.1"

    # Research limits
    max_researcher_iterations: int = 6
    max_react_tool_calls: int = 10
    max_concurrent_research_units: int = 5

    # Council
    use_council: bool = True
    council_models: list = ["openai:gpt-4.1", "anthropic:claude-sonnet-4-5-20250929"]
    council_min_consensus: float = 0.7
    council_max_revisions: int = 3

    # HITL
    review_mode: str = "none"  # "none", "brief", "full"
    allow_clarification: bool = True

    # Brief context
    enable_brief_context: bool = True
    brief_context_max_queries: int = 3
    brief_context_days: int = 90

    # Verification
    use_claim_verification: bool = False
    use_tavily_extract: bool = True

    # Source filtering
    max_total_sources: int = 200
    min_source_content_length: int = 500

    # Test mode (reduces all limits)
    test_mode: bool = False
```

---

## Appendix B: Key Prompts

### Brief Generation Prompt
```
You will be given a set of messages that have been exchanged so far between yourself and the user.
Your job is to translate these messages into a more detailed and concrete research question.

Guidelines:
1. Maximize Specificity and Detail
2. Fill in Unstated But Necessary Dimensions as Open-Ended
3. Avoid Unwarranted Assumptions
4. Use the First Person
5. If specific sources should be prioritized, specify them
```

### Research System Prompt
```
You are a research assistant conducting research on the user's input topic.

Instructions:
1. Read the question carefully
2. Start with broader searches
3. After each search, pause and assess
4. Execute narrower searches as you gather information
5. Stop when you can answer confidently

Hard Limits:
- Simple queries: 2-3 search tool calls maximum
- Complex queries: up to 5 search tool calls maximum
- Always stop after 5 if you cannot find the right sources
```

### Compression Prompt
```
You are a research assistant that has conducted research on a topic.
Your job is to clean up the findings, but preserve all relevant statements.

CRITICAL GROUNDING RULES - PREVENT HALLUCINATION:
- DO NOT invent, fabricate, or extrapolate any facts
- Every single claim MUST have a citation
- If you are unsure whether something is in the search results, DO NOT include it
```

### Selector-Only Prompt (Verified Findings)
```
You are generating ONLY the "Verified Findings" section for a research report.

STRICT RULES:
1. Start with "## Verified Findings" heading
2. Create a bullet list with 3-5 of the most relevant quotes
3. DIVERSITY REQUIRED: Select quotes from DIFFERENT sources
4. Copy quotes EXACTLY - do not paraphrase or modify
5. Do not add any quotes not in the list above
```
