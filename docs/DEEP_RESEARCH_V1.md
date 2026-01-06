# Deep Research V1 - Architecture & Roadmap

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

### I3: Canonical Store
**Contract**: LangGraph Store is the single source of truth for document text (raw/extracted). AgentState stores only metadata + stable IDs.
**Enforcement**:
- Tools write content to Store keyed by `(thread_id, "raw")` or `(thread_id, "extracted")`
- Nodes read full content from Store, never from state
- State carries only: `source_id`, `url`, `title`, `timestamp`, `extraction_method`, `has_content` flag
- This prevents token explosion from duplicating full content in state

**Note**: Current implementation (V0) still uses `state.source_store` as primary with Store as fallback. Migration to canonical Store is a Phase 2 goal.

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

### 5.5 North Star: Autonomous Innovation Pipeline

**Vision**: Research runs continuously, generating MVPs of new ideas with minimal human intervention.

```
┌─────────────────────────────────────────────────────────────────┐
│              CONTINUOUS INNOVATION PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │   TRIGGER    │  Scheduled (daily) or event-driven            │
│  │  (cron/hook) │  "What's new in AI agents/tooling?"           │
│  └──────┬───────┘                                               │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │   RESEARCH   │  Deep Research runs autonomously              │
│  │    Agent     │  Gathers context, verifies sources            │
│  └──────┬───────┘                                               │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │   FILTER     │  HITL: "Worth building?" (30 sec decision)    │
│  │   + HITL     │  📱 Mobile alert for approval                 │
│  └──────┬───────┘                                               │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │     PRD      │  Generate PRD from research                   │
│  │  Generator   │  + Additional tool calls for context          │
│  └──────┬───────┘                                               │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │  PRD Review  │  HITL: "Scope right?" (2 min review)          │
│  │   + HITL     │  📱 Mobile alert for approval                 │
│  └──────┬───────┘                                               │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ REQUIREMENTS │  Technical requirements from PRD              │
│  │  Generator   │  + Tool calls for implementation details      │
│  └──────┬───────┘                                               │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ CLAUDE CODE  │  Builds MVP from requirements                 │
│  │   Builder    │  Runs autonomously                            │
│  └──────┬───────┘                                               │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │  AUTO-TEST   │  Run against test suite                       │
│  │              │  HITL only if tests fail                      │
│  └──────┬───────┘                                               │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │    OUTPUT    │  Working MVP + docs                           │
│  │              │  📱 Alert: "MVP ready for review"             │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features**:

1. **Scheduled Research**: Cron job triggers research on topics of interest
2. **Research → PRD Pipeline**: Research output feeds structured PRD generation
3. **Tool Calls at Each Stage**: PRD and Requirements agents can search/read for additional context
4. **Mobile HITL Alerts**: Push notifications for approval gates (Pushover/Twilio/Slack)
5. **Auto-Testing**: MVPs run against existing test suites before human review
6. **Minimal Human Time**: ~3-5 minutes of HITL per idea, system does hours of work

**Mobile Alert Implementation Options**:
- **Pushover**: Simple push notifications ($5 one-time)
- **Twilio SMS**: Text message alerts
- **Slack/Discord Webhooks**: Channel notifications with action buttons
- **iOS Shortcuts + Webhook**: Custom automation

**HITL Checkpoint Pattern**:
```python
async def hitl_checkpoint(stage: str, content: str, config: RunnableConfig):
    """Send mobile alert and wait for approval."""
    # Send notification
    await send_mobile_alert(
        title=f"[{stage}] Approval Needed",
        body=content[:200] + "...",
        actions=["approve", "reject", "edit"]
    )

    # Interrupt and wait for response
    response = interrupt(format_review_request(stage, content))
    return process_hitl_response(response)
```

**Why This Works**:
- Research grounds everything in real developments (no hallucinated ideas)
- Each stage has evaluation criteria before proceeding
- Human judgment only where it matters (is it worth building? is scope right?)
- Compounding value: each MVP adds to your toolkit

### 5.6 Ultimate North Star: Autonomous Idea-to-Investment Pipeline

**Vision**: The innovation pipeline becomes a general-purpose **idea evaluator** that can make build/buy/invest decisions autonomously.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IDEA-TO-INVESTMENT PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐                                                     │
│  │    RESEARCH    │  Continuous scanning for interesting developments   │
│  │     Agent      │  "What's emerging in [domain]?"                     │
│  └───────┬────────┘                                                     │
│          ▼                                                               │
│  ┌────────────────┐                                                     │
│  │   PMF/MARKET   │  Automated product-market fit analysis              │
│  │   ANALYSIS     │  • TAM/SAM/SOM estimation                           │
│  │                │  • Competitor landscape                              │
│  │                │  • Customer pain point validation                    │
│  │                │  • Timing analysis (why now?)                        │
│  └───────┬────────┘                                                     │
│          ▼                                                               │
│  ┌────────────────┐                                                     │
│  │   FEASIBILITY  │  Cost/complexity modeling                           │
│  │    ANALYSIS    │  • Technical complexity score                       │
│  │                │  • Build cost estimate                               │
│  │                │  • Time-to-MVP estimate                              │
│  │                │  • Moat/defensibility assessment                     │
│  └───────┬────────┘                                                     │
│          ▼                                                               │
│  ┌────────────────┐                                                     │
│  │    DECISION    │  Route based on analysis                            │
│  │     GATE       │                                                      │
│  └───────┬────────┘                                                     │
│          │                                                               │
│    ┌─────┴─────┬──────────────┬──────────────┐                          │
│    ▼           ▼              ▼              ▼                          │
│ ┌──────┐  ┌────────┐    ┌──────────┐   ┌───────────┐                   │
│ │ SKIP │  │ BUILD  │    │  START   │   │  INVEST   │                   │
│ │      │  │INTERNAL│    │ COMPANY  │   │   IN CO   │                   │
│ └──────┘  └────┬───┘    └────┬─────┘   └─────┬─────┘                   │
│                │             │               │                          │
│                ▼             ▼               ▼                          │
│         ┌──────────┐  ┌───────────┐  ┌─────────────┐                   │
│         │ MVP Flow │  │ Full PRD  │  │ Find Cos    │                   │
│         │(§5.5)    │  │ + Pitch   │  │ Building    │                   │
│         └────┬─────┘  │ Deck Gen  │  │ This        │                   │
│              │        └─────┬─────┘  └──────┬──────┘                   │
│              │              │               │                          │
│              ▼              ▼               ▼                          │
│         ┌──────────┐  ┌───────────┐  ┌─────────────┐                   │
│         │  Deploy  │  │ Validate  │  │ Build Proto │                   │
│         │  + Use   │  │ w/ Users  │  │ to Evaluate │                   │
│         └──────────┘  └───────────┘  │ Their Idea  │                   │
│                                      └──────┬──────┘                   │
│                                             │                          │
│                                             ▼                          │
│                                      ┌─────────────┐                   │
│                                      │ Investment  │                   │
│                                      │   Thesis    │                   │
│                                      │ Generation  │                   │
│                                      └─────────────┘                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Three Output Modes**:

| Mode | When | Output |
|------|------|--------|
| **Build Internal** | Small scope, personal utility, low TAM | MVP for your toolkit |
| **Start Company** | Large TAM, timing right, you have edge | Full PRD + pitch deck + validation plan |
| **Invest in Existing** | Good idea but others ahead | Investment thesis + prototype to validate their approach |

**PMF/Market Analysis Agent**:
```python
class MarketAnalysis(BaseModel):
    tam_estimate: str           # Total addressable market
    sam_estimate: str           # Serviceable addressable market
    competition: List[Competitor]
    timing_score: float         # Why now? (0-1)
    pain_point_severity: float  # How bad is the problem? (0-1)
    willingness_to_pay: str     # Evidence of WTP
    distribution_channels: List[str]
    risks: List[str]
    recommendation: Literal["skip", "build_internal", "start_company", "invest"]
```

**Feasibility Analysis Agent**:
```python
class FeasibilityAnalysis(BaseModel):
    technical_complexity: float  # 0-1 (0=trivial, 1=moonshot)
    estimated_build_cost: str    # "$X-Y range"
    time_to_mvp: str            # "X-Y weeks"
    required_skills: List[str]
    moat_assessment: str        # Network effects, data, etc.
    key_risks: List[str]
    go_no_go: bool
```

**Investment Thesis Generator** (for "Invest" path):
```python
class InvestmentThesis(BaseModel):
    company_name: str
    what_they_do: str
    why_now: str
    market_size: str
    competition: str
    team_assessment: str        # If available
    prototype_findings: str     # From building their idea ourselves
    key_risks: List[str]
    recommendation: Literal["strong_buy", "buy", "hold", "pass"]
    conviction_score: float     # 0-1
```

**The "Build to Evaluate" Pattern**:
For investment decisions, actually building a prototype of the target company's product provides:
- Ground truth on technical feasibility
- Understanding of implementation challenges they'll face
- Validation of whether the approach works
- Edge in evaluating their team's execution

**Use Cases**:

1. **Personal Productivity**: "What AI tools should I build for myself?"
   → Research → PMF (just you) → Build Internal

2. **Side Project Validation**: "Is this worth turning into a product?"
   → Research → PMF → Feasibility → Start Company (or not)

3. **Angel/VC Analysis**: "Should I invest in this AI agent startup?"
   → Research their space → Build prototype of their idea → Generate thesis

4. **Corporate Innovation**: "What should we build next quarter?"
   → Research trends → PMF for your market → Prioritized feature list

**Why This is Powerful**:
- Research prevents hallucinated opportunities (grounded in real developments)
- Building prototypes provides ground truth vs. just reading pitch decks
- Automated PMF analysis scales to evaluate many ideas quickly
- HITL at decision gates keeps human judgment on high-stakes choices
- Same infrastructure serves multiple use cases (build/start/invest)

---

## 6. Key Code Reference

### 6.1 Key Files

| File | Purpose |
|------|---------|
| `src/open_deep_research/graph.py` | Main LangGraph construction |
| `src/open_deep_research/state.py` | State definitions |
| `src/open_deep_research/configuration.py` | Configuration class |
| `src/open_deep_research/nodes/brief.py` | Brief generation & validation |
| `src/open_deep_research/nodes/supervisor.py` | Research coordination |
| `src/open_deep_research/nodes/researcher.py` | Individual research agents |
| `src/open_deep_research/nodes/extract.py` | Quote extraction (S03) |
| `src/open_deep_research/nodes/verify.py` | Quote verification (S04) |
| `src/open_deep_research/nodes/report.py` | Final report generation |
| `src/open_deep_research/logic/sanitize.py` | HTML/markdown sanitization |
| `src/open_deep_research/council.py` | Multi-model council voting |

### 6.2 Configuration Reference

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
    review_mode: str = "brief"  # "none", "brief", "full"
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

### 6.3 Key Prompts

#### Brief Generation
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

#### Research System
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

#### Selector-Only (Verified Findings)
```
You are generating ONLY the "Verified Findings" section for a research report.

STRICT RULES:
1. Start with "## Verified Findings" heading
2. Create a bullet list with 3-5 of the most relevant quotes
3. DIVERSITY REQUIRED: Select quotes from DIFFERENT sources
4. Copy quotes EXACTLY - do not paraphrase or modify
5. Do not add any quotes not in the list above
```
