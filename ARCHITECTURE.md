# Deep Research Architecture

This document provides a complete technical breakdown of how the Deep Research agent works, from user query to final verified report.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [State Management](#state-management)
6. [Configuration Options](#configuration-options)
7. [Anti-Hallucination System](#anti-hallucination-system)
8. [Key Files Reference](#key-files-reference)

---

## System Overview

Deep Research is a LangGraph-based research agent that:

1. Takes a user's research question
2. Generates a research brief (plan)
3. Delegates research to parallel sub-agents
4. Gathers sources from web searches (Tavily)
5. Compresses and synthesizes findings
6. Extracts and verifies evidence quotes
7. Generates a final report with verified citations

### High-Level Pipeline

```
User Query
    │
    ▼
┌─────────────────┐
│   check_store   │  ← S02: Checks if LangGraph Store is available
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  clarify_with_user  │  ← Optional: Ask clarifying questions
└─────────┬───────────┘
          │
          ▼
┌────────────────────────┐
│  write_research_brief  │  ← Generate research plan from query
└─────────┬──────────────┘
          │
          ▼
┌─────────────────────┐
│   validate_brief    │  ← Council 1: Multi-model validation (optional)
└─────────┬───────────┘
          │
          ▼
┌────────────────────────┐
│  research_supervisor   │  ← Main research loop (subgraph)
│  ┌──────────────────┐  │
│  │    supervisor    │◄─┼──── Decides what to research
│  └────────┬─────────┘  │
│           │            │
│           ▼            │
│  ┌──────────────────┐  │
│  │ supervisor_tools │  │
│  │   ┌──────────┐   │  │
│  │   │researcher│   │  │ ← Parallel sub-agents do actual searches
│  │   │subgraph  │   │  │
│  │   └──────────┘   │  │
│  └──────────────────┘  │
└─────────┬──────────────┘
          │
          ▼
┌─────────────────────┐
│  validate_findings  │  ← Council 2: Fact-check research notes
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  extract_evidence   │  ← S03: Extract candidate quotes from sources
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   verify_evidence   │  ← S04: Verify quotes exist in sources
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────┐
│ final_report_generation │  ← Generate report with verified findings
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────┐
│    verify_claims    │  ← Health check: Claim verification
└─────────┬───────────┘
          │
          ▼
      Final Report
```

---

## Pipeline Stages

### Stage 1: Store Gating (`check_store`)

**File:** `src/open_deep_research/nodes/store.py`

**Purpose:** Fail-fast check for LangGraph Store availability.

**What it does:**
- Calls `langgraph.config.get_store()` to check if external store is available
- Checks if `thread_id` is present in config
- If store unavailable: Sets `verified_disabled = True` (disables Verified Findings section)
- If store available: Sets `verified_disabled = False`

**Why it exists:** The Verified Findings section requires storing source content for verification. If the store is unavailable, we disable this feature rather than fail silently.

---

### Stage 2: User Clarification (`clarify_with_user`)

**File:** `src/open_deep_research/nodes/clarify.py`

**Purpose:** Ask the user clarifying questions before starting research.

**What it does:**
- Analyzes user messages for ambiguity
- Generates clarifying questions if needed (controlled by `allow_clarification` config)
- Returns structured response: `need_clarification`, `question`, `verification`

**Config:** `allow_clarification: bool` (default: true)

---

### Stage 3: Brief Generation (`write_research_brief`)

**File:** `src/open_deep_research/nodes/brief.py`

**Purpose:** Transform user query into a detailed research plan.

**What it does:**

1. **Optional Context Injection** (if `enable_brief_context: true`):
   - Runs preliminary Tavily searches to gather recent context
   - Extracts key entities, recent events, metrics
   - Injects this context into brief generation prompt

2. **Brief Generation:**
   - Uses `research_model` with structured output
   - Generates `ResearchQuestion` with detailed `research_brief` field
   - Initializes supervisor messages with the brief

**Key Prompt:** `transform_messages_into_research_topic_prompt`

**Output:** `research_brief` string stored in state

---

### Stage 4: Brief Validation (`validate_brief`)

**File:** `src/open_deep_research/nodes/brief.py`

**Purpose:** Multi-model council review of the research brief.

**What it does:**
- If `use_council: true`: Runs multiple models in parallel to review brief
- Models vote: approve/revise/reject with feedback
- Calculates consensus score
- If consensus below threshold: Routes back to `write_research_brief` for revision
- If `review_mode != "none"`: Interrupts for human review

**Config:**
- `use_council: bool` (default: true)
- `council_models: List[str]` (default: ["openai:gpt-4.1", "anthropic:claude-sonnet-4-5-20250929"])
- `council_min_consensus: float` (default: 0.7)
- `council_max_revisions: int` (default: 3)

---

### Stage 5: Research Supervisor (`research_supervisor` subgraph)

**File:** `src/open_deep_research/nodes/supervisor.py`

The supervisor is a subgraph with its own loop:

#### Supervisor Node (`supervisor`)

**Purpose:** Decide what research tasks to delegate.

**Tools available:**
- `ConductResearch` - Delegate a research task to a sub-agent
- `ResearchComplete` - Signal that research is complete
- `think_tool` - Strategic reflection/planning

**Flow:**
1. Receives research brief
2. Uses `think_tool` to plan approach
3. Calls `ConductResearch` with specific topics
4. Analyzes results, decides if more research needed
5. Calls `ResearchComplete` when satisfied

#### Supervisor Tools Node (`supervisor_tools`)

**Purpose:** Execute supervisor's tool calls.

**What it does:**

1. **think_tool calls:** Records reflection, continues loop

2. **ConductResearch calls:**
   - Spawns `researcher_subgraph` for each topic (up to `max_concurrent_research_units`)
   - Executes in parallel with `asyncio.gather`
   - Aggregates results: `compressed_research`, `raw_notes`, `source_store`

3. **ResearchComplete calls:** Exits to next stage

**Exit conditions:**
- `research_iterations > max_researcher_iterations`
- No tool calls made
- `ResearchComplete` called

---

### Stage 6: Researcher Subgraph

**File:** `src/open_deep_research/nodes/researcher.py`

Each `ConductResearch` call spawns a researcher subgraph:

```
researcher_messages (topic)
        │
        ▼
   ┌────────────┐
   │ researcher │◄────── Loop: Search, reflect, search...
   └─────┬──────┘
         │
         ▼
┌────────────────────┐
│  researcher_tools  │
└─────────┬──────────┘
          │ (when done)
          ▼
┌───────────────────┐
│ compress_research │
└─────────┬─────────┘
          │
          ▼
    Return to supervisor
```

#### Researcher Node (`researcher`)

**Tools available:**
- `tavily_search` - Web search with summarization
- `think_tool` - Strategic reflection
- `ResearchComplete` - Signal done (optional)

**System prompt:** `research_system_prompt`

#### Researcher Tools Node (`researcher_tools`)

**What it does:**
- Executes search tools in parallel
- Creates `ToolMessage` for each result
- Checks iteration limits (`max_react_tool_calls`)
- Routes to `compress_research` when done

#### Compress Research Node (`compress_research`)

**Purpose:** Synthesize all findings into clean summary.

**What it does:**
1. Collects all tool outputs and AI messages
2. Uses `compression_model` to clean up findings
3. **Extracts sources from tool messages** using `extract_sources_from_tool_messages()`
4. Returns: `compressed_research`, `raw_notes`, `source_store`

**Source Extraction:**
- Parses Tavily output format: `--- SOURCE N: {title} ---\nURL: {url}\n\nSUMMARY:\n{content}`
- Creates `SourceRecord` dicts with url, title, content, extraction_method, timestamp
- Deduplicates by URL

---

### Stage 7: Findings Validation (`validate_findings`)

**File:** `src/open_deep_research/nodes/findings.py`

**Purpose:** Fact-check research findings before report generation.

**What it does:**
- Uses `FindingsReview` structured output
- Checks for: fabricated names, impossible dates, uncited claims, contradictions
- Logs flagged issues (non-blocking, advisory only)
- Stores `flagged_issues` in state for human review

**Config:** `use_findings_council: bool` (default: true)

---

### Stage 8: Evidence Extraction (`extract_evidence`)

**File:** `src/open_deep_research/nodes/extract.py`

**Purpose:** Extract candidate quotes from source content.

**What it does:**
1. Gets sources from `state.source_store` or LangGraph Store
2. For each source:
   - If `extraction_method == "extract_api"`: Uses spacy sentence chunking
   - If `extraction_method == "search_raw"`: Uses regex sanitization + paragraph extraction
3. Creates `EvidenceSnippet` for each passage (15-60 words)
4. Sets `status = "PENDING"`

**Output:** `evidence_snippets` list in state

**This is DETERMINISTIC - no LLM calls.**

---

### Stage 9: Evidence Verification (`verify_evidence`)

**File:** `src/open_deep_research/nodes/verify.py`

**Purpose:** Verify extracted quotes actually exist in sources.

**What it does:**
1. For each `EvidenceSnippet`:
   - **Check 1 (Strict):** Does `quote` appear in `source_content`?
   - **Check 2 (Fuzzy):** Jaccard similarity > 0.8 between quote and any source window?
2. Sets `status = "PASS"` or `status = "FAIL"`

**This is DETERMINISTIC - no LLM calls.**

---

### Stage 10: Final Report Generation (`final_report_generation`)

**File:** `src/open_deep_research/nodes/report.py`

**Purpose:** Generate comprehensive research report.

**What it does:**

1. **Generate Verified Findings Section (separate LLM call):**
   - Uses PASS snippets only
   - Selector-only prompt: "Pick 3-5 most relevant quotes, copy EXACTLY"
   - Prevents hallucination by limiting to verified quotes

2. **Generate Main Report:**
   - Injects Verified Findings as "IMMUTABLE" block
   - Uses `final_report_generation_prompt`
   - Includes: research brief, messages, findings

3. **Post-Check:**
   - `enforce_verified_section()` ensures LLM didn't modify the Verified Findings

4. **Optional Human Review** (if `review_mode == "full"`):
   - Interrupts with report preview
   - Human can: approve, edit, or provide feedback

---

### Stage 11: Claim Verification (`verify_claims`)

**File:** `src/open_deep_research/nodes/verify.py`

**Purpose:** Health check on final report claims.

**What it does:**
- Extracts claims from final report
- Matches against stored sources using embeddings
- Calculates support confidence
- Logs warnings for low-confidence claims

**Config:** `use_claim_verification: bool` (default: false)

---

## Core Components

### Tavily Search Tool

**File:** `src/open_deep_research/utils.py` (function: `tavily_search`)

**What it does:**
1. Executes search queries via Tavily API
2. Deduplicates results by URL
3. Summarizes each webpage using `summarization_model`
4. Stores source records for verification:
   - Tries Tavily Extract API first (cleaner content)
   - Falls back to raw search content
5. Returns formatted output with sources

**Output format:**
```
--- SOURCE 1: {title} ---
URL: {url}

SUMMARY:
{summary}
```

### Think Tool

**File:** `src/open_deep_research/utils.py` (function: `think_tool`)

**Purpose:** Strategic reflection during research.

**What it does:**
- Takes `reflection` string argument
- Returns confirmation message
- Creates deliberate pause for quality decision-making

**Used by:** Supervisor and Researcher for planning between tool calls.

### Council System

**File:** `src/open_deep_research/council.py`

**What it does:**
- Runs multiple models in parallel on same prompt
- Each model returns: decision, confidence, feedback
- Synthesizes feedback into single recommendation
- Calculates consensus score

---

## Data Flow

### Source Storage Flow

```
Tavily Search
     │
     ▼
┌─────────────────────┐
│ Parse search results │
│ Create SourceRecord  │
│ (url, title, content)│
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
 LangGraph    state.source_store
   Store      (in-memory backup)
     │           │
     └─────┬─────┘
           │
           ▼
    ┌─────────────┐
    │ Aggregate   │  ← supervisor_tools aggregates from all researchers
    │ from all    │
    │ researchers │
    └──────┬──────┘
           │
           ▼
    extract_evidence → verify_evidence → final_report
```

### Message Flow

```
User Message
     │
     ▼
state.messages  ─────────────────────────────────────┐
     │                                               │
     ▼                                               │
write_research_brief                                 │
     │                                               │
     ▼                                               │
state.research_brief ────────────────┐               │
state.supervisor_messages            │               │
     │                               │               │
     ▼                               │               │
supervisor loop                      │               │
     │                               │               │
     ▼                               │               │
researcher subgraph                  │               │
     │                               │               │
     ▼                               │               │
state.raw_notes ─────────────────────┤               │
state.notes ─────────────────────────┤               │
state.source_store ──────────────────┤               │
                                     │               │
                                     ▼               ▼
                              final_report_generation
                                     │
                                     ▼
                              state.final_report
```

---

## State Management

### Main States

#### AgentState (main graph)

```python
class AgentState(MessagesState):
    supervisor_messages: list          # Supervisor conversation
    research_brief: str                 # Generated research plan
    raw_notes: list[str]                # Raw research output
    notes: list[str]                    # Cleaned research notes
    final_report: str                   # Generated report

    # Council tracking
    council_revision_count: int
    feedback_on_brief: list[str]
    findings_revision_count: int
    feedback_on_findings: list[str]

    # Human review
    council_brief_feedback: str
    flagged_issues: list[str]
    human_approved_brief: str

    # Verification
    source_store: list[SourceRecord]    # Stored sources
    verification_result: VerificationResult
    verified_disabled: bool
    evidence_snippets: list[EvidenceSnippet]
```

#### SupervisorState (supervisor subgraph)

```python
class SupervisorState(TypedDict):
    supervisor_messages: list
    research_brief: str
    notes: list[str]
    research_iterations: int
    raw_notes: list[str]
    source_store: list[SourceRecord]
```

#### ResearcherState (researcher subgraph)

```python
class ResearcherState(TypedDict):
    researcher_messages: list
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: list[str]
    source_store: list[SourceRecord]
```

### State Reducers

- `override_reducer`: Allows replacing list values (used for supervisor_messages reset)
- `operator.add`: Appends to lists (used for notes, source_store)

---

## Configuration Options

### Core Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `test_mode` | false | Reduces iterations for fast testing |
| `search_api` | tavily | Search provider (tavily, openai, anthropic) |
| `allow_clarification` | true | Ask clarifying questions |

### Research Limits

| Setting | Default | Test Mode |
|---------|---------|-----------|
| `max_researcher_iterations` | 6 | 2 |
| `max_react_tool_calls` | 10 | 3 |
| `max_concurrent_research_units` | 5 | 2 |

### Model Configuration

| Setting | Default |
|---------|---------|
| `research_model` | openai:gpt-4.1 |
| `summarization_model` | openai:gpt-4.1-mini |
| `compression_model` | openai:gpt-4.1 |
| `final_report_model` | openai:gpt-4.1 |

### Council Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `use_council` | true | Enable brief validation |
| `council_models` | [gpt-4.1, claude-sonnet] | Models for council |
| `council_min_consensus` | 0.7 | Approval threshold |
| `council_max_revisions` | 3 | Max revision attempts |
| `use_findings_council` | true | Enable findings fact-check |

### Verification Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `use_claim_verification` | false | Post-report claim check |
| `use_tavily_extract` | true | Use Extract API for cleaner content |
| `enable_brief_context` | true | Pre-search for brief context |

---

## Anti-Hallucination System

The system uses multiple layers to prevent hallucinated information:

### Layer 1: Grounding Prompts

All prompts include explicit grounding rules:
```
You may ONLY include information that appears in the search results.
DO NOT invent, fabricate, or extrapolate any facts.
Every single claim MUST have a citation.
```

### Layer 2: Council Fact-Checking

- `validate_findings` checks for:
  - Fabricated names
  - Impossible dates (future dates)
  - Uncited claims
  - Suspicious statistics

### Layer 3: Evidence Extraction (S03)

- **Deterministic** - No LLM calls
- Extracts candidate quotes from raw source content
- Uses sentence chunking or paragraph extraction

### Layer 4: Evidence Verification (S04)

- **Deterministic** - No LLM calls
- Verifies quotes exist in sources
- Two checks: strict substring, fuzzy Jaccard (0.8 threshold)

### Layer 5: Verified Findings Section

- Separate LLM call with selector-only prompt
- Can ONLY use PASS-verified quotes
- Post-check enforces section wasn't modified

### Layer 6: Claim Verification (Optional)

- Post-report health check
- Uses embeddings to match claims to sources
- Logs warnings for unsupported claims

---

## Key Files Reference

### Graph Construction

| File | Purpose |
|------|---------|
| `src/open_deep_research/graph.py` | Main LangGraph construction |
| `src/open_deep_research/state.py` | State definitions |
| `src/open_deep_research/configuration.py` | Configuration class |
| `src/open_deep_research/models.py` | Model initialization |

### Nodes

| File | Nodes |
|------|-------|
| `nodes/store.py` | `check_store` |
| `nodes/clarify.py` | `clarify_with_user` |
| `nodes/brief.py` | `write_research_brief`, `validate_brief` |
| `nodes/supervisor.py` | `supervisor`, `supervisor_tools` |
| `nodes/researcher.py` | `researcher`, `researcher_tools`, `compress_research` |
| `nodes/findings.py` | `validate_findings` |
| `nodes/extract.py` | `extract_evidence` |
| `nodes/verify.py` | `verify_evidence`, `verify_claims` |
| `nodes/report.py` | `final_report_generation` |

### Utilities

| File | Purpose |
|------|---------|
| `utils.py` | Tavily search, tool helpers, source storage |
| `prompts.py` | All system prompts |
| `council.py` | Multi-model council logic |
| `verification.py` | Claim verification with embeddings |
| `logic/sanitize.py` | HTML sanitization for quotes |
| `logic/document_processing.py` | Spacy-based chunking |

### Test Scripts

| File | Purpose |
|------|---------|
| `scripts/test_brief.py` | Test brief generation (~30s) |
| `scripts/test_research.py` | Test research phase (~2-3 min) |
| `scripts/test_report.py` | Test report generation (~30s) |
| `scripts/staged_config.py` | Shared minimal config |

---

## Performance Notes

### Typical Run Metrics

| Metric | Full Run | Test Mode |
|--------|----------|-----------|
| Time | 5-15 min | 1-3 min |
| Tokens | 500k-1.5M | 50k-150k |
| Sources | 20-50 | 5-25 |
| Cost | $1-5 | $0.10-0.30 |

### Bottlenecks

1. **Tavily searches** - Each search takes 2-5 seconds
2. **Summarization** - Summarizing webpage content is slow
3. **Council calls** - Running multiple models adds latency
4. **Report generation** - Final report can take 30-60 seconds

### Optimization Tips

1. Use `test_mode: true` for iteration
2. Disable council (`use_council: false`) when not needed
3. Reduce `max_researcher_iterations` and `max_react_tool_calls`
4. Use faster models for summarization
