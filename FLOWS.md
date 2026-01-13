# Deep Research: Execution Flows

Detailed traces of how data moves through the system.

---

## Table of Contents

1. [Main Pipeline Flow](#1-main-pipeline-flow)
2. [Research Phase Flow](#2-research-phase-flow)
3. [Pointer Extraction Flow](#3-pointer-extraction-flow)
4. [Deduplication Flow](#4-deduplication-flow)
5. [Arrangement Flow](#5-arrangement-flow)
6. [Synthesis Flow](#6-synthesis-flow)
7. [Council Validation Flow](#7-council-validation-flow)
8. [Evaluation Flow](#8-evaluation-flow)

---

## 1. Main Pipeline Flow

The complete path from user query to final report.

### Graph Definition (`graph.py`)

```
START
  │
  ▼
check_store (S02)
  │ Verifies Store available for source storage
  │ Sets verified_disabled=True if unavailable
  │
  ▼
clarify_with_user
  │ If allow_clarification=True:
  │   LLM analyzes if query needs clarification
  │   If need_clarification: interrupt with question
  │
  ▼
write_research_brief
  │ Transforms user messages into research plan
  │ If enable_brief_context=True:
  │   Pre-search Tavily for context injection
  │
  ▼
validate_brief (Council 1) ─── if use_council=True
  │ Multi-model voting on brief quality
  │ Loop: revise → vote → revise (max council_max_revisions)
  │
  ▼
research_supervisor (subgraph)
  │ See: Research Phase Flow
  │ Output: notes, raw_notes, source_store
  │
  ▼
validate_findings (Council 2) ─── if use_findings_council=True
  │ Fact-check research findings
  │ Flag issues for human review
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ CONDITIONAL: use_safeguarded_generation?                │
│                                                         │
│ TRUE (default):                    FALSE (legacy):      │
│ ┌─────────────────┐                ┌─────────────────┐  │
│ │safeguarded_report│               │ extract_evidence │  │
│ │  (Pipeline v2)   │               │      (S03)      │  │
│ │                 │                └────────┬────────┘  │
│ │ 1. Batch extract │                        │          │
│ │ 2. Dedup + clean │               ┌────────▼────────┐  │
│ │ 3. Arrange themes│               │ verify_evidence │  │
│ │ 4. Per-theme     │               │      (S04)      │  │
│ │    synthesis     │               └────────┬────────┘  │
│ └─────────────────┘                         │          │
│                                    ┌────────▼────────┐  │
│                                    │ claim_pre_check │  │
│                                    │   (Layer 3)     │  │
│                                    └────────┬────────┘  │
│                                             │          │
│                                    ┌────────▼────────┐  │
│                                    │final_report_gen │  │
│                                    └────────┬────────┘  │
└─────────────────────────────────────────────────────────┘
  │
  ▼
run_evaluation ─── if run_evaluation=True
  │ Post-hoc quality check
  │
  ▼
END
```

### State Transitions

```
AgentState at each stage:

START
├── messages: [HumanMessage("query")]
└── (all other fields empty)

After check_store:
├── verified_disabled: bool
└── verified_disabled_reason: str

After write_research_brief:
├── research_brief: str (research plan)
└── council_revision_count: int

After research_supervisor:
├── notes: List[str] (compressed findings)
├── raw_notes: List[str] (full search results)
└── source_store: List[SourceRecord] (20-200 sources)

After safeguarded_report:
├── final_report: str (HTML)
├── hybrid_report: dict (structured data)
└── checkpoints: dict (pre/post dedup, arrangement)

After run_evaluation:
└── eval_result: dict (metrics)
```

---

## 2. Research Phase Flow

How the supervisor delegates to researchers.

### Supervisor Subgraph (`nodes/supervisor.py`)

```
ENTRY: supervisor
  │
  │ supervisor(state) → Command["supervisor_tools"]
  │   Configure model with tools: ConductResearch, ResearchComplete, think_tool
  │   Generate response based on supervisor_messages
  │
  ▼
supervisor_tools
  │
  ├── Check exit conditions:
  │   - research_iterations > max_researcher_iterations?
  │   - No tool calls?
  │   - ResearchComplete called?
  │   → If any: goto END with notes, source_store
  │
  ├── Handle think_tool calls (reflection):
  │   Add ToolMessage: "Reflection recorded: {content}"
  │
  └── Handle ConductResearch calls (delegation):
      │
      │ For each ConductResearch (up to max_concurrent_research_units):
      │
      ▼
    ┌─────────────────────────────────────────┐
    │ researcher_subgraph.ainvoke({          │
    │   researcher_messages: [HumanMessage(  │
    │     content=research_topic             │
    │   )],                                  │
    │   research_topic: research_topic       │
    │ })                                     │
    └─────────────────────────────────────────┘
      │
      │ Parallel execution via asyncio.gather
      │
      ▼
    Aggregate results:
      - raw_notes: concatenated raw findings
      - source_store: merged, deduped, filtered
        - Skip duplicates (by URL)
        - Skip low quality (< min_source_content_length)
        - Enforce limit (max_total_sources)
      │
      ▼
    Return Command["supervisor"] with tool messages
      │
      │ Loop continues until exit condition
      │
      ▼
EXIT: notes, raw_notes, source_store → parent AgentState
```

### Researcher Subgraph (`nodes/researcher.py`)

```
ENTRY: researcher
  │
  │ researcher(state) → Command["researcher_tools"]
  │   Configure model with tools from get_all_tools():
  │     - tavily_search (if search_api=TAVILY)
  │     - MCP tools (if mcp_config set)
  │     - think_tool
  │
  ▼
researcher_tools
  │
  ├── Check early exit:
  │   - No tool calls? → goto compress_research
  │
  ├── Execute all tool calls in parallel:
  │   │
  │   ├── tavily_search:
  │   │   queries: List[str] → results with URLs, content
  │   │   Cache sources via cache_sources()
  │   │
  │   ├── think_tool:
  │   │   Reflection recorded
  │   │
  │   └── MCP tools:
  │       External tool execution
  │
  ├── Check late exit:
  │   - tool_call_iterations >= max_react_tool_calls?
  │   - ResearchComplete called?
  │   → If any: goto compress_research
  │
  └── Continue: goto researcher with tool messages
      │
      ▼
compress_research
  │
  │ compress_research(state) → dict
  │   Model: compression_model
  │   Prompt: compress_research_system_prompt
  │
  │   Extract sources from tool messages:
  │     1. Try get_cached_sources() (Extract API cache)
  │     2. Fallback: parse tool message format
  │
  │   Handle token limit: remove older messages, retry
  │
  ▼
EXIT: {
  compressed_research: str,
  raw_notes: List[str],
  source_store: List[dict]
}
```

### Tavily Search Tool (`utils.py`)

```python
@tool
async def tavily_search(
    queries: List[str],
    config: RunnableConfig
) -> str:
    """
    For each query:
    1. Check blocked_domains
    2. If use_tavily_extract=True:
       - Call client.extract(urls) for clean content
       - Cache results
    3. Else:
       - Use raw_content from search results
    4. Format as:
       --- SOURCE N: {title} ---
       URL: {url}

       SUMMARY:
       {content}
    """
```

---

## 3. Pointer Extraction Flow

How facts are extracted from sources (Pipeline v2).

### Entry Point (`nodes/safeguarded_report.py`)

```python
async def safeguarded_report_generation(state, config):
    # Build sources dict from state.source_store
    sources = {
        f"src_{i:03d}": {
            "content": src.get("content") or src.get("raw_content"),
            "url": src.get("url"),
            "title": src.get("title"),
        }
        for i, src in enumerate(state.source_store)
    }

    # Call pipeline v2
    report = await run_pipeline_v2(
        sources=sources,
        topic=research_brief,
        title=title,
        llm_call=llm_call,
        on_progress=on_progress
    )
```

### Pipeline v2 Stages (`pipeline_v2.py`)

```
Stage 1: BATCH EXTRACTION
══════════════════════════════════════════════════════════════

sources: Dict[source_id → {content, url, title}]
  │
  ▼
batch_sources(sources, batch_size=1)
  │ Split into single-source batches for thoroughness
  │
  ▼
For each batch (parallel via asyncio.gather):
  │
  ├── If content > CHUNK_THRESHOLD (100k chars):
  │   │
  │   ▼
  │   extract_from_source_chunked():
  │     1. chunk_content() → split at paragraph/sentence boundaries
  │     2. For each chunk:
  │        - LLM extracts keywords
  │        - Prompt: "Extract facts that help answer: {topic}"
  │     3. Verify against full content
  │
  └── Else:
      │
      ▼
      extract_batch():
        1. format_sources_for_prompt() → formatted text
        2. POINTER_PROMPT + topic → LLM
        3. parse_pointer_response() → List[Pointer]
        4. For each Pointer:
           - extract_from_pointer() → Extraction
  │
  ▼
All extractions: List[Extraction]
  ├── status: "verified" | "partial" | "not_found"
  ├── extracted_text: str (cleaned)
  ├── match_score: float (0.0-1.0)
  └── source_url: str

CHECKPOINT: checkpoints["pre_dedup"] = all extractions
```

### Pointer Extraction (`pointer_extract.py`)

```
POINTER_PROMPT
══════════════════════════════════════════════════════════════

"Research question: {topic}

A fact is a specific, verifiable claim. Examples:
- 'Model X has 150ms latency' ✓
- 'Pricing starts at $0.01 per 1000 chars' ✓
- 'This is where things get interesting' ✗ (intro fluff)

Extract facts that help answer the research question.
For each, output 3-5 unique keywords:

Text:
{sources}

Output JSON array:
[{\"source_id\": \"src_001\", \"keywords\": [\"Model\", \"X\", \"150ms\"]}]"
```

### Extraction from Pointer

```
extract_from_pointer(pointer, sources, min_score)
  │
  │ pointer.keywords: ["Model", "X", "150ms", "latency"]
  │
  ▼
find_best_match(keywords, source_content, min_score):
  │
  │ 1. Check if keywords exist in source
  │    keywords_found = [k for k in keywords if k in content_lower]
  │    match_ratio = len(found) / len(keywords)
  │
  │ 2. If match_ratio < min_score: return (None, match_ratio)
  │
  │ 3. Find sentences containing most keywords
  │    Split on: paragraphs, headers, then sentences
  │
  │ 4. Score each sentence:
  │    - keyword_count / len(keywords)
  │    - Boost for consecutive keywords
  │
  │ 5. Return best sentence
  │
  ▼
clean_extracted_text(text, max_length=200):
  │
  │ Strip HTML, markdown, table syntax
  │ Truncate at sentence boundary
  │
  ▼
is_quality_extraction(text):
  │
  │ Reject if:
  │   - < 50 chars
  │   - > 50 words
  │   - Contains 2+ pipe chars (table)
  │   - Navigation patterns
  │   - Low alpha ratio (< 0.35)
  │   - Markdown artifacts
  │   - Header patterns
  │
  ▼
Extraction(
  pointer=pointer,
  status="verified" | "partial" | "not_found",
  extracted_text=cleaned_text,
  match_score=score,
  source_url=source["url"]
)
```

---

## 4. Deduplication Flow

Two-stage deduplication: LLM semantic + cross-batch text similarity.

```
Input: List[Extraction] (all verified)
  │
  ▼
STAGE 1: LLM SEMANTIC DEDUP
══════════════════════════════════════════════════════════════

deduplicate_llm(extractions, llm_call):
  │
  │ Prompt:
  │ "You are a deduplication expert.
  │  Compare these facts and identify SEMANTIC duplicates.
  │  NOT duplicates: '200ms' vs '180ms' (different numbers)
  │  ARE duplicates: 'under 500ms' vs 'below 500 milliseconds'
  │
  │  Facts:
  │  [1] {text}
  │  [2] {text}
  │  ...
  │
  │  Output JSON: {\"remove_ids\": [2, 5, 8]}"
  │
  ▼
LLM identifies semantically duplicate facts
  │
  ▼
Remove duplicates, keep originals

CHECKPOINT: checkpoints["post_dedup"] = after LLM dedup


STAGE 2: CROSS-BATCH TEXT SIMILARITY
══════════════════════════════════════════════════════════════

cross_batch_dedup(extractions, threshold=0.5):
  │
  │ For each pair of extractions:
  │   1. normalize_for_comparison() → strip markdown, lowercase
  │   2. compute_text_similarity():
  │      - extract_numbers() → protect different numbers
  │      - Jaccard similarity on word sets
  │      - If different numbers exist: return 0.0
  │   3. If similarity > threshold: mark later one as duplicate
  │
  ▼
Remove cross-batch duplicates

Result: deduplicated List[Extraction]
```

### Number Protection Logic

```python
def extract_numbers(text: str) -> set:
    """
    Match: 10000, 10,000, 10.5, 10%, 200ms, $1.2B
    Normalize: remove commas, lowercase
    """

def compute_text_similarity(text1, text2, protect_numbers=True):
    """
    If protect_numbers=True:
      nums1, nums2 = extract_numbers(text1), extract_numbers(text2)
      if nums1 and nums2 and nums1 != nums2:
        return 0.0  # Different numbers = different facts

    Then: Jaccard similarity on word sets
    """
```

---

## 5. Arrangement Flow

LLM groups facts by theme and excludes irrelevant content.

```
Input: List[Extraction] (deduplicated)
  │
  ▼
CHECKPOINT: checkpoints["pre_arrangement"] = facts
  │
  ▼
ARRANGER PROMPT
══════════════════════════════════════════════════════════════

"You are organizing research findings into a coherent report.

Research topic: {topic}

Facts to organize (each has ID):
[1] {extracted_text} (from: {domain})
[2] {extracted_text} (from: {domain})
...

Tasks:
1. Group into 3-7 themes
2. Exclude irrelevant/redundant facts (30-50% typically)

Output JSON:
{
  \"themes\": [
    {\"name\": \"Theme 1\", \"fact_ids\": [1, 4, 7]},
    {\"name\": \"Theme 2\", \"fact_ids\": [2, 5]}
  ],
  \"excluded_ids\": [3, 6, 8]
}"
  │
  ▼
LLM groups facts and excludes noise
  │
  ▼
parse_arrangement_response():
  │
  │ Validate:
  │   - All fact_ids reference valid facts
  │   - No ID in both theme and excluded
  │   - Handle edge cases (empty themes, etc.)
  │
  ▼
CuratedFacts(
  groups: List[ThemeGroup],
  excluded_ids: List[int]
)

CHECKPOINT: checkpoints["post_arrangement"] = {
  "themes": [...],
  "excluded_ids": [...],
  "grouped_count": N,
  "excluded_count": M
}
```

---

## 6. Synthesis Flow

Per-theme prose generation with citations.

```
Input: CuratedFacts with themed groups
  │
  ▼
For each ThemeGroup:
══════════════════════════════════════════════════════════════

SYNTHESIS PROMPT:
"Write prose connecting these verified facts about '{theme}'.

Facts (use [N] citations):
[1] {extracted_text}
[2] {extracted_text}
...

Rules:
- Every sentence with a factual claim MUST have a [N] citation
- Do not rewrite facts, reference them
- Write 2-4 sentences connecting the facts
- Output plain prose only, no JSON"
  │
  ▼
LLM writes prose with [N] citations
  │
  ▼
Parse citations: re.findall(r'\[(\d+)\]', prose)
  │
  ▼
ThemedSection(
  theme: str,
  prose: str,  # "Research shows [1] that latency is critical [2]..."
  citations: List[Citation],  # [{marker: "[1]", fact_index: 0}, ...]
  facts: List[Extraction]  # The actual facts
)
  │
  ▼
ASSEMBLE REPORT
══════════════════════════════════════════════════════════════

Generate:
  - executive_summary: LLM writes based on all themes
  - conclusion: LLM writes based on all themes
  - analysis: (hidden by default)
  │
  ▼
HybridReport(
  title=title,
  executive_summary=summary,
  sections=List[ThemedSection],
  analysis=analysis,
  conclusion=conclusion,
  excluded_facts=excluded,
  checkpoints=checkpoints
)
  │
  ▼
RENDER TO HTML
══════════════════════════════════════════════════════════════

render_report(report):
  │
  │ 1. report_to_dict() → template-friendly dict
  │    - Assign global footnote IDs across sections
  │    - Track which facts are cited
  │
  │ 2. Load template from templates/report.html
  │
  │ 3. Render sections:
  │    - Theme header
  │    - Prose with linked [N] citations
  │    - Facts displayed with source links
  │    - Green styling for verified facts
  │    - Gray styling for AI prose
  │
  │ 4. Render footnotes section
  │
  ▼
HTML string → final_report in state
```

---

## 7. Council Validation Flow

Multi-model consensus for brief and findings validation.

### Council Vote (`council.py`)

```
Input: research_brief or findings
  │
  ▼
council_vote_on_brief(brief, query, config):
  │
  │ council_config = CouncilConfig(
  │   models=config.council_models,  # ["openai:gpt-4.1", "anthropic:claude-sonnet"]
  │   min_consensus_for_approve=config.council_min_consensus
  │ )
  │
  ▼
For each model (parallel):
══════════════════════════════════════════════════════════════

BRIEF_REVIEW_PROMPT:
"Review this research brief:

Query: {query}
Brief: {brief}

Evaluate:
1. Does it capture the user's intent?
2. Is it specific enough to guide research?
3. Are there problematic assumptions?

Respond with JSON:
{
  \"decision\": \"approve\" | \"revise\" | \"reject\",
  \"confidence\": 0.0-1.0,
  \"strengths\": [...],
  \"weaknesses\": [...],
  \"suggested_changes\": [...],
  \"reasoning\": \"...\"
}"
  │
  ▼
Parse structured response → BriefReview
  │
  ▼
Aggregate votes
══════════════════════════════════════════════════════════════

calculate_consensus(votes):
  │
  │ approve_weight = sum(v.confidence for v if v.decision == "approve")
  │ total_weight = sum(v.confidence for v in votes)
  │ consensus_score = approve_weight / total_weight
  │
  │ If consensus_score >= min_consensus_for_approve:
  │   decision = "approve"
  │ Elif any reject:
  │   decision = "reject" (if require_unanimous_for_reject)
  │ Else:
  │   decision = "revise"
  │
  ▼
CouncilVerdict(
  decision="approve" | "revise" | "reject",
  consensus_score=0.85,
  votes=List[CouncilVote],
  synthesized_feedback="...",
  requires_revision=bool
)
```

### Revision Loop (`nodes/brief.py`)

```
validate_brief(state, config):
  │
  ▼
While council_revision_count < council_max_revisions:
  │
  │ verdict = council_vote_on_brief(brief, query)
  │
  │ If verdict.decision == "approve":
  │   Break → proceed to research
  │
  │ If verdict.decision == "reject":
  │   Break → abort or force proceed
  │
  │ If verdict.decision == "revise":
  │   │
  │   ▼
  │   Revise brief using synthesized_feedback
  │   council_revision_count += 1
  │   Continue loop
  │
  ▼
Proceed to research_supervisor
```

---

## 8. Evaluation Flow

Post-hoc quality assessment.

### Entry Point (`nodes/eval.py`)

```python
async def run_evaluation_node(state, config):
    # Only runs if run_evaluation=True

    result = await evaluate_report(
        report=state.final_report,
        sources=state.source_store,
        config=EvalConfig(
            max_claims=config.max_claims_to_verify,
            model=config.evaluation_model
        )
    )

    return {"eval_result": result}
```

### Evaluation Flow (`evaluation.py`)

```
evaluate_report(report, sources, config):
  │
  ▼
STEP 1: EXTRACT CLAIMS
══════════════════════════════════════════════════════════════

extract_claims(report):
  │
  │ Parse report for sentences with:
  │   - [N] citations
  │   - Factual claims (numbers, names, dates)
  │
  │ Classify each:
  │   - Cited: has [N] reference
  │   - Uncited: factual claim without citation
  │
  ▼
List[ExtractedClaim]
  │
  ▼
STEP 2: VERIFY EACH CLAIM
══════════════════════════════════════════════════════════════

Citation-First Approach:

For each claim (parallel batches):
  │
  ├── If claim has citations [N][M]:
  │   │
  │   │ 1. Find source N and M from sources list
  │   │ 2. find_relevant_passages(claim, source.content)
  │   │    - Chunk source into sentences
  │   │    - Embed claim and chunks
  │   │    - Cosine similarity ranking
  │   │    - Entity boost for names/numbers
  │   │
  │   │ 3. verify_single_claim(claim, passages):
  │   │    │
  │   │    │ Prompt:
  │   │    │ "Does this passage support the claim?
  │   │    │  Claim: {claim}
  │   │    │  Passage: {passage}
  │   │    │
  │   │    │  Respond: TRUE, FALSE, or UNVERIFIABLE"
  │   │    │
  │   │    ▼
  │   │    VerificationVote(status, confidence, evidence)
  │   │
  │   ▼
  │   ClaimResult(status="TRUE" | "FALSE" | "UNVERIFIABLE")
  │
  └── If uncited (high risk):
      │
      │ 1. Embedding search across ALL sources
      │ 2. Find best matching passages
      │ 3. Verify against found passages
      │ 4. Flag as uncited in result
      │
      ▼
      ClaimResult(is_uncited=True, ...)
  │
  ▼
STEP 3: COMPUTE METRICS
══════════════════════════════════════════════════════════════

claim_metrics = ClaimMetrics(
  total=len(claims),
  true_count=count(TRUE),
  false_count=count(FALSE),
  unverifiable_count=count(UNVERIFIABLE),
  uncited_count=count(is_uncited),
  hallucination_rate=false_count / total,
  grounding_rate=true_count / total
)

citation_metrics = CitationMetrics(
  total=total_citations,
  valid=citations_pointing_to_real_sources,
  supported=citations_where_source_supports_claim,
  accuracy=supported / total
)
  │
  ▼
EvaluationResult(
  claim_metrics=claim_metrics,
  citation_metrics=citation_metrics,
  claims=List[ClaimResult],
  evaluated_at=timestamp
)
```

### Standalone Eval (`eval/run_eval.py`)

```
run_eval.py dataset.json --mode medium
  │
  ▼
Load dataset (gold_query fixture)
  │
  ├── research_brief
  ├── source_store
  └── hybrid_report
  │
  ▼
Three evaluation phases:
  │
  ├── Brief Eval:
  │   - Preservation: Did brief keep query specifics?
  │   - Dilution: Did brief avoid generalizing?
  │   - Assumptions: Did brief avoid adding constraints?
  │
  ├── Upstream Eval:
  │   - Fact quality (1-5 scale)
  │   - Theme coverage (1-5 scale)
  │   - Duplicate rate
  │   - Low quality rate
  │   - Match score
  │
  └── Downstream Eval:
      - Citation accuracy (1-5 scale)
      - Synthesis quality (1-5 scale)
      - Uncited rate
  │
  ▼
PASS / WARN / FAIL based on thresholds in metrics.py
```

---

## Quick Reference: Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `deep_researcher` | `graph.py` | Main compiled graph |
| `supervisor` | `nodes/supervisor.py` | Delegates research tasks |
| `researcher` | `nodes/researcher.py` | Executes web searches |
| `run_pipeline_v2` | `pipeline_v2.py` | Three-stage extraction |
| `extract_from_pointer` | `pointer_extract.py` | Keyword → text extraction |
| `deduplicate_llm` | `pipeline_v2.py` | Semantic deduplication |
| `arrange_facts` | `pipeline_v2.py` | Theme grouping |
| `synthesize_theme` | `pipeline_v2.py` | Per-theme prose |
| `render_report` | `render.py` | HTML generation |
| `council_vote_on_brief` | `council.py` | Multi-model voting |
| `evaluate_report` | `evaluation.py` | Post-hoc quality check |
