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
7. [Synthesis Validation Flow](#7-synthesis-validation-flow) ← NEW (Phase 8)
8. [Council Validation Flow](#8-council-validation-flow)
9. [Artifacts Flow](#9-artifacts-flow) ← NEW (Phase 8)
10. [Evaluation Flow](#10-evaluation-flow)

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
│   └── sections[].facts[]:
│       ├── extracted_text: str
│       ├── span_start, span_end: int ← NEW (Phase 8)
│       ├── keywords_matched: List[str] ← NEW
│       ├── verification_method: str ← NEW
│       └── failure_reason: Optional[str] ← NEW
├── checkpoints: dict (pre/post dedup, arrangement)
│   └── Now persisted to checkpoint_YYYYMMDD_HHMMSS.json ← NEW (Phase 8)
└── artifacts: (if artifacts_dir provided) ← NEW (Phase 8)
    └── Saved to run_{id}_{date}.json

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

How facts are extracted from sources (Pipeline v2). Updated 2026-01-13 with Phase 8 changes.

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
        on_progress=on_progress,
        artifacts_dir=Path("artifacts"),   # NEW: Artifact persistence
        checkpoint_dir=Path("checkpoints") # NEW: Checkpoint persistence
    )
```

### Pipeline v2 Stages (`pipeline_v2.py`)

```
Stage 1: BATCH EXTRACTION
══════════════════════════════════════════════════════════════

sources: Dict[source_id → {content, url, title}]
  │
  ▼
CREATE ARTIFACTS RECORD (NEW)
  │ artifacts = create_run_artifacts(topic)
  │ For each source:
  │   content_hash = SHA256(content)[:16]
  │   Record SourceArtifact(url, title, content_hash, content_length)
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
  │        - LLM extracts keywords + micro_quote
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
           - verify_span() → validate extraction (NEW)
  │
  ▼
All extractions: List[Extraction]
  ├── status: "verified" | "partial" | "not_found" | "span_mismatch"
  ├── extracted_text: str (cleaned)
  ├── match_score: float (0.0-1.0)
  ├── source_url: str
  ├── span_start, span_end: int (character offsets) ← NEW
  ├── keywords_matched: List[str] ← NEW
  ├── verification_method: str ← NEW
  └── failure_reason, failure_details: Optional ← NEW

CHECKPOINT: checkpoints["pre_dedup"] = all extractions
  │ If checkpoint_dir: save to checkpoint_YYYYMMDD_HHMMSS.json (NEW)
```

### Pointer Prompt (Updated)

```
POINTER_PROMPT
══════════════════════════════════════════════════════════════

"Topic: {topic}

Extract FACTUAL CLAIMS - single sentences with specific, verifiable information.

CRITICAL: Each fact = ONE sentence. Keywords must all appear in the SAME sentence.

For each fact, provide:
- source_id: exactly as shown (src_000, src_001, etc)
- keywords: 3-5 distinctive words from ONE sentence only
- micro_quote: 8-15 word phrase that MUST appear VERBATIM in the source ← NEW
- context: brief 3-6 word label

The micro_quote is CRITICAL - it anchors the extraction to exact text.

Output JSON array:
[
  {\"source_id\": \"src_000\", \"keywords\": [...], \"micro_quote\": \"exact phrase from source\", \"context\": \"...\"}
]"
```

### Extraction from Pointer (Updated)

```
extract_from_pointer(pointer, sources, min_score)
  │
  │ pointer.keywords: ["Model", "X", "150ms", "latency"]
  │ pointer.micro_quote: "Model X achieves 150ms latency" ← NEW
  │
  ▼
find_best_match(keywords, source_content, min_score, micro_quote):
  │
  │ ┌────────────────────────────────────────────────────────────┐
  │ │ MATCHING ORDER (NEW - 3-tier approach):                     │
  │ │                                                             │
  │ │ 1. MICRO-QUOTE EXACT MATCH (highest confidence)             │
  │ │    - Try exact substring match of micro_quote               │
  │ │    - Then case-insensitive match                            │
  │ │    - If found → score=1.0, method="micro_quote"             │
  │ │                                                             │
  │ │ 2. TIGHTEST KEYWORD WINDOW (NEW)                            │
  │ │    - find_tightest_keyword_window(content, keywords)        │
  │ │    - Sliding window finds minimal span covering keywords    │
  │ │    - More robust than sentence splitting (handles Dr., vs.) │
  │ │    - If found → method="keyword_window"                     │
  │ │                                                             │
  │ │ 3. SENTENCE FALLBACK                                        │
  │ │    - Split on paragraphs, headers, sentences                │
  │ │    - Score each sentence by keyword count                   │
  │ │    - Try sentence pairs and triplets                        │
  │ │    - If found → method="sentence_fallback"                  │
  │ └────────────────────────────────────────────────────────────┘
  │
  ▼
expand_to_sentence_bounds(content, span_start, span_end): ← NEW
  │
  │ Expand keyword window to sentence boundaries for readability
  │ Look for .!? within max_expand chars in each direction
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
  source_url=source["url"],
  span_start=start,           ← NEW
  span_end=end,               ← NEW
  keywords_matched=matched,   ← NEW
  verification_method=method, ← NEW
  failure_reason=reason,      ← NEW
  failure_details=details     ← NEW
)
```

### Tightest Keyword Window Algorithm (NEW)

```
find_tightest_keyword_window(content, keywords, max_window_chars=500)
══════════════════════════════════════════════════════════════

PURPOSE: Find minimal span covering most keywords without
         relying on sentence splitting (which breaks on
         "Dr.", "vs.", "Inc.", etc.)

ALGORITHM:
  1. Find all positions of each keyword in content
  2. Build list of (position, keyword, end_pos) tuples
  3. Sort by position
  4. Sliding window to find tightest coverage:
     │
     │ for i in range(n):
     │   window_start = positions[i]
     │   covered_keywords = set()
     │
     │   for j in range(i, n):
     │     if window_end - window_start > max_window_chars:
     │       break
     │     covered_keywords.add(keyword[j])
     │
     │     if coverage > best_coverage:
     │       best_window = (start, end, covered)
     │
  5. Return (text, span_start, span_end, keywords_matched, coverage_ratio)

EXAMPLE:
  Content: "In 2024, Model X achieved 150ms latency under load."
  Keywords: ["Model", "X", "150ms", "latency"]

  Positions: [(9, "Model"), (15, "X"), (25, "150ms"), (31, "latency")]

  Best window: (9, 38) covering all 4 keywords
  → "Model X achieved 150ms latency"
```

### Post-Extraction Verification (NEW)

```
AFTER EXTRACTION:
══════════════════════════════════════════════════════════════

verify_span(extraction, source_content):
  │
  │ PURPOSE: Deterministic reverification that extracted text
  │          actually exists at the recorded span position.
  │
  │ CHECKS:
  │   - span_start >= 0 and span_end > span_start
  │   - extracted_text exists
  │   - span_end <= len(source_content)
  │   - extracted_text in span_text OR span_text in extracted_text
  │
  │ If verify_span() returns False:
  │   extraction.status = "span_mismatch"
  │   extraction.failure_reason = "span_verification_failed"
  │
  ▼
Return verified or rejected extraction
```

### Cleanup Guards (NEW)

```
verify_and_apply_cleanup(original, cleaned):
══════════════════════════════════════════════════════════════

PURPOSE: Prevent cleanup from removing semantically important content.

GUARDS:
  1. NEGATION TOKENS - reject if cleanup removes:
     {"not", "never", "no", "without", "except", "unless", "don't", ...}

  2. QUALIFIER TOKENS - reject if cleanup removes:
     {"only", "just", "approximately", "about", "up to", "at least", ...}

  3. NUMBER PROTECTION - reject if cleanup removes:
     Any numbers with units: 10%, 200ms, $1.2B, 50k, etc.

If any guard triggers → return original text unchanged
Otherwise → return cleaned text
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

## 7. Synthesis Validation Flow

Post-synthesis checks to catch hallucination and citation errors (NEW - Phase 8).

### No-New-Facts Validation (`pipeline_v2.py`)

```
validate_no_new_facts(prose, facts) → List[str]
══════════════════════════════════════════════════════════════

PURPOSE: Catch when synthesis introduces claims not in the source facts.
         This is a key anti-hallucination gate.

FLOW:
  prose: "The framework achieves 2.5x speedup [1] with 99.9% accuracy..."
  facts: [Extraction with "2.2x faster", Extraction with "95% accuracy"]
  │
  ▼
STEP 1: Extract numbers from prose
  │ Pattern: r'\d+(?:\.\d+)?(?:%|ms|k|M|GB|MB|TB|x|X)?'
  │ Result: {"2.5x", "99.9%"}
  │
  ▼
STEP 2: Extract numbers from all facts
  │ Result: {"2.2x", "95%"}
  │
  ▼
STEP 3: Find new numbers (not in facts)
  │ new_numbers = prose_numbers - fact_numbers
  │ Filter out citation markers ([1], [2], etc.)
  │ Result: {"2.5x", "99.9%"} ← VIOLATIONS!
  │
  ▼
STEP 4: Check superlatives without attribution
  │ Superlatives: "best", "fastest", "most", "only", "first", etc.
  │ For each superlative in prose:
  │   - Find within 50 chars of a citation marker?
  │   - If not → flag as unattributed superlative
  │
  ▼
Return list of violations:
  ["New number 2.5x not found in cited facts",
   "New number 99.9% not found in cited facts"]
```

### Citation Validation (`pipeline_v2.py`)

```
validate_section_citations(section: ThemedSection) → List[str]
══════════════════════════════════════════════════════════════

PURPOSE: Verify that citation anchors (numbers, proper nouns)
         actually appear in the cited fact.

FLOW:
  section.prose: "Model X achieves 150ms latency [1]."
  section.facts[0].extracted_text: "Model Y has 200ms response time"
  │
  ▼
For each citation in section.citations:
  │
  ├── Check fact exists
  │   if citation.fact_index >= len(facts):
  │     violation: "Citation [1] references non-existent fact"
  │
  ├── Check fact has text
  │   if not fact.extracted_text:
  │     violation: "Citation [1] references fact with no text"
  │
  └── Check anchor match
      │ 1. Find sentence containing citation marker
      │ 2. Extract anchors from sentence:
      │    - Numbers: r'\d+(?:\.\d+)?(?:%|ms|k|M)?'
      │    - Proper nouns: r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'
      │ 3. For each anchor:
      │    - Is it in fact.extracted_text?
      │    - If not → violation
      │
      │ Example:
      │   Sentence: "Model X achieves 150ms latency [1]."
      │   Anchors: ["Model X", "150ms"]
      │   Fact: "Model Y has 200ms response time"
      │   → violation: "Anchor 'Model X' not in cited fact"
      │   → violation: "Anchor '150ms' not in cited fact"
  │
  ▼
Return list of violations
```

### Integration in Synthesis

```
synthesize_theme(theme_group, topic, llm_call):
  │
  │ ... generate prose with citations ...
  │
  ▼
VALIDATION STEP (NEW):
  │
  │ violations = validate_no_new_facts(prose, facts)
  │ violations += validate_section_citations(section)
  │
  │ if violations:
  │   logger.warning(f"Synthesis violations for {theme}: {violations}")
  │   # Note: Currently logs only; could block in strict mode
  │
  ▼
Return ThemedSection (with logged violations)
```

---

## 8. Council Validation Flow

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

## 9. Artifacts Flow

Run artifact storage for reproducibility and debugging (NEW - Phase 8).

### Overview (`artifacts.py`)

```
PURPOSE: Store immutable records of pipeline runs for:
  - Replaying runs with different prompts
  - Diffing runs to understand changes
  - Attributing regressions to specific prompt changes

ARTIFACT STRUCTURE:
══════════════════════════════════════════════════════════════

RunArtifacts
├── run_id: str           # UUID[:8]
├── timestamp: str        # ISO format
├── query: str            # Research topic
├── config_hash: str      # SHA256 of config
├── prompt_versions: Dict[str, str]  # prompt_name → SHA256[:8]
│
├── sources: List[SourceArtifact]
│   ├── url: str
│   ├── title: str
│   ├── content_hash: str   # SHA256[:16] of content
│   └── content_length: int
│
├── pointers: List[PointerArtifact]
│   ├── source_id: str
│   ├── keywords: List[str]
│   ├── micro_quote: Optional[str]
│   └── context: str
│
├── extractions: List[ExtractionArtifact]
│   ├── pointer_source_id: str
│   ├── status: str
│   ├── extracted_text: Optional[str]
│   ├── match_score: float
│   ├── span_start, span_end: int
│   ├── keywords_matched: List[str]
│   ├── verification_method: str
│   └── failure_reason: Optional[str]
│
├── dedup_decisions: List[DedupDecision]
│   ├── kept_index: int
│   ├── removed_index: int
│   ├── similarity: float
│   └── reason: str
│
├── arrangement: Dict
├── synthesis_themes: List[str]
├── synthesis_violations: List[str]  # From validation
│
└── report_hash: str      # SHA256 of final output
```

### Prompt Versioning

```
compute_prompt_versions() → Dict[str, str]
══════════════════════════════════════════════════════════════

PURPOSE: Track which prompt versions were used in a run.
         Enables attributing regressions to specific changes.

PROMPTS TRACKED:
  POINTER_PROMPT         → e.g., "5919c7fa"
  CLEANUP_PROMPT         → e.g., "d8fd00f9"
  ARRANGER_PROMPT        → e.g., "9b061b8e"
  THEME_SYNTHESIS_PROMPT → e.g., "f3db638a"
  EXECUTIVE_SUMMARY_PROMPT
  ANALYSIS_PROMPT
  CONCLUSION_PROMPT

COMPUTATION:
  version = SHA256(prompt_text).hexdigest()[:8]
```

### Pipeline Integration

```
run_pipeline_v2(sources, topic, ..., artifacts_dir, checkpoint_dir):
══════════════════════════════════════════════════════════════

AT START:
  │
  │ if artifacts_dir:
  │   artifacts = create_run_artifacts(topic)
  │
  │   # Record sources with content hashes
  │   for src_id, src in sources.items():
  │     content = src.get("content", "")
  │     artifacts.sources.append(SourceArtifact(
  │       url=src.get("url", ""),
  │       title=src.get("title", ""),
  │       content_hash=SHA256(content)[:16],
  │       content_length=len(content)
  │     ))
  │
  ▼

DURING PIPELINE:
  │ (Checkpoints captured as before)
  │
  ▼

AT END:
  │
  │ if artifacts_dir:
  │   artifacts.report_hash = SHA256(final_report)[:16]
  │   artifacts.verified_count = count(verified extractions)
  │   artifacts.total_extracted = total extractions
  │
  │   save_run_artifacts(artifacts, artifacts_dir)
  │   # Creates: run_{run_id}_{date}.json
  │
  │ if checkpoint_dir:
  │   save checkpoints to checkpoint_{timestamp}.json
```

### Checkpoint Persistence

```
CHECKPOINTS SAVED TO DISK:
══════════════════════════════════════════════════════════════

File: checkpoint_YYYYMMDD_HHMMSS.json

{
  "pre_dedup": [
    {
      "extracted_text": "...",
      "source_id": "src_001",
      "match_score": 0.85,
      "span_start": 123,
      "span_end": 456,
      ...
    },
    ...
  ],
  "post_dedup": [...],
  "pre_arrangement": [...],
  "post_arrangement": {
    "themes": [
      {"name": "Theme 1", "fact_ids": [1, 4, 7]},
      ...
    ],
    "excluded_ids": [3, 6, 8]
  }
}
```

### Diffing Runs

```
diff_prompt_versions(old_artifacts, new_artifacts) → Dict
══════════════════════════════════════════════════════════════

PURPOSE: Compare two runs to identify what changed.

USAGE:
  old = load_run_artifacts(Path("run_abc_2026-01-12.json"))
  new = load_run_artifacts(Path("run_xyz_2026-01-13.json"))

  changes = diff_prompt_versions(old, new)
  # Returns: {
  #   "POINTER_PROMPT": ("5919c7fa", "a1b2c3d4"),
  #   "THEME_SYNTHESIS_PROMPT": ("f3db638a", "e5f6g7h8")
  # }

  # If verified_count dropped, blame the changed prompts
```

---

## 10. Evaluation Flow

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
| `find_best_match` | `pointer_extract.py` | Micro-quote/keyword matching |
| `find_tightest_keyword_window` | `pointer_extract.py` | Minimal keyword span (NEW) |
| `expand_to_sentence_bounds` | `pointer_extract.py` | Expand span for readability (NEW) |
| `verify_span` | `pointer_extract.py` | Post-extraction verification (NEW) |
| `verify_and_apply_cleanup` | `pointer_extract.py` | Cleanup with guards (NEW) |
| `deduplicate_llm` | `pipeline_v2.py` | Semantic deduplication |
| `arrange_facts` | `pipeline_v2.py` | Theme grouping |
| `synthesize_theme` | `pipeline_v2.py` | Per-theme prose |
| `validate_no_new_facts` | `pipeline_v2.py` | Anti-hallucination check (NEW) |
| `validate_section_citations` | `pipeline_v2.py` | Citation anchor validation (NEW) |
| `render_report` | `render.py` | HTML generation |
| `council_vote_on_brief` | `council.py` | Multi-model voting |
| `evaluate_report` | `evaluation.py` | Post-hoc quality check |
| `create_run_artifacts` | `artifacts.py` | Initialize run record (NEW) |
| `save_run_artifacts` | `artifacts.py` | Persist artifacts to disk (NEW) |
| `compute_prompt_versions` | `artifacts.py` | Hash all prompts (NEW) |
| `diff_prompt_versions` | `artifacts.py` | Compare runs (NEW) |
