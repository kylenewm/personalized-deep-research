# Deep Research: Module Reference

Complete inventory of every file in the project.

---

## Source Code (`src/open_deep_research/`)

### Core Pipeline

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 14 | Package entry - exports `deep_researcher` |
| `graph.py` | 138 | **Main LangGraph construction** - composes all nodes into workflow |
| `state.py` | 233 | **State definitions** - AgentState, SupervisorState, ResearcherState, all TypedDict/Pydantic models |
| `configuration.py` | 743 | **All config options** - SearchAPI enum, presets, 40+ configurable fields |
| `models.py` | 34 | Shared models - FindingsReview, configurable_model init |

### Pipeline v2 (Safeguarded Generation)

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline_v2.py` | ~1760 | **Three-stage extraction pipeline** - batch extraction, dedup, arrangement, synthesis |
| `pointer_extract.py` | ~1010 | **Keyword matching extraction** - Pointer/Extraction dataclasses, fuzzy matching, quality filters |
| `synthesis.py` | 136 | **Prose synthesis** - connects verified facts with LLM-written transitions |
| `artifacts.py` | ~245 | **NEW: Run artifact storage** - prompt versioning, reproducibility, run diffing |

### Verification & Evaluation

| File | Lines | Purpose |
|------|-------|---------|
| `verification.py` | 682 | **Claim-level verification** - embedding similarity, entity extraction, multi-source verify |
| `evaluation.py` | 1229 | **Eval framework** - citation-first approach, ClaimResult, ClaimMetrics, post-hoc quality check |
| `council.py` | 349 | **Multi-model voting** - BriefReview, CouncilVote, CouncilVerdict, consensus calculation |

### Utilities & Rendering

| File | Lines | Purpose |
|------|-------|---------|
| `utils.py` | 1583 | **Everything else** - retry logic, domain filtering, source caching, Tavily tools, MCP integration |
| `prompts.py` | 602 | **All LLM prompts** - clarify, brief, supervisor, researcher, compression, final report |
| `render.py` | 617 | **HTML rendering** - converts HybridReport to HTML using templates |
| `export.py` | 747 | **PDF export** - WeasyPrint-based PDF generation with styling |
| `metrics.py` | ~100 | Internal metrics tracking |

---

## Pipeline Nodes (`src/open_deep_research/nodes/`)

Each node is a function that transforms state and returns a Command.

| File | Lines | Key Functions | Purpose |
|------|-------|---------------|---------|
| `__init__.py` | 52 | - | Re-exports all nodes |
| `store.py` | ~50 | `check_store` | **S02**: Trust Store gating - verifies Store availability |
| `clarify.py` | ~80 | `clarify_with_user` | Ask clarifying questions before research |
| `brief.py` | ~180 | `write_research_brief`, `validate_brief` | Generate research plan + Council 1 validation |
| `supervisor.py` | 295 | `supervisor`, `supervisor_tools` | **Research delegation** - plans strategy, spawns parallel researchers |
| `researcher.py` | 490 | `researcher`, `researcher_tools`, `compress_research` | **Web search execution** - Tavily/MCP tools, compression |
| `findings.py` | ~80 | `validate_findings` | **Council 2**: Fact-check findings before report |
| `safeguarded_report.py` | ~150 | `safeguarded_report_generation` | **Pipeline v2 entry** - calls run_pipeline_v2 |
| `extract.py` | ~120 | `extract_evidence` | **S03**: Deterministic quote mining from sources |
| `verify.py` | ~200 | `verify_evidence`, `verify_claims` | **S04**: Substring/Jaccard verification |
| `claim_gate.py` | ~100 | `claim_pre_check` | **Layer 3**: Soft-gate claims before report |
| `report.py` | ~200 | `final_report_generation` | **Legacy**: Generate final report from notes |
| `eval.py` | ~60 | `run_evaluation_node`, `should_run_evaluation` | Optional post-hoc evaluation |

---

## Logic Utilities (`src/open_deep_research/logic/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 10 | Re-exports |
| `document_processing.py` | 320 | Sentence chunking (spacy-based), document cleaning |
| `sanitize.py` | 150 | HTML/markdown stripping, quote extraction |

---

## Standalone Eval Framework (`eval/`)

Self-contained eval system that can run without the main pipeline.

| File | Lines | Purpose |
|------|-------|---------|
| `run_eval.py` | 590 | **Main runner** - runs brief/upstream/downstream evals |
| `metrics.py` | 170 | **Thresholds** - PASS/WARN/FAIL thresholds for each metric |
| `llm.py` | 65 | OpenAI wrapper with retry logic |
| `prompts/` | - | Eval-specific prompts (brief, upstream, downstream) |
| `EVAL_SPEC.md` | 126 | Evaluation specification |

---

## Scripts (`scripts/`)

### Main Entry Points

| Script | Purpose |
|--------|---------|
| `test_e2e_quick.py` | **Quick demo** - runs abbreviated pipeline with test_mode=True |
| `run_research.py` | Full pipeline run |
| `run_pipeline_standalone.py` | Run pipeline v2 directly from run_state |

### Sandboxes (Component Testing)

| Script | Tests |
|--------|-------|
| `prompt_sandbox.py` | Extraction quality - word count, artifacts, header rejection |
| `dedup_sandbox.py` | Deduplication accuracy - LLM semantic vs Jaccard |
| `arrangement_sandbox.py` | Theme grouping - coherence, exclusion rate |
| `citation_sandbox.py` | Citation rate - all facts cited? |
| `quality_sandbox.py` | Source quality - per-domain limits |
| `brief_sandbox.py` | Brief generation - preservation, dilution |
| `researcher_sandbox.py` | Search behavior testing |
| `search_sandbox.py` | Tavily search testing |

### Quality & Validation

| Script | Purpose |
|--------|---------|
| `run_all_sandboxes.py` | **Unified runner** - runs all sandboxes, reports status |
| `run_eval.py` | Run eval framework standalone |
| `audit_pipeline.py` | Full audit with detailed traces |
| `benchmark.py` | Performance benchmarking |

### Fixtures & Data

| Script | Purpose |
|--------|---------|
| `extract_fixtures.py` | Extract component fixtures from run_state |
| `prune_fixtures.py` | Fixture maintenance - stale detection, limits |
| `generate_run_report.py` | Generate execution summary from state |

### Re-run & Preview

| Script | Purpose |
|--------|---------|
| `resynthesis_test.py` | Re-run synthesis on existing data |
| `preview_report.py` | Preview report before export |
| `rerender_report.py` | Re-render HTML from saved data |

### Testing

| Script | Tests |
|--------|-------|
| `test_brief.py` | Brief generation in isolation |
| `test_research.py` | Research phase in isolation |
| `test_report.py` | Report generation in isolation |
| `test_pipeline_v2.py` | Pipeline v2 stages |
| `test_pointer_extract.py` | Pointer extraction |
| `test_extraction_prompt.py` | Extraction prompt variations |
| `test_cleanup.py` | LLM cleanup on garbage data |
| `test_cleanup_v2.py` | Cleanup v2 (contiguous substring) |
| `test_dedup_edge_cases.py` | Dedup edge cases |
| `test_render_fixes.py` | Render fixes without LLM |
| `test_presets.py` | Preset configurations |
| `test_voice_models.py` | Voice models query |
| `test_voice_search.py` | Voice search + pipeline v2 |
| `test_full_flow.py` | Full flow testing |
| `test_tavily_extract.py` | Tavily Extract API |
| `mock_verification_test.py` | Mock verification |
| `run_verification.py` | Verification layer standalone |
| `stress_test.py` | Pipeline stress testing |

### Utilities

| Script | Purpose |
|--------|---------|
| `demo_output.py` | Demo output formatting |
| `staged_config.py` | Fast iteration config |
| `sandbox_pipeline.py` | Fast pipeline v2 iteration |

---

## Test Fixtures (`tests/fixtures/`)

| Directory | Purpose | Limit |
|-----------|---------|-------|
| `gold_queries/` | Full pipeline outputs for eval | 10 |
| `extraction/` | Source content + expected extractions | 20 |
| `dedup/` | Labeled duplicate pairs | 15 |
| `synthesis/` | Theme + facts for synthesis testing | 20 |
| `arrangement/` | Facts for grouping tests | 15 |
| `sources/` | Domain distribution samples | 10 |
| `research_traces/` | Research iteration traces | - |

---

## Templates (`templates/`)

| File | Purpose |
|------|---------|
| `report.html` | Main HTML template for rendered reports |
| `report.css` | Styling for HTML reports |
| `pdf_template.html` | Template for PDF export |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, tool config |
| `.env.example` | Environment variable template |
| `.python-version` | Python version (3.11) |

---

## Documentation

### Core

| File | Lines | Purpose |
|------|-------|---------|
| `README.md` | 311 | Project overview, quick start |
| `ARCHITECTURE.md` | 858 | Technical breakdown, data flow |
| `STATE.md` | 572 | Development status, decisions |
| `INVARIANTS.md` | 95 | System contracts, safety rules |
| `WORKFLOW.md` | 125 | User workflow guide |
| `CLAUDE.md` | 191 | Development guidelines |

### Traces & Audits

| File | Lines | Purpose |
|------|-------|---------|
| `EXECUTION_TRACE.md` | 1388 | Step-by-step debug trace with prompts |
| `PIPELINE_TRACE.md` | 1096 | Pipeline execution trace |
| `QUALITY_AUDIT.md` | 564 | Quality assessment results |
| `LOG.md` | 1930 | Development history (append-only) |

### Design & Planning

| File | Lines | Purpose |
|------|-------|---------|
| `docs/EVAL_FRAMEWORK_PLAN.md` | 1772 | Eval system design doc |
| `docs/EVAL_OVERVIEW.md` | 168 | Eval framework overview |
| `docs/REFACTORING_SUMMARY.md` | 307 | Refactoring notes |
| `docs/DEEP_RESEARCH_V1.md` | 997 | V1 documentation (legacy) |

---

## Key Data Structures

### State Types (`state.py`)

```python
AgentState(MessagesState)
├── supervisor_messages: List[Message]
├── research_brief: str
├── notes: List[str]
├── raw_notes: List[str]
├── final_report: str
├── hybrid_report: dict  # HybridReport as dict
├── source_store: List[SourceRecord]  # All collected sources
├── evidence_snippets: List[EvidenceSnippet]
├── verification_result: VerificationResult
├── eval_result: dict
├── council_revision_count: int
├── findings_revision_count: int
├── verified_disabled: bool
└── claim_warnings: List[str]

SupervisorState(TypedDict)
├── supervisor_messages: List[Message]
├── research_brief: str
├── notes: List[str]
├── research_iterations: int
└── source_store: List[SourceRecord]

ResearcherState(TypedDict)
├── researcher_messages: List[Message]
├── tool_call_iterations: int
├── research_topic: str
├── compressed_research: str
└── source_store: List[dict]
```

### Pipeline v2 Types (`pipeline_v2.py`, `pointer_extract.py`)

```python
Pointer
├── source_id: str
├── keywords: List[str]
├── context: str
└── micro_quote: Optional[str]  # NEW: 8-15 word verbatim phrase for strict matching

Extraction
├── pointer: Pointer
├── status: str  # "verified", "partial", "not_found", "span_mismatch"
├── extracted_text: str
├── match_score: float
├── source_url: str
├── span_start: int  # NEW: Character position in source
├── span_end: int  # NEW: Character position in source
├── keywords_matched: List[str]  # NEW: Keywords actually found
├── verification_method: str  # NEW: "micro_quote", "keyword_window", "sentence_fallback"
├── failure_reason: Optional[str]  # NEW: "keywords_missing", "score_too_low", etc.
└── failure_details: Optional[dict]  # NEW: Structured diagnostics

HybridReport
├── title: str
├── executive_summary: str
├── sections: List[ThemedSection]
├── analysis: str
├── conclusion: str
├── excluded_facts: List[Extraction]
├── checkpoints: dict
└── verified_count: int (property)

ThemedSection
├── theme: str
├── prose: str
├── citations: List[Citation]
└── facts: List[Extraction]
```

### Artifact Types (`artifacts.py`)

```python
SourceArtifact
├── url: str
├── title: str
├── content_hash: str  # SHA256[:16] of content
└── content_length: int

PointerArtifact
├── source_id: str
├── keywords: List[str]
├── micro_quote: Optional[str]
└── context: str

ExtractionArtifact
├── pointer_source_id: str
├── status: str
├── extracted_text: Optional[str]
├── match_score: float
├── span_start: int
├── span_end: int
├── keywords_matched: List[str]
├── verification_method: str
└── failure_reason: Optional[str]

DedupDecision
├── kept_index: int
├── removed_index: int
├── similarity: float
└── reason: str  # "jaccard_duplicate", "same_source", etc.

RunArtifacts
├── run_id: str
├── timestamp: str
├── query: str
├── config_hash: str
├── prompt_versions: Dict[str, str]  # prompt_name -> sha256[:8]
├── sources: List[SourceArtifact]
├── pointers: List[PointerArtifact]
├── extractions: List[ExtractionArtifact]
├── dedup_decisions: List[DedupDecision]
├── arrangement: Dict[str, Any]
├── synthesis_themes: List[str]
├── synthesis_violations: List[str]
├── report_hash: str
├── verified_count: int
└── total_extracted: int
```

### Configuration (`configuration.py`)

```python
Configuration
├── test_mode: bool = False
├── search_api: SearchAPI = TAVILY
├── use_tavily_extract: bool = True
├── blocked_domains: List[str]
├── max_sources_per_domain: int = 3
├── max_researcher_iterations: int = 6
├── max_react_tool_calls: int = 10
├── max_concurrent_research_units: int = 5
├── max_total_sources: int = 200
├── use_council: bool = False
├── use_findings_council: bool = False
├── use_safeguarded_generation: bool = True
├── safeguarded_batch_size: int = 12
├── claim_pre_check: bool = True
├── run_evaluation: bool = False
├── prefer_authoritative_sources: bool = True
└── preset: str = None  # "fast", "balanced", "thorough"
```

---

## Key Functions (Phase 8)

### artifacts.py

| Function | Signature | Purpose |
|----------|-----------|---------|
| `compute_prompt_versions()` | `() -> Dict[str, str]` | Computes SHA256[:8] hashes for all prompts (POINTER_PROMPT, CLEANUP_PROMPT, ARRANGER_PROMPT, THEME_SYNTHESIS_PROMPT, EXECUTIVE_SUMMARY_PROMPT, ANALYSIS_PROMPT, CONCLUSION_PROMPT) |
| `save_run_artifacts()` | `(artifacts: RunArtifacts, path: Path) -> Path` | Serializes run artifacts to JSON file |
| `load_run_artifacts()` | `(path: Path) -> RunArtifacts` | Loads run artifacts from JSON file |
| `diff_prompt_versions()` | `(old: RunArtifacts, new: RunArtifacts) -> Dict[str, tuple]` | Compares prompt versions between runs, returns changed prompts |
| `create_run_artifacts()` | `(query: str, config: dict, run_id: str) -> RunArtifacts` | Creates new run artifact record with prompt versions |

### pointer_extract.py

| Function | Signature | Purpose |
|----------|-----------|---------|
| `find_tightest_keyword_window()` | `(content: str, keywords: List[str], max_window: int) -> Tuple[str, int, int, List[str], float]` | Sliding window algorithm to find minimal span covering most keywords |
| `expand_to_sentence_bounds()` | `(content: str, start: int, end: int, max_expand: int) -> Tuple[int, int]` | Expands character spans to sentence boundaries for readability |
| `verify_span()` | `(extraction: Extraction, source_content: str) -> bool` | Deterministic reverification: checks extracted_text matches span position |
| `find_best_match()` | `(keywords: List[str], source: str, min_score: float, micro_quote: str) -> Tuple[str, float, int, int, List[str], str]` | Returns 6-tuple with text, score, span offsets, matched keywords, and method |
| `verify_and_apply_cleanup()` | `(original: str, cleaned: str) -> Optional[str]` | Guards against semantic loss (negation/qualifier/number removal) |

**Constants:**
- `NEGATION_TOKENS`: Set of words that must not be removed (`not`, `never`, `no`, `without`, etc.)
- `QUALIFIER_TOKENS`: Set of precision words that must not be removed (`only`, `approximately`, `at least`, etc.)

### pipeline_v2.py

| Function | Signature | Purpose |
|----------|-----------|---------|
| `validate_no_new_facts()` | `(prose: str, facts: List[Extraction]) -> List[str]` | Checks prose doesn't introduce numbers/claims not in cited facts |
| `validate_section_citations()` | `(section: ThemedSection) -> List[str]` | Validates citation markers reference actual content in facts |
| `deduplicate_with_diversity()` | `(extractions: List[Extraction], max_per_source: int, thresholds: dict) -> List[Extraction]` | Dedup allowing top-K diverse facts per source (intra/cross-source thresholds) |
| `run_pipeline_v2()` | New params: `artifacts_dir: Path`, `checkpoint_dir: Path` | Supports artifact persistence (I10) and checkpoint saving (I11) |

---

## File Size Summary

```
Source Code:     ~14,000 lines (31 files)  # +artifacts.py, expanded pointer_extract/pipeline_v2
Scripts:         10,669 lines (47 files)
Tests:            2,500+ lines
Documentation:   12,000+ lines
Templates:          500+ lines
─────────────────────────────────
Total:          ~42,000 lines
```

---

## Phase 8 Changes Summary (2026-01-13)

**New module:** `artifacts.py` - Run artifact storage for reproducibility and prompt versioning.

**Major updates to `pointer_extract.py`:**
- Extraction dataclass expanded with span tracking (span_start, span_end), keyword matching (keywords_matched), verification method tracking, and structured failure diagnostics
- Pointer dataclass gained micro_quote field for strict substring matching
- New sliding window algorithm (find_tightest_keyword_window) for robust extraction
- Sentence boundary expansion for cleaner extractions
- Deterministic span reverification (verify_span)
- Cleanup guards against semantic loss (NEGATION_TOKENS, QUALIFIER_TOKENS)

**Major updates to `pipeline_v2.py`:**
- Synthesis validation: validate_no_new_facts() checks prose doesn't hallucinate
- Citation validation: validate_section_citations() verifies citations match facts
- Improved deduplication: deduplicate_with_diversity() with per-source limits
- New pipeline parameters: artifacts_dir and checkpoint_dir for persistence
