# Deep Research Agent

AI-powered research agent that searches the web, gathers sources, and generates verified research reports. Built on top of LangGraph's open source deep research project, with added quote verification, source quality filtering, and multi-model validation. 

## What it does

1. Takes a research query
2. Searches the web using Tavily
3. Extracts and analyzes content from sources
4. Verifies claims against source material
5. Generates a research report with citations

## How it works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                      │
│                    "What are the latest AI models?"                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. RESEARCH PLANNING                                │
│                                                                              │
│   • Analyzes the query                                                       │
│   • Creates a research brief with sub-questions to investigate               │
│   • Validates the plan (optional multi-model consensus)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          2. WEB RESEARCH                                     │
│                                                                              │
│   • Spawns multiple researcher agents in parallel                            │
│   • Each researcher searches the web via Tavily API                          │
│   • Extracts content from relevant pages                                     │
│   • Collects sources with full text for verification                         │
│                                                                              │
│   Sources stored:                                                            │
│   ┌──────────────────────────────────────────────────────────┐              │
│   │  { url, title, content, query, timestamp }               │              │
│   └──────────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          3. FINDINGS VALIDATION                              │
│                                                                              │
│   • Reviews all gathered research                                            │
│   • Validates findings (optional multi-model consensus)                      │
│   • Identifies gaps and contradictions                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       4. QUOTE VERIFICATION                                  │
│                          (Trust Pipeline)                                    │
│                                                                              │
│   Before writing the report, verify that quotes actually exist in sources:   │
│                                                                              │
│   For each quote:                                                            │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  1. Find the quote in source text (substring match)                │    │
│   │  2. Calculate similarity score (Jaccard similarity ≥ 0.8)          │    │
│   │  3. Mark as PASS or FAIL                                           │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   Only PASS quotes are included in the "Verified Findings" section           │
│                                                                              │
│   ⚠️  Requires LangGraph Store. Without it, falls back to post-report        │
│       verification using embeddings.                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        5. REPORT GENERATION                                  │
│                                                                              │
│   Two-step generation:                                                       │
│   1. Write "Verified Findings" section using only PASS quotes                │
│   2. Write main report body with citations                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        6. CLAIMS VERIFICATION                                │
│                           (Health Check)                                     │
│                                                                              │
│   After the report is written, double-check claims:                          │
│                                                                              │
│   1. Extract factual claims from the report                                  │
│   2. For each claim, find the most relevant source using:                    │
│      • Embedding similarity (semantic matching)                              │
│      • Entity overlap (names, dates, numbers)                                │
│   3. Ask LLM: "Does this source support this claim?"                         │
│   4. Report confidence score and any unsupported claims                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FINAL REPORT                                      │
│                                                                              │
│   • Research findings with citations                                         │
│   • Verified quotes from sources                                             │
│   • Confidence score for claims                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Two Verification Systems

The agent has two ways to verify information:

| System | When it runs | How it works | Purpose |
|--------|--------------|--------------|---------|
| **Quote Verification** | Before report | Checks if quotes exist in source text (exact matching) | Prevent hallucinated quotes |
| **Claims Verification** | After report | Checks if claims are supported by sources (semantic matching) | Confidence score on final report |

**Quote Verification** is more strict (requires exact text match). It runs before the report is written.

**Claims Verification** is more flexible (uses embeddings to find relevant passages). It runs after to give an overall confidence score.

## Example Output

Here's a sample from a real research run on multi-agent AI frameworks:

**Query:** *What are the best multi-agent orchestration frameworks for AI coding assistants in 2026? Compare LangGraph, CrewAI, AutoGen...*

**Result:** 73 verified facts across 5 themes in ~4 minutes

| Theme | Sample Finding |
|-------|----------------|
| Adoption & Scale | CrewAI: 150+ enterprise customers, 100,000+ daily agent executions |
| Architecture | LangGraph: Pregel-inspired distributed model with checkpoint-based state |
| Performance | AutoGen: 82.9% task completion in SWE-bench evaluations |
| Human-in-Loop | All frameworks support breakpoints; CrewAI adds role-based approval gates |
| Enterprise Ready | LangGraph integrates with LangSmith for observability at scale |

**Sample prose with citations:**
> By mid-2025, CrewAI demonstrated significant enterprise adoption with over 150 enterprise customers and more than 100,000 daily agent executions [1][4]. Meanwhile, LangGraph's Pregel-inspired architecture enables distributed state management across agent nodes [7], while AutoGen achieved 82.9% on SWE-bench coding benchmarks [12].

Full dataset: `tests/fixtures/gold_queries/agentic_coding_2026.json`

## Setup

```bash
# Clone and install
git clone https://github.com/kylenewm/personalized-deep-research.git
cd personalized-deep-research

python -m venv venv
source venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm

# Configure API keys
cp .env.example .env
# Add your keys to .env
```

**Required API keys:**
- `OPENAI_API_KEY` - For LLM calls
- `TAVILY_API_KEY` - For web search

**Optional:**
- `ANTHROPIC_API_KEY` - For multi-model validation
- `LANGSMITH_API_KEY` - For tracing/debugging

## Usage

```bash
# Full pipeline test (~8 min)
python scripts/test_e2e_quick.py "What is quantum computing?"

# Test verification logic with mock data (~10s)
python scripts/mock_verification_test.py
```

## Staged Testing

Test each pipeline stage independently for faster iteration:

```bash
# Test brief generation only (~30s)
python scripts/test_brief.py "What are the latest AI models?"

# Test research phase (~2-3 min)
python scripts/test_research.py "What is quantum computing?"

# Test report generation with mock data (~30s)
python scripts/test_report.py --mock

# Test report from saved research state
python scripts/test_report.py --state test_research_state.json
```

Each staged test outputs a detailed markdown report showing exactly what happened.

## Configuration

Edit `src/open_deep_research/configuration.py`:

| Option | Default | Description |
|--------|---------|-------------|
| `test_mode` | `False` | Reduces iterations for faster testing |
| `use_council` | `True` | Enable multi-model validation |
| `use_claim_verification` | `True` | Run post-report claims check |
| `use_tavily_extract` | `True` | Use Tavily Extract for cleaner content |
| `max_total_sources` | `200` | Cap on sources to prevent token explosion |
| `min_source_content_length` | `500` | Filter out thin/low-quality content |

## Project Structure

```
src/open_deep_research/
├── graph.py              # Main LangGraph workflow
├── state.py              # State definitions
├── configuration.py      # Config options
├── verification.py       # Claims verification (post-report)
├── nodes/
│   ├── brief.py          # Research planning
│   ├── supervisor.py     # Coordinates parallel researchers
│   ├── researcher.py     # Web research agent
│   ├── findings.py       # Findings validation
│   ├── extract.py        # Quote extraction
│   ├── verify.py         # Quote verification (pre-report)
│   └── report.py         # Report generation
└── logic/
    ├── document_processing.py  # Text chunking with spaCy
    └── sanitize.py             # HTML cleaning

scripts/
├── test_e2e_quick.py           # Full pipeline test (~8 min)
├── test_brief.py               # Test brief generation (~30s)
├── test_research.py            # Test research phase (~2-3 min)
├── test_report.py              # Test report generation (~30s)
├── mock_verification_test.py   # Test verification with mock data
├── staged_config.py            # Shared minimal config for staged tests
└── generate_run_report.py      # Generate analysis from run state
```

## Output

After each run, you get:
- `run_report_<timestamp>.md` - Human-readable analysis of what happened
- `run_state_<timestamp>.json` - Raw state for debugging
- `metrics/metrics_<timestamp>.json` - Structured metrics (from staged tests)

## Metrics System

The staged tests output structured JSON metrics for each run:

```json
{
  "run_id": "20260104_192816",
  "query": "What are AI agents?",
  "total_duration_seconds": 138.1,
  "stages": [
    {"name": "full_pipeline", "duration_seconds": 138.1, "success": true}
  ],
  "sources_stored": 25,
  "notes_generated": 0,
  "report_length_chars": 16298,
  "snippets_extracted": 100,
  "snippets_verified_pass": 45
}
```

Use metrics to track optimization progress across runs.

## Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Complete technical breakdown of the system
  - All pipeline stages explained in detail
  - Data flow diagrams
  - State management
  - Anti-hallucination system
  - Key files reference

- **[docs/EVAL_OVERVIEW.md](./docs/EVAL_OVERVIEW.md)** - Evaluation system guide
  - Brief, upstream, downstream metrics
  - Running evals on gold datasets
  - Quality thresholds and baselines

## Next Steps

- [ ] **Personalization layer** - Let users configure speed vs depth vs cost tradeoffs
- [ ] **Preset profiles** - Quick settings like "fast & cheap", "thorough", "balanced"
- [ ] **Better source ranking** - Prioritize high-quality sources over quantity
- [ ] **Streaming output** - Show progress as research happens instead of waiting for the end
- [ ] **Memory/context** - Remember past research to build on previous findings
- [ ] **Multi-format export** - PDF, DOCX, etc. in addition to markdown
