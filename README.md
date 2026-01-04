# Deep Research Agent

AI-powered research agent that searches the web, gathers sources, and generates verified research reports. This project signficantly expands off of LangGraph's open source deep research project. The next steps are optimizing the current codebase and then adding in personalization to filter for speed and cost among other things. 

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
# Run a research query
python scripts/test_e2e_quick.py "What is quantum computing?"

# Test verification logic with mock data
python scripts/mock_verification_test.py
```

## Configuration

Edit `src/open_deep_research/configuration.py`:

| Option | Default | Description |
|--------|---------|-------------|
| `test_mode` | `False` | Reduces iterations for faster testing |
| `use_council` | `True` | Enable multi-model validation |
| `use_claim_verification` | `True` | Run post-report claims check |
| `use_tavily_extract` | `True` | Use Tavily Extract for cleaner content |

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
├── test_e2e_quick.py           # Quick end-to-end test
├── mock_verification_test.py   # Test verification with mock data
└── generate_run_report.py      # Generate analysis from run state
```

## Output

After each run, you get:
- `run_report_<timestamp>.md` - Human-readable analysis of what happened
- `run_state_<timestamp>.json` - Raw state for debugging
