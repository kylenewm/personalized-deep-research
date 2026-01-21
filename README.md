# Deep Research

Research agent that extracts facts directly from source text. LLM points to content, code extracts verbatim.

---

## Example Reports

**[Voice AI Observability: Monitoring Agentic Systems](https://kylenewm.github.io/personalized-deep-research/reports/observability_voice_agents_report.html)** — 70 verified facts across 7 themes

Green text = extracted verbatim from sources. Gray text = AI-written transitions.

---

## Core Approach

```
LLM reads source → LLM outputs keywords → Code fuzzy-matches → Verbatim extraction
```

The LLM never writes factual content. It points to what to extract, and code does the extraction.

### Pipeline

```
Query
  ↓
Research Brief → Multi-model council validation (optional)
  ↓
Parallel Web Search (Tavily) → 20-200 sources
  ↓
Pointer Extraction (batched)
  │  LLM outputs: "source X, keywords: [A, B, C]"
  │  Code fuzzy-matches → extracts actual sentences
  ↓
LLM Deduplication (semantic, 0% false positive rate)
  ↓
Cross-batch Text Similarity (catches duplicates across batches)
  ↓
Quality Filter (rejects tables, artifacts, fragments)
  ↓
Arrangement (LLM groups by theme, excludes 40-60% redundant)
  ↓
Per-Theme Synthesis (LLM writes transitions, facts stay locked)
  ↓
Report (HTML)
```

### Verification Layers

| Layer | Method | What it catches |
|-------|--------|-----------------|
| Pointer extraction | Code fuzzy-match | Facts not in source |
| Quality filter | Regex + heuristics | Table rows, markdown artifacts |
| LLM dedup | Semantic comparison | Paraphrases and near-duplicates |
| Cross-batch dedup | Text similarity | Duplicates across extraction batches |
| Council validation | Multi-model voting | Fabricated names, impossible dates |
| Citation accuracy | LLM evaluation | Claims not supported by cited fact |

---

## Quick Start

```bash
git clone https://github.com/kylenewm/personalized-deep-research.git
cd personalized-deep-research

python -m venv venv
source venv/bin/activate
pip install -e .

cp .env.example .env
# Add API keys
```

**Required:** `OPENAI_API_KEY`, `TAVILY_API_KEY`

**Optional:** `ANTHROPIC_API_KEY` (council), `LANGSMITH_API_KEY` (tracing)

---

## Usage

```bash
# Full pipeline on a query
python scripts/run_research.py "your research question" --sources 100

# Quick test with reduced iterations
python scripts/test_e2e_quick.py "your research question"
```

---

## Configuration

Edit `src/open_deep_research/configuration.py`.

### Presets

| Preset | Cost | Description |
|--------|------|-------------|
| `fast` | ~$0.15 | 2 iterations, no councils |
| `balanced` | ~$0.25 | Brief context, no councils |
| `thorough` | ~$0.50 | All features except claim verification |

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_researcher_iterations` | 6 | Research loops (2 in test mode) |
| `max_total_sources` | 200 | Source cap |
| `max_concurrent_research_units` | 5 | Parallel sub-agents |
| `prefer_authoritative_sources` | True | Bias toward official docs |
| `use_council` | False | Multi-model brief validation |
| `use_safeguarded_generation` | True | Pointer extraction (recommended) |

---

## Outputs

| File | Contents |
|------|----------|
| `report_<timestamp>.html` | Formatted research report |
| `run_state_<timestamp>.json` | Full pipeline state (for re-runs) |

---

## Limitations

- **Search quality** — Results depend on Tavily API
- **~6% uncited sentences** — Some synthesized text can't be traced to a specific fact
- **API costs** — Large queries cost $1-5 (use `test_mode: true` for iteration)
- **Match score variability** — Fuzzy matching works better with distinctive keywords

---

## Documentation

| Document | Contents |
|----------|----------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Technical breakdown, data flow, configuration |
| [INVARIANTS.md](./INVARIANTS.md) | System contracts and safety rules |
