# Deep Research

Research agent with anti-hallucination architecture. The LLM never writes factual content—it points to what to extract, and code pulls verbatim text from sources.

**[Example Report: Voice AI Observability](https://kylenewm.github.io/personalized-deep-research/reports/observability_voice_agents_report.html)**

Green text = extracted verbatim from sources. Gray text = AI-written transitions.

---

## How It Works

```
Query
  ↓
Research Brief (defines scope, themes)
  ↓
Web Search (Tavily) → 20-200 sources
  ↓
Pointer Extraction
  LLM outputs keywords → Code fuzzy-matches → Verbatim extraction
  ↓
Deduplication (semantic + cross-batch)
  ↓
Quality Filter (rejects tables, artifacts, fragments)
  ↓
Theme Arrangement (groups facts, excludes redundant)
  ↓
Synthesis (LLM writes transitions, facts stay locked)
  ↓
HTML Report
```

---

## Quick Start

```bash
git clone https://github.com/kylenewm/personalized-deep-research.git
cd personalized-deep-research

python -m venv venv
source venv/bin/activate
pip install -e .

cp .env.example .env
# Add: OPENAI_API_KEY, TAVILY_API_KEY
```

---

## Usage

```bash
python scripts/run_research.py "your research question"
```

---

## Limitations

- Search quality depends on Tavily API
- ~6% of synthesized sentences can't be traced to a specific fact
- Fuzzy matching works better with distinctive keywords

---

## Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Technical details, configuration options
- [INVARIANTS.md](./INVARIANTS.md) — System contracts
