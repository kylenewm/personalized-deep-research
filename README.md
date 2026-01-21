# Deep Research

Research agent with anti-hallucination architecture. The LLM never writes factual content—it points to what to extract, and code pulls verbatim text from sources.

**[Example Report: Voice AI Observability](https://kylenewm.github.io/personalized-deep-research/reports/observability_voice_agents_report.html)**

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
For more detail: [ARCHITECTURE.md](./ARCHITECTURE.md) 
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



