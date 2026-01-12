# Evaluation System Overview

## Purpose

Measure pipeline quality at three stages:
1. **Brief** - Does query→brief transformation preserve intent?
2. **Upstream** - Are extracted facts good enough?
3. **Downstream** - Is the final report well-cited and synthesized?

## Running Evals

```bash
# Single dataset
python eval/run_eval.py tests/fixtures/gold_queries/agentic_coding_2026.json

# With options
python eval/run_eval.py <dataset> --mode mini    # 15 facts (fast, ~$0.05)
python eval/run_eval.py <dataset> --mode medium  # 50 facts (~$0.10)
python eval/run_eval.py <dataset> --mode full    # all facts (~$0.15)
python eval/run_eval.py <dataset> --limit 25     # custom fact count

# All gold datasets
python eval/run_eval.py --all

# Cost estimate only
python eval/run_eval.py --all --dry-run
```

## Metrics & Thresholds

### Brief Eval
| Metric | Target | Meaning |
|--------|--------|---------|
| preservation | ≥4/5 | Did brief keep all query specifics? |
| dilution | ≥4/5 | Did brief avoid generalizing? |
| assumptions | ≥4/5 | Did brief avoid adding constraints? |

**Status:** GOOD/WARN/FAIL based on scores

### Upstream Eval
| Metric | Target | Hard Fail |
|--------|--------|-----------|
| avg_fact_quality | ≥3.5 | <2.0 |
| avg_theme_coverage | ≥3.5 | - |
| duplicate_rate | ≤15% | >20% |
| low_quality_rate | ≤10% | - |
| match_score_avg | ≥0.8 | - |

**Scoring (1-5):**
- 5 = Informative to expert in field
- 3 = Informative to someone familiar with topic
- 1 = Fluff, vague, or not useful

### Downstream Eval
| Metric | Target | Hard Fail |
|--------|--------|-----------|
| avg_citation_accuracy | ≥4.0 | - |
| avg_synthesis_quality | ≥3.5 | - |
| uncited_rate | ≤5% | >30% |

## Gold Datasets

Located in `tests/fixtures/gold_queries/`

| Dataset | Query Focus | Facts | Has Report | Notes |
|---------|-------------|-------|------------|-------|
| `agentic_coding_2026.json` | Multi-agent frameworks (LangGraph, CrewAI, AutoGen) | ~73 | ✅ Yes | Primary eval dataset |
| `latest_research.json` | Voice agent orchestration | ~42 | ❌ No | Upstream only |
| `baseline_2026-01-12.json` | Baseline metrics snapshot | - | - | Reference only |

### Gold Dataset Structure

```json
{
  "query": "Original user query",
  "research_brief": "Generated brief (for brief eval)",
  "hybrid_report": {
    "sections": [
      {
        "theme": "Theme Name",
        "prose": "Synthesized prose with [N] citations..."
      }
    ],
    "footnotes": [
      {
        "id": 1,
        "extracted_text": "The actual fact",
        "source_url": "https://...",
        "source_domain": "example.com",
        "match_score": 0.92,
        "theme": "Theme Name"
      }
    ]
  }
}
```

### Creating New Gold Datasets

1. Run pipeline on a query with `save_gold=True` or manually save output
2. Verify the output looks reasonable
3. Run eval to check metrics
4. Add to `gold_queries/` directory

## Files

```
eval/
├── run_eval.py          # Main CLI runner
├── metrics.py           # EvalResult dataclass, thresholds
├── llm.py               # OpenAI wrapper (loads .env)
└── prompts/
    ├── brief_eval.txt      # Query→brief preservation check
    ├── upstream_eval.txt   # Fact quality scoring
    └── downstream_eval.txt # Citation accuracy scoring
```

## Example Output

```
============================================================
EVALUATION RESULT: agentic_coding_2026.json
============================================================

BRIEF (GOOD):
  Preservation:  5.0/5 (target: ≥4)
  Dilution:      5.0/5 (target: ≥4, higher=less dilution)
  Assumptions:   5.0/5 (target: ≥4, higher=fewer assumptions)

UPSTREAM (PASS):
  Fact quality:    4.27 (target: ≥3.5)
  Theme coverage:  4.20 (target: ≥3.5)
  Duplicate rate:  0.0% (target: ≤15%)
  Low quality:     0.0% (target: ≤10%)
  Match score:     0.91 (target: ≥0.8)

DOWNSTREAM (WARN (uncited_rate)):
  Citation accuracy: 4.80 (target: ≥4.0)
  Synthesis quality: 4.30 (target: ≥3.5)
  Uncited rate:      6.4% (target: ≤5%)

OVERALL: WARN
Facts: 15, Themes: 5
```

## Current Baseline (2026-01-12)

| Metric | Value | Status |
|--------|-------|--------|
| Brief preservation | 5/5 | ✅ |
| Brief dilution | 5/5 | ✅ |
| Fact quality | 4.27 | ✅ |
| Theme coverage | 4.20 | ✅ |
| Duplicate rate | 0% | ✅ |
| Citation accuracy | 4.80 | ✅ |
| Uncited rate | ~7% | ⚠️ |

## Known Limitations

1. **Eval variability** - LLM scoring has ~5-10% run-to-run variance
2. **Uncited rate** - Some sentences legitimately don't need citations (transitions, conclusions)
3. **Cost** - Full eval on large datasets can cost $0.15-0.30

## Future Improvements

- [ ] Source verification (check if facts actually appear in source)
- [ ] Cross-dataset consistency tracking
- [ ] Automated regression detection
