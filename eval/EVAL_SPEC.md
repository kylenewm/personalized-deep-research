# Evaluation Framework Spec

## Overview

Two-stage LLM-based evaluation for deep-research pipeline quality.

```
┌─────────────────────────────────────────────────────────┐
│                    UPSTREAM EVAL                        │
│  Query + Sources → Extract → Facts                      │
│                              ↓                          │
│                    Evaluate: quality, coverage, dupes   │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│                   DOWNSTREAM EVAL                       │
│  Facts → Arrange + Synthesize → Report                  │
│                                 ↓                       │
│                    Evaluate: citations, synthesis       │
└─────────────────────────────────────────────────────────┘
```

## Metrics

### Upstream (Fact Extraction)

| Metric | Target | Hard Fail | Formula |
|--------|--------|-----------|---------|
| `avg_fact_quality` | ≥3.5 | <2.0 | mean(fact_scores) |
| `avg_theme_coverage` | ≥3.5 | - | mean(theme_scores) |
| `duplicate_rate` | ≤2% | >20% | duplicates / total_facts |
| `low_quality_rate` | ≤10% | - | facts_scoring_≤2 / total |
| `match_score_avg` | ≥0.8 | - | mean(match_scores) [proxy] |

### Downstream (Report Generation)

| Metric | Target | Hard Fail | Formula |
|--------|--------|-----------|---------|
| `avg_citation_accuracy` | ≥4.0 | - | mean(citation_scores) |
| `avg_synthesis_quality` | ≥3.5 | - | mean(synthesis_scores) |
| `uncited_rate` | ≤5% | >30% | uncited / total_sentences |

## Scoring Rubrics

### Fact Quality (1-5)
- **5**: Specific, quantified, expert would cite
- **4**: Useful detail with some specificity
- **3**: Relevant but generic
- **2**: Tangentially related or vague
- **1**: Fluff, marketing, irrelevant

### Theme Coverage (1-5)
- **5**: Could write detailed expert section
- **4**: Good coverage, minor gaps
- **3**: Adequate overview, missing depth
- **2**: Sparse, surface-level only
- **1**: Theme barely addressed

### Citation Accuracy (1-5)
- **5**: Citations directly support claim
- **4**: Support with minor interpretation
- **3**: Partial support
- **2**: Weak connection
- **1**: Hallucination

### Synthesis Quality (1-5)
- **5**: Expert-level, adds insight
- **4**: Good, clear, accurate
- **3**: Adequate, restates facts
- **2**: Awkward, loses meaning
- **1**: Misrepresents facts

## Test Modes

| Mode | Facts | Queries | When | Est. Cost |
|------|-------|---------|------|-----------|
| Mini | 15 | 1 | Every PR | ~$0.05 |
| Medium | 50 | 1-2 | Medium changes | ~$0.10 |
| Full | 50×3 | 3 | Large changes / weekly | ~$0.30 |

## Gold Datasets

| Dataset | Sources | Report | Status |
|---------|---------|--------|--------|
| `agentic_coding_2026.json` | 58 | ✅ | Ready |
| `latest_research.json` | 160 | ❌ | Upstream only |
| `process_management.json` | TBD | TBD | Pending |

## Failure Handling

| Condition | Action |
|-----------|--------|
| Hard fail threshold crossed | FAIL build, block merge |
| Below target but above fail | WARN, flag for review |
| At or above target | PASS |

## Implementation

### Files
```
eval/
├── EVAL_SPEC.md          # This file
├── prompts/
│   ├── upstream_eval.txt   # Fact quality + coverage prompt
│   └── downstream_eval.txt # Citation + synthesis prompt
└── run_eval.py           # TODO: LLM-based evaluator
```

### Usage (planned)
```bash
# Run mini eval (1 query, 15 facts)
python eval/run_eval.py --mode mini

# Run full eval (3 queries)
python eval/run_eval.py --mode full

# Run on specific dataset
python eval/run_eval.py tests/fixtures/gold_queries/agentic_coding_2026.json
```

## Deferred (v2)

- [ ] LLM source verification (expensive, ~500 chars per fact)
- [ ] Perfect facts generation for downstream isolation
- [ ] Human labeling pipeline for ground truth
- [ ] Trend tracking over time
