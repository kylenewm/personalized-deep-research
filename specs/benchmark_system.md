# Benchmark System for Report Quality Evaluation

## Purpose

Test pipeline changes against fixed source datasets to measure quality impact without re-running expensive web search.

## Architecture

```
Gold Datasets (fixed sources)
         ↓
    run_pipeline_v2()
         ↓
    HybridReport output
         ↓
    evaluate_report()
         ↓
    Quality metrics JSON
```

## Gold Datasets (3)

| Dataset | Query Type | Sources | Purpose |
|---------|-----------|---------|---------|
| `agentic_coding_2026` | Technical/frameworks | 58 | Test framework comparison facts |
| `latest_research` | Overnight agents | 160 | Test practical technique extraction |
| `process_management` | Process supervision | ~60 | Test concrete implementation facts |

Each dataset contains:
```json
{
  "query": "...",
  "research_brief": "...",
  "sources": [
    {"url": "...", "title": "...", "content": "..."}
  ],
  "captured_at": "2026-01-12"
}
```

## Quality Metrics

### 1. Extraction Quality
| Metric | Target | Measurement |
|--------|--------|-------------|
| `specificity_rate` | > 30% | Facts with numbers/metrics |
| `vague_rate` | < 20% | Facts with "can be", "enables", etc. |
| `fluff_rate` | 0% | Marketing language |
| `avg_word_count` | 20-30 | Concise but complete |
| `over_40_words` | < 15% | Not too verbose |

### 2. Deduplication Quality
| Metric | Target | Measurement |
|--------|--------|-------------|
| `duplicate_pairs` | 0 | Manually flagged duplicates in output |
| `false_positive_rate` | 0% | Distinct facts incorrectly merged |

### 3. Source Quality
| Metric | Target | Measurement |
|--------|--------|-------------|
| `domain_diversity` | > 10 | Unique source domains |
| `authoritative_rate` | > 40% | Official docs, papers, established sources |
| `blog_rate` | < 30% | Medium, dev.to, personal blogs |

### 4. Synthesis Quality
| Metric | Target | Measurement |
|--------|--------|-------------|
| `citation_rate` | > 85% | Sentences with [N] citations |
| `uncited_rate` | < 15% | Sentences marked [u] |
| `theme_coverage` | 100% | All query aspects addressed |

## Benchmark Runner

```python
# scripts/benchmark.py

async def run_benchmark(dataset_path: str) -> dict:
    """Run pipeline on gold dataset and return metrics."""

    # Load gold dataset
    data = json.loads(Path(dataset_path).read_text())
    sources = {s['url']: s for s in data['sources']}

    # Run pipeline
    report = await run_pipeline_v2(
        sources=sources,
        topic=data['query'],
        title="Benchmark Report",
        llm_call=make_llm_call()
    )

    # Evaluate
    metrics = evaluate_report(report, data['sources'])

    return {
        "dataset": dataset_path,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "report": report
    }

def evaluate_report(report: HybridReport, sources: list) -> dict:
    """Calculate quality metrics."""

    facts = []
    for section in report.sections:
        facts.extend([f.text for f in section.facts])

    return {
        "extraction": {
            "total_facts": len(facts),
            "specificity_rate": count_specific(facts) / len(facts),
            "vague_rate": count_vague(facts) / len(facts),
            "fluff_rate": count_fluff(facts) / len(facts),
            "avg_word_count": avg_words(facts),
            "over_40_words": count_over_40(facts) / len(facts)
        },
        "source": {
            "domain_diversity": count_unique_domains(report),
            "authoritative_rate": count_authoritative(report) / len(facts)
        },
        "synthesis": {
            "citation_rate": count_cited_sentences(report) / count_sentences(report),
            "theme_count": len(report.sections)
        }
    }
```

## Usage

```bash
# Run single benchmark
python scripts/benchmark.py tests/fixtures/gold_queries/agentic_coding_2026.json

# Run all benchmarks
python scripts/benchmark.py --all

# Compare before/after change
python scripts/benchmark.py --compare baseline.json current.json
```

## Baseline Results (2026-01-12)

### agentic_coding_2026
```
Total facts: 78
Specificity rate: 29.5%
Vague rate: 21.8%
Fluff rate: 3.8%
Avg words: 28.4
Domain diversity: 17
```

### latest_research (overnight agents)
```
Total facts: 112
Citation rate: 90.2%
Domain diversity: 33
Top sources: code.claude.com (15), anthropic.com (8)
```

## Regression Testing

After any pipeline change:
1. Run `python scripts/benchmark.py --all`
2. Compare to baseline
3. If any metric regresses > 5%, investigate before merging

## Future: Human Evaluation

For facts that metrics can't catch:
- Sample 20 random facts per dataset
- Human labels: GOOD / VAGUE / WRONG / DUPLICATE
- Track inter-annotator agreement
- Build labeled pairs for training dedup model
