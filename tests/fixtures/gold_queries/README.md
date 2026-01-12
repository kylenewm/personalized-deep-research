# Gold Query Datasets

Test datasets for evaluating pipeline quality. Each contains a query, the generated brief, extracted facts, and synthesized report.

## Datasets

### agentic_coding_2026.json
**Query:** Multi-agent orchestration frameworks for AI coding assistants (LangGraph, CrewAI, AutoGen comparison)

- Facts: ~73
- Themes: 5
- Has full report: Yes
- Primary use: Full eval (brief + upstream + downstream)

### latest_research.json
**Query:** Voice agent orchestration

- Facts: ~42
- Themes: Variable
- Has full report: No
- Primary use: Upstream eval only

### baseline_2026-01-12.json
**Query:** N/A - metrics snapshot

- Purpose: Reference baseline for regression detection
- Not for direct evaluation

## Usage

```bash
# Run eval on single dataset
python eval/run_eval.py tests/fixtures/gold_queries/agentic_coding_2026.json --mode mini

# Run on all (skips baseline files)
python eval/run_eval.py --all
```

## Adding New Datasets

1. Run a research query through the pipeline
2. Save output as JSON with structure:
   ```json
   {
     "query": "...",
     "research_brief": "...",
     "hybrid_report": { "sections": [...], "footnotes": [...] }
   }
   ```
3. Run eval to verify quality metrics
4. Add to this directory

## Query Templates

Good eval queries are:
- **Specific** - Named entities, comparisons, constraints
- **Measurable** - Asks for numbers, benchmarks, concrete details
- **Bounded** - Clear scope, not open-ended

Example:
```
What are the best multi-agent orchestration frameworks for AI coding assistants in 2026?
Compare LangGraph, CrewAI, AutoGen for building collaborative AI coding agents.
Focus on specific numbers, benchmarks, and metrics - NOT generic claims.
```
