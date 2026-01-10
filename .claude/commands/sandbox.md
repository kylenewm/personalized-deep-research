# Run Sandbox Pipeline

Test Pipeline v2 without re-running research.

## Commands

```bash
# List available fixtures
python scripts/sandbox_pipeline.py --list

# Run pipeline on fixture
python scripts/sandbox_pipeline.py --run <name>

# Run with brief
python scripts/sandbox_pipeline.py --run <name> --use-brief

# Run diagnostics (see extraction breakdown)
python scripts/sandbox_pipeline.py --diagnose <name>
```

## Output

Reports saved to: `sandbox_output/`

## When to Use

- Testing pipeline changes without Tavily costs
- Debugging extraction/filter issues
- Iterating on report quality
