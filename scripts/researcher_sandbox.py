#!/usr/bin/env python3
"""Researcher sandbox - analyze research iteration efficiency.

The researcher loop is ~80% of token cost. This sandbox:
1. Captures real research traces for replay testing
2. Analyzes iteration efficiency (diminishing returns)
3. Tests with mock search results

Usage:
    python scripts/researcher_sandbox.py --capture "query"  # Capture real trace
    python scripts/researcher_sandbox.py --analyze trace.json  # Analyze captured trace
    python scripts/researcher_sandbox.py --stats  # Show stats from all traces
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

TRACES_DIR = project_root / "tests/fixtures/research_traces"


def ensure_traces_dir():
    """Create traces directory if needed."""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)


def analyze_trace(trace_path: Path) -> dict:
    """Analyze a captured research trace.

    Returns:
        Dict with metrics:
        - iterations: total search iterations
        - facts_per_iteration: list of fact counts per iteration
        - cumulative_facts: cumulative unique facts over iterations
        - diminishing_point: iteration where marginal value drops below 10%
    """
    with open(trace_path) as f:
        trace = json.load(f)

    iterations = trace.get("iterations", [])

    if not iterations:
        return {"error": "No iterations in trace"}

    # Track facts per iteration
    all_facts = set()
    facts_per_iter = []
    cumulative = []
    marginal_values = []

    for i, iter_data in enumerate(iterations):
        iter_facts = set(iter_data.get("facts_found", []))
        new_facts = iter_facts - all_facts
        all_facts.update(new_facts)

        facts_per_iter.append(len(new_facts))
        cumulative.append(len(all_facts))

        if i > 0 and cumulative[i-1] > 0:
            marginal = len(new_facts) / cumulative[i-1]
        else:
            marginal = 1.0
        marginal_values.append(marginal)

    # Find point where marginal value drops below 10%
    diminishing_point = len(iterations)
    for i, mv in enumerate(marginal_values):
        if i > 0 and mv < 0.1:  # Less than 10% new facts
            diminishing_point = i
            break

    # Calculate efficiency metrics
    total_facts = len(all_facts)
    if len(iterations) > 0:
        facts_from_first_half = cumulative[len(iterations) // 2 - 1] if len(iterations) > 1 else cumulative[0]
        first_half_pct = facts_from_first_half / total_facts if total_facts > 0 else 0
    else:
        first_half_pct = 0

    return {
        "query": trace.get("query", "unknown"),
        "total_iterations": len(iterations),
        "total_facts": total_facts,
        "facts_per_iteration": facts_per_iter,
        "cumulative_facts": cumulative,
        "marginal_values": [round(m, 2) for m in marginal_values],
        "diminishing_point": diminishing_point,
        "first_half_coverage": round(first_half_pct, 2),
        "avg_facts_per_iter": round(total_facts / len(iterations), 1) if iterations else 0,
    }


def print_analysis(analysis: dict):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print(f"RESEARCH TRACE ANALYSIS")
    print(f"{'='*60}")
    print(f"Query: {analysis.get('query', 'unknown')[:60]}...")
    print(f"\n  Total Iterations: {analysis['total_iterations']}")
    print(f"  Total Unique Facts: {analysis['total_facts']}")
    print(f"  Avg Facts/Iteration: {analysis['avg_facts_per_iter']}")

    print(f"\n  Iteration Breakdown:")
    for i, (facts, marginal) in enumerate(zip(
        analysis['facts_per_iteration'],
        analysis['marginal_values']
    )):
        bar = "█" * min(facts, 20)
        pct = f"({marginal*100:.0f}% new)" if i > 0 else "(baseline)"
        print(f"    [{i+1:2d}] {facts:3d} facts {bar} {pct}")

    print(f"\n  Key Metrics:")
    print(f"    Diminishing returns at iteration: {analysis['diminishing_point']}")
    print(f"    First half coverage: {analysis['first_half_coverage']*100:.0f}%")

    # Recommendation
    optimal = max(3, analysis['diminishing_point'])
    if optimal < analysis['total_iterations']:
        savings = 100 * (1 - optimal / analysis['total_iterations'])
        print(f"\n  💡 Recommendation: {optimal} iterations would be optimal")
        print(f"     Potential savings: {savings:.0f}% of research cost")
    else:
        print(f"\n  ✅ Current iteration count is reasonable")


def show_all_stats():
    """Show aggregated stats from all captured traces."""
    traces = list(TRACES_DIR.glob("*.json"))

    if not traces:
        print("No captured traces found.")
        print(f"Run: python scripts/researcher_sandbox.py --capture 'your query'")
        return

    print(f"\n{'='*60}")
    print(f"AGGREGATED RESEARCH STATS ({len(traces)} traces)")
    print(f"{'='*60}")

    all_diminishing = []
    all_first_half = []
    all_avg_facts = []

    for trace_path in traces:
        analysis = analyze_trace(trace_path)
        if "error" not in analysis:
            all_diminishing.append(analysis['diminishing_point'])
            all_first_half.append(analysis['first_half_coverage'])
            all_avg_facts.append(analysis['avg_facts_per_iter'])
            print(f"\n  {trace_path.stem}:")
            print(f"    Iterations: {analysis['total_iterations']}, Facts: {analysis['total_facts']}")
            print(f"    Diminishing at: {analysis['diminishing_point']}, First half: {analysis['first_half_coverage']*100:.0f}%")

    if all_diminishing:
        print(f"\n{'='*60}")
        print("AGGREGATE METRICS")
        print(f"{'='*60}")
        print(f"  Avg diminishing point: {sum(all_diminishing)/len(all_diminishing):.1f}")
        print(f"  Avg first-half coverage: {sum(all_first_half)/len(all_first_half)*100:.0f}%")
        print(f"  Avg facts/iteration: {sum(all_avg_facts)/len(all_avg_facts):.1f}")

        median_dim = sorted(all_diminishing)[len(all_diminishing)//2]
        print(f"\n  💡 Suggested default: max_react_tool_calls = {max(3, median_dim)}")


async def capture_trace(query: str):
    """Capture a research trace from a real query.

    To capture real traces, run the research pipeline with RESEARCH_TRACE_ENABLED=true:

        RESEARCH_TRACE_ENABLED=true ./venv/bin/python -m open_deep_research.cli "your query"

    Traces are automatically saved to tests/fixtures/research_traces/
    """
    print(f"To capture real research traces, run:")
    print(f"\n  RESEARCH_TRACE_ENABLED=true ./venv/bin/python -m open_deep_research.cli \"{query}\"")
    print(f"\nTraces are automatically saved to: {TRACES_DIR}")
    print(f"\nThen analyze with:")
    print(f"  python scripts/researcher_sandbox.py --stats")

    # Check for existing traces
    traces = list(TRACES_DIR.glob("*.json")) if TRACES_DIR.exists() else []
    if traces:
        print(f"\n{len(traces)} existing trace(s) found:")
        for t in traces[-5:]:  # Show last 5
            print(f"  {t.name}")
    else:
        # Create example trace for demo
        print("\nCreating example trace for demo...")
        example = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "iterations": [
                {"iteration": 1, "search_queries": ["query 1"], "facts_found": ["fact1", "fact2", "fact3"], "fact_count": 3},
                {"iteration": 2, "search_queries": ["query 2"], "facts_found": ["fact4", "fact5"], "fact_count": 2},
                {"iteration": 3, "search_queries": ["query 3"], "facts_found": ["fact6"], "fact_count": 1},
            ]
        }

        ensure_traces_dir()
        trace_path = TRACES_DIR / f"example_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(trace_path, 'w') as f:
            json.dump(example, f, indent=2)

        print(f"Example trace saved to: {trace_path}")
        print(f"\nRun: python scripts/researcher_sandbox.py --analyze {trace_path}")


def main():
    args = sys.argv[1:]

    if "--capture" in args:
        idx = args.index("--capture")
        if idx + 1 < len(args):
            query = args[idx + 1]
            asyncio.run(capture_trace(query))
        else:
            print("Usage: --capture 'query string'")
    elif "--analyze" in args:
        idx = args.index("--analyze")
        if idx + 1 < len(args):
            trace_path = Path(args[idx + 1])
            if trace_path.exists():
                analysis = analyze_trace(trace_path)
                print_analysis(analysis)
            else:
                print(f"Trace not found: {trace_path}")
        else:
            print("Usage: --analyze trace.json")
    elif "--stats" in args:
        show_all_stats()
    else:
        print(__doc__)
        print("\nAvailable commands:")
        print("  --capture 'query'    Capture research trace")
        print("  --analyze file.json  Analyze a trace")
        print("  --stats              Show aggregated stats")


if __name__ == "__main__":
    main()
