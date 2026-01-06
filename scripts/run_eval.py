#!/usr/bin/env python3
"""
Run Evaluation Framework Standalone

This script runs the evaluation framework on saved reports/state to produce
structured quality metrics. Separate from the pipeline - runs as post-hoc check.

Usage:
    # From saved state JSON
    python scripts/run_eval.py --state state.json --output eval.json

    # From a LangGraph thread (gets state from LangSmith)
    python scripts/run_eval.py --thread-id d58fd825-... --project my-project --output eval.json

    # From report + sources files
    python scripts/run_eval.py --report report.md --sources sources.json --output eval.json

    # From a saved run directory (from run_verification.py)
    python scripts/run_eval.py --run-dir runs/2024-12-06_123456/ --output eval.json

Quality Targets:
    - Hallucination rate: <2%
    - Grounding rate: >85%
    - Citation accuracy: >90%

Requirements:
    - OPENAI_API_KEY environment variable set
    - LANGSMITH_API_KEY environment variable set (for --thread-id)
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


async def load_from_state_file(state_file: str) -> dict:
    """Load complete state from a JSON file.

    Expected format:
    {
        "final_report": "...",
        "source_store": [...],
        "evidence_snippets": [...],
        "messages": [...]
    }
    """
    with open(state_file, 'r') as f:
        state = json.load(f)

    # Validate required fields
    if not state.get("final_report"):
        raise ValueError("State file must contain 'final_report' field")

    print(f"Loaded state from {state_file}")
    print(f"  Report: {len(state.get('final_report', ''))} chars")
    print(f"  Sources: {len(state.get('source_store', []))}")
    print(f"  Evidence snippets: {len(state.get('evidence_snippets', []))}")

    return state


async def load_from_thread(thread_id: str, project_name: str = None) -> dict:
    """Load state from LangSmith.

    Args:
        thread_id: LangGraph thread ID
        project_name: LangSmith project name (defaults to LANGSMITH_PROJECT env var)

    Returns:
        State dict with final_report, source_store, etc.
    """
    from langsmith import Client

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY environment variable not set")

    project = project_name or os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    if not project:
        raise ValueError(
            "Project name required. Set LANGSMITH_PROJECT env var or use --project flag"
        )

    client = Client(api_key=api_key)

    print(f"\n{'='*70}")
    print(f"LOADING FROM THREAD: {thread_id}")
    print(f"PROJECT: {project}")
    print(f"{'='*70}")

    state = {}

    # Try to find the run and extract outputs
    runs = []
    try:
        runs = list(client.list_runs(
            project_name=project,
            trace_id=thread_id,
            is_root=True,
            limit=10
        ))
    except Exception:
        pass

    if not runs:
        # Search by metadata
        all_runs = list(client.list_runs(project_name=project, is_root=True, limit=100))
        runs = [
            r for r in all_runs
            if r.extra and r.extra.get("metadata", {}).get("thread_id") == thread_id
        ]

    # Extract state from run outputs
    for run in runs:
        if run.outputs:
            if run.outputs.get("final_report"):
                state["final_report"] = run.outputs["final_report"]
            if run.outputs.get("source_store"):
                state["source_store"] = run.outputs["source_store"]
            if run.outputs.get("evidence_snippets"):
                state["evidence_snippets"] = run.outputs["evidence_snippets"]
            if run.outputs.get("messages"):
                state["messages"] = run.outputs["messages"]
            break

    if not state.get("final_report"):
        raise ValueError(f"No final_report found for thread {thread_id}")

    print(f"  Report: {len(state.get('final_report', ''))} chars")
    print(f"  Sources: {len(state.get('source_store', []))}")

    return state


async def load_from_files(report_file: str, sources_file: str) -> dict:
    """Load report and sources from separate files."""
    # Load report
    with open(report_file, 'r') as f:
        content = f.read()

    # Try to parse as JSON first
    try:
        data = json.loads(content)
        report = data.get("final_report") or data.get("report") or data.get("content", content)
    except json.JSONDecodeError:
        # Assume it's raw markdown/text
        report = content

    # Load sources
    with open(sources_file, 'r') as f:
        sources = json.load(f)

    return {
        "final_report": report,
        "source_store": sources,
        "evidence_snippets": []
    }


async def load_from_run_dir(run_dir: str) -> dict:
    """Load state from a saved run directory (from run_verification.py)."""
    run_path = Path(run_dir)

    report_file = run_path / "report.json"
    sources_file = run_path / "sources.json"

    if not report_file.exists():
        raise FileNotFoundError(f"Report file not found: {report_file}")
    if not sources_file.exists():
        raise FileNotFoundError(f"Sources file not found: {sources_file}")

    # Load report
    with open(report_file, 'r') as f:
        report_data = json.load(f)
    report = report_data.get("final_report") or report_data.get("report", "")

    # Load sources
    with open(sources_file, 'r') as f:
        sources = json.load(f)

    return {
        "final_report": report,
        "source_store": sources,
        "evidence_snippets": []
    }


async def run_evaluation(
    state: dict,
    model: str = "openai:gpt-4.1-mini",
    dry_run: bool = False,
    max_claims: int = 30,
    batch_size: int = 5
) -> "EvalResult":
    """Run the evaluation framework on the state."""
    from open_deep_research.evaluation import evaluate_report, EvalConfig

    config = EvalConfig(
        model=model,
        max_claims=max_claims,
        verify_citations=True,
        check_verified_findings=True,
        dry_run=dry_run,
        parallel_batch_size=batch_size,
        fallback_to_embedding=True
    )

    print(f"\n{'='*70}")
    print("EVALUATION" + (" (DRY RUN)" if dry_run else ""))
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Max claims: {max_claims}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*70}\n")

    result = await evaluate_report(state, config)

    return result


def print_result(result):
    """Print evaluation result in a readable format."""
    claims = result.claims
    citations = result.citations
    vf = result.verified_findings

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")

    # Cost estimate
    if result.cost_estimate:
        print(f"\nCOST: ~${result.cost_estimate.estimated_cost_usd:.2f}")

    # Claim metrics
    print(f"\nCLAIMS:")
    print(f"  Total: {claims.total}")
    print(f"  TRUE: {claims.true_count} ({claims.grounding_rate:.1%} grounding)")
    print(f"  FALSE: {claims.false_count} ({claims.hallucination_rate:.1%} hallucination)")
    print(f"  UNVERIFIABLE: {claims.unverifiable_count}")
    print(f"  UNCITED (high-risk): {claims.uncited_count}")

    # Quality targets
    print(f"\nQUALITY TARGETS:")
    halluc_status = "PASS" if claims.hallucination_rate < 0.02 else "FAIL"
    ground_status = "PASS" if claims.grounding_rate > 0.85 else "FAIL"
    cite_status = "PASS" if citations.accuracy > 0.90 else "FAIL"

    print(f"  Hallucination <2%: {claims.hallucination_rate:.1%} [{halluc_status}]")
    print(f"  Grounding >85%: {claims.grounding_rate:.1%} [{ground_status}]")
    print(f"  Citation >90%: {citations.accuracy:.1%} [{cite_status}]")

    # Citations
    print(f"\nCITATIONS:")
    print(f"  Total refs: {citations.total}")
    print(f"  Valid sources: {citations.valid}")
    print(f"  Supported: {citations.supported}")
    print(f"  Accuracy: {citations.accuracy:.1%}")
    print(f"  Unique sources: {citations.unique_sources}")

    # Verified Findings
    print(f"\nVERIFIED FINDINGS:")
    print(f"  Quotes: {vf.quotes}")
    print(f"  All PASS: {'Yes' if vf.all_pass else 'No'}")
    print(f"  Source diversity: {vf.source_diversity}")
    if vf.error:
        print(f"  Error: {vf.error}")

    # Warnings
    if result.warnings:
        print(f"\nWARNINGS ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  - {w}")

    # Per-claim breakdown (first 10)
    if result.per_claim:
        print(f"\n{'='*70}")
        print("PER-CLAIM BREAKDOWN (first 10)")
        print(f"{'='*70}")

        for claim in result.per_claim[:10]:
            status_icon = {"TRUE": "T", "FALSE": "F", "UNVERIFIABLE": "?"}.get(claim.status, "?")
            uncited_flag = " [UNCITED]" if claim.is_uncited else ""

            print(f"\n[{claim.claim_id}] {status_icon} ({claim.confidence:.0%}){uncited_flag}")
            print(f"       {claim.claim_text[:70]}...")
            if claim.citations:
                print(f"       Citations: {claim.citations}")
            if claim.sources_checked:
                print(f"       Checked: {claim.sources_checked[0][:50]}...")
            if claim.evidence_snippet:
                print(f"       Evidence: {claim.evidence_snippet[:60]}...")

    print(f"\n{'='*70}")


async def main():
    parser = argparse.ArgumentParser(description='Run evaluation framework standalone')

    # Input options
    parser.add_argument('--state', '-s', help='Path to state JSON file')
    parser.add_argument('--thread-id', '-t', help='LangGraph thread ID')
    parser.add_argument('--project', '-p', help='LangSmith project name')
    parser.add_argument('--report', '-r', help='Path to report file (markdown or JSON)')
    parser.add_argument('--sources', help='Path to sources JSON file')
    parser.add_argument('--run-dir', '-d', help='Path to saved run directory')

    # Model options
    parser.add_argument('--model', '-m', default='openai:gpt-4.1-mini',
                       help='Model for evaluation (default: openai:gpt-4.1-mini)')

    # Evaluation options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show cost estimate without running evaluation')
    parser.add_argument('--max-claims', type=int, default=30,
                       help='Maximum claims to verify (default: 30)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Parallel batch size (default: 5)')

    # Output options
    parser.add_argument('--output', '-o', help='Path to write evaluation JSON')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress detailed output')

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Load state based on input method
    state = None

    if args.state:
        state = await load_from_state_file(args.state)
    elif args.thread_id:
        state = await load_from_thread(args.thread_id, args.project)
    elif args.run_dir:
        state = await load_from_run_dir(args.run_dir)
    elif args.report and args.sources:
        state = await load_from_files(args.report, args.sources)
    else:
        parser.print_help()
        print("\nERROR: Must provide one of:")
        print("  --state <file>                      (complete state JSON)")
        print("  --thread-id <id>                    (retrieves from LangSmith)")
        print("  --run-dir <path>                    (loads from saved run)")
        print("  --report <file> --sources <file>    (loads from separate files)")
        sys.exit(1)

    # Run evaluation
    result = await run_evaluation(
        state,
        model=args.model,
        dry_run=args.dry_run,
        max_claims=args.max_claims,
        batch_size=args.batch_size
    )

    # Print results
    if not args.quiet:
        print_result(result)

    # Save output
    if args.output:
        result.to_json(args.output)
        print(f"\nResults saved to: {args.output}")
    else:
        # Default output path
        output_path = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.to_json(output_path)
        print(f"\nResults saved to: {output_path}")

    # Exit code based on quality targets
    failed_targets = 0
    if result.claims.hallucination_rate >= 0.02:
        failed_targets += 1
    if result.claims.grounding_rate < 0.85:
        failed_targets += 1
    if result.citations.accuracy < 0.90:
        failed_targets += 1

    if failed_targets > 0:
        print(f"\n{failed_targets} quality target(s) not met")
        sys.exit(1)
    else:
        print("\nAll quality targets met!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
