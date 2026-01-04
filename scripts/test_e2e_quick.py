"""Quick E2E test with test_mode=True for abbreviated runs.

This script runs an abbreviated research pipeline for quick validation.
With test_mode=True:
- max_researcher_iterations: 6 -> 2
- max_react_tool_calls: 10 -> 3
- max_concurrent_research_units: 5 -> 2

Usage:
    # Basic run
    python scripts/test_e2e_quick.py

    # With LangSmith tracing
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=your_key
    python scripts/test_e2e_quick.py

    # Custom query
    python scripts/test_e2e_quick.py "What is quantum computing?"

    # Skip report generation
    python scripts/test_e2e_quick.py --no-report "Your query"
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load .env file if it exists
env_file = project_root / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"[INFO] Loaded environment from {env_file}")
    except ImportError:
        # Manual .env loading if python-dotenv not installed
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        print(f"[INFO] Loaded environment from {env_file} (manual parse)")


async def run_quick_test(
    query: str = "What are the latest developments in AI safety?",
    generate_report: bool = True,
):
    """Run abbreviated E2E test."""
    from open_deep_research.graph import deep_researcher as graph
    from generate_run_report import generate_run_report

    print("=" * 60)
    print("Quick E2E Test (test_mode=True)")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"LangSmith tracing: {'enabled' if os.getenv('LANGCHAIN_TRACING_V2') else 'disabled'}")
    print("=" * 60)

    # Check if Anthropic key is available for council
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    config = {
        "configurable": {
            "thread_id": f"test_e2e_{int(time.time())}",  # Unique thread ID
            "test_mode": True,  # Reduces iterations significantly
            "use_claim_verification": True,  # Enable verification
            "use_tavily_extract": True,  # Use Extract API for cleaner content
            # Disable council if no Anthropic key (council uses both OpenAI + Anthropic)
            "use_council": has_anthropic,
            "use_findings_council": has_anthropic,
            # Disable clarification for automated testing
            "allow_clarification": False,
        }
    }

    if not has_anthropic:
        print("[WARN] ANTHROPIC_API_KEY not set - council disabled (uses OpenAI + Anthropic)")

    start_time = time.time()

    try:
        print("\n[1/4] Starting research pipeline...")
        result = await graph.ainvoke(
            {"messages": [("user", query)]},
            config=config
        )

        elapsed = time.time() - start_time
        print(f"\n[2/4] Pipeline complete in {elapsed:.1f}s")

        # Extract results
        final_report = result.get("final_report", "")
        verification = result.get("verification_result", {})
        evidence_snippets = result.get("evidence_snippets", [])
        source_store = result.get("source_store", [])

        print(f"\n[3/4] Results Summary:")
        print(f"  - Report length: {len(final_report)} chars")
        print(f"  - Sources stored: {len(source_store)}")
        print(f"  - Evidence snippets: {len(evidence_snippets)}")

        # Verification summary
        if verification:
            summary = verification.get("summary", {})
            print(f"\n[4/4] Verification Summary:")
            print(f"  - Total claims: {summary.get('total_claims', 0)}")
            print(f"  - Supported: {summary.get('supported', 0)}")
            print(f"  - Partially supported: {summary.get('partially_supported', 0)}")
            print(f"  - Unsupported: {summary.get('unsupported', 0)}")
            print(f"  - Uncertain: {summary.get('uncertain', 0)}")
            print(f"  - Overall confidence: {summary.get('overall_confidence', 0):.0%}")

            # Show warnings if any
            warnings = summary.get("warnings", [])
            if warnings:
                print(f"\n  Warnings ({len(warnings)}):")
                for w in warnings[:3]:
                    print(f"    - {w[:80]}...")

            # Show data issues if any
            data_issues = summary.get("data_issues", [])
            if data_issues:
                print(f"\n  Data issues ({len(data_issues)}):")
                for issue in data_issues[:3]:
                    print(f"    - {issue[:80]}...")
        else:
            print("\n[4/4] Verification: skipped or disabled")

        # Show snippet of report
        print("\n" + "=" * 60)
        print("Report Preview (first 500 chars):")
        print("=" * 60)
        print(final_report[:500] if final_report else "(no report generated)")
        print("...")

        # Show extraction methods used
        if source_store:
            methods = {}
            for s in source_store:
                method = s.get("extraction_method", "unknown")
                methods[method] = methods.get(method, 0) + 1
            print(f"\n[INFO] Extraction methods used: {methods}")

        # Show evidence snippet stats
        if evidence_snippets:
            statuses = {}
            for e in evidence_snippets:
                status = e.get("status", "unknown")
                statuses[status] = statuses.get(status, 0) + 1
            print(f"[INFO] Evidence snippet statuses: {statuses}")

        # Generate detailed run report
        if generate_report:
            print("\n" + "=" * 60)
            print("Generating detailed run report...")
            print("=" * 60)

            # Add query to result for report
            result["messages"] = [("user", query)]

            report_path = project_root / f"run_report_{int(time.time())}.md"
            generate_run_report(result, str(report_path), run_duration=elapsed)

            # Also save raw state as JSON for debugging
            state_path = project_root / f"run_state_{int(time.time())}.json"
            try:
                # Filter serializable items
                serializable_state = {}
                for k, v in result.items():
                    try:
                        json.dumps(v)
                        serializable_state[k] = v
                    except (TypeError, ValueError):
                        serializable_state[k] = f"<non-serializable: {type(v).__name__}>"

                with open(state_path, "w") as f:
                    json.dump(serializable_state, f, indent=2, default=str)
                print(f"[STATE] Saved to: {state_path}")
            except Exception as e:
                print(f"[WARN] Could not save state JSON: {e}")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Pipeline failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    # Parse arguments
    args = sys.argv[1:]
    generate_report = True
    query = "What are the latest developments in AI safety?"

    if "--no-report" in args:
        generate_report = False
        args.remove("--no-report")

    if args:
        query = args[0]

    # Check required env vars
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"[ERROR] Missing required environment variables: {missing}")
        print("Please set them before running this script.")
        sys.exit(1)

    # Run the test
    result = asyncio.run(run_quick_test(query, generate_report=generate_report))

    if result:
        print("\n" + "=" * 60)
        print("TEST PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
