"""Test research phase with visible Tavily output.

This script tests the research phase in isolation, showing exactly
what sources are found and what content is extracted.

Usage:
    python scripts/test_research.py "What is quantum computing?"
    python scripts/test_research.py --brief test_brief_output.md

Output:
    test_research_output.md - Detailed research log
    test_research_state.json - Raw state for debugging
    metrics/metrics_{timestamp}.json - Run metrics
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load .env file
env_file = project_root / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from staged_config import MINIMAL_CONFIG, print_config_summary

# Import metrics
from open_deep_research.metrics import MetricsCollector

DEFAULT_QUERY = "What are the latest developments in AI safety?"


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


async def test_tavily_directly(query: str):
    """Test Tavily search directly to see what it returns."""
    from tavily import AsyncTavilyClient

    print("\n" + "=" * 60)
    print("STEP 1: Direct Tavily Search")
    print("=" * 60)

    client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    print(f"Query: {query}")
    start = time.time()

    results = await client.search(
        query=query,
        max_results=5,
        include_raw_content=True,
        topic="general"
    )

    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s")
    print(f"Results: {len(results.get('results', []))}")

    sources = []
    for i, r in enumerate(results.get("results", []), 1):
        title = r.get("title", "N/A")
        url = r.get("url", "N/A")
        content = r.get("content", "")  # Snippet
        raw_content = r.get("raw_content", "")  # Full content

        print(f"\n--- Source {i}: {title[:50]}... ---")
        print(f"URL: {url}")
        print(f"Snippet: {len(content)} chars")
        print(f"Raw content: {len(raw_content) if raw_content else 0} chars")
        print(f"Preview: {truncate(content, 150)}")

        sources.append({
            "title": title,
            "url": url,
            "snippet_length": len(content),
            "raw_content_length": len(raw_content) if raw_content else 0,
            "snippet_preview": truncate(content, 300),
            "raw_preview": truncate(raw_content, 300) if raw_content else "(empty)"
        })

    return sources, results


async def test_research_pipeline(query: str):
    """Run minimal research pipeline and capture output."""
    from open_deep_research.graph import deep_researcher as graph

    print("\n" + "=" * 60)
    print("STEP 2: Full Research Pipeline (minimal config)")
    print("=" * 60)
    print_config_summary()

    # Initialize metrics collector
    metrics = MetricsCollector(query=query, config=MINIMAL_CONFIG)

    config = {
        "configurable": {
            "thread_id": f"test_research_{int(time.time())}",
            **MINIMAL_CONFIG,
        }
    }

    try:
        with metrics.stage("full_pipeline"):
            result = await graph.ainvoke(
                {"messages": [("user", query)]},
                config=config
            )

        # Populate metrics from result
        metrics.metrics.brief_length_chars = len(result.get("research_brief", ""))
        metrics.metrics.sources_stored = len(result.get("source_store", []))
        metrics.metrics.notes_generated = len(result.get("notes", []))
        metrics.metrics.raw_notes_chars = sum(len(n) for n in result.get("raw_notes", []))
        metrics.metrics.report_length_chars = len(result.get("final_report", ""))

        # Evidence metrics
        snippets = result.get("evidence_snippets", [])
        metrics.metrics.snippets_extracted = len(snippets)
        metrics.metrics.snippets_verified_pass = sum(1 for s in snippets if s.get("status") == "PASS")
        metrics.metrics.snippets_verified_fail = sum(1 for s in snippets if s.get("status") == "FAIL")

        # Verification metrics
        verification = result.get("verification_result")
        if verification:
            summary = verification.get("summary", {})
            metrics.metrics.claims_verified = summary.get("total_claims", 0)
            metrics.metrics.claims_supported = summary.get("supported", 0)
            metrics.metrics.verification_confidence = summary.get("overall_confidence", 0.0)

        metrics.finish()

        # Save metrics to file
        metrics_dir = project_root / "metrics"
        metrics_path = metrics.save(str(metrics_dir))
        print(f"\n[METRICS] Saved to: {metrics_path}")
        metrics.print_summary()

        return {
            "success": True,
            "elapsed": metrics.metrics.total_duration_seconds,
            "brief": result.get("research_brief", ""),
            "notes": result.get("notes", []),
            "raw_notes": result.get("raw_notes", []),
            "source_store": result.get("source_store", []),
            "final_report": result.get("final_report", ""),
            "metrics": metrics.to_dict(),
        }

    except Exception as e:
        import traceback
        metrics.log_error(str(e))
        metrics.finish()

        # Still save metrics even on failure
        try:
            metrics_dir = project_root / "metrics"
            metrics.save(str(metrics_dir))
        except Exception:
            pass

        return {
            "success": False,
            "elapsed": metrics.metrics.total_duration_seconds,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "metrics": metrics.to_dict(),
        }


async def test_research(query: str):
    """Run complete research test."""

    print("=" * 60)
    print("RESEARCH PHASE TEST")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    md = []
    md.append("# Research Phase Test\n")
    md.append(f"**Query:** {query}\n")
    md.append(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 1: Direct Tavily test
    md.append("\n---\n## Step 1: Direct Tavily Search\n")
    md.append("Testing Tavily API directly to see raw results.\n")

    tavily_sources, raw_results = await test_tavily_directly(query)

    md.append(f"\n**Results:** {len(tavily_sources)} sources found\n")
    md.append("\n| # | Title | Snippet | Raw Content |\n")
    md.append("|---|-------|---------|-------------|\n")
    for i, s in enumerate(tavily_sources, 1):
        md.append(f"| {i} | {s['title'][:40]}... | {s['snippet_length']} chars | {s['raw_content_length']} chars |\n")

    md.append("\n### Source Details\n")
    for i, s in enumerate(tavily_sources, 1):
        md.append(f"\n#### Source {i}: {s['title'][:60]}\n")
        md.append(f"**URL:** {s['url']}\n")
        md.append(f"**Snippet ({s['snippet_length']} chars):**\n```\n{s['snippet_preview']}\n```\n")
        if s['raw_content_length'] > 0:
            md.append(f"**Raw content ({s['raw_content_length']} chars, first 300):**\n```\n{s['raw_preview']}\n```\n")
        else:
            md.append("**Raw content:** (empty)\n")

    # Step 2: Full pipeline test
    md.append("\n---\n## Step 2: Full Research Pipeline\n")

    result = await test_research_pipeline(query)

    if result["success"]:
        md.append(f"\n**Status:** Success\n")
        md.append(f"**Time:** {result['elapsed']:.1f}s\n")

        # Brief
        md.append("\n### Research Brief\n")
        md.append(f"```\n{result['brief'][:1000]}{'...' if len(result['brief']) > 1000 else ''}\n```\n")

        # Sources stored
        md.append(f"\n### Sources Stored ({len(result['source_store'])})\n")
        if result['source_store']:
            md.append("\n| # | Title | Method | Content Size |\n")
            md.append("|---|-------|--------|-------------|\n")
            for i, s in enumerate(result['source_store'][:10], 1):
                title = s.get('title', 'N/A')[:40]
                method = s.get('extraction_method', 'unknown')
                content_len = len(s.get('content', ''))
                md.append(f"| {i} | {title}... | {method} | {content_len} chars |\n")
        else:
            md.append("(no sources stored)\n")

        # Notes
        md.append(f"\n### Research Notes ({len(result['notes'])} entries)\n")
        for i, note in enumerate(result['notes'][:3], 1):
            preview = truncate(note, 500)
            md.append(f"\n**Note {i}:**\n```\n{preview}\n```\n")

        # Report preview
        md.append("\n### Final Report Preview\n")
        report = result['final_report']
        md.append(f"**Length:** {len(report)} chars\n")
        md.append(f"```\n{report[:1000]}{'...' if len(report) > 1000 else ''}\n```\n")

        print(f"\n[RESULT] Success in {result['elapsed']:.1f}s")
        print(f"  - Brief: {len(result['brief'])} chars")
        print(f"  - Sources: {len(result['source_store'])}")
        print(f"  - Notes: {len(result['notes'])}")
        print(f"  - Report: {len(result['final_report'])} chars")

    else:
        md.append(f"\n**Status:** FAILED\n")
        md.append(f"**Error:** {result['error']}\n")
        md.append(f"```\n{result['traceback']}\n```\n")
        print(f"\n[ERROR] Failed: {result['error']}")

    # Summary
    md.append("\n---\n## Summary\n")
    md.append("\n| Metric | Value |\n")
    md.append("|--------|-------|\n")
    md.append(f"| Tavily sources | {len(tavily_sources)} |\n")
    if result["success"]:
        md.append(f"| Pipeline sources | {len(result['source_store'])} |\n")
        md.append(f"| Notes generated | {len(result['notes'])} |\n")
        md.append(f"| Report length | {len(result['final_report'])} chars |\n")
        md.append(f"| Total time | {result['elapsed']:.1f}s |\n")
    else:
        md.append(f"| Status | FAILED |\n")

    # Save outputs
    output_path = project_root / "test_research_output.md"
    output_path.write_text("\n".join(md))
    print(f"\nOutput saved to: {output_path}")

    # Save state as JSON
    state_path = project_root / "test_research_state.json"
    try:
        state_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "tavily_sources": tavily_sources,
            "pipeline_result": {
                k: v for k, v in result.items()
                if k not in ["traceback"]  # Skip non-serializable
            }
        }
        with open(state_path, "w") as f:
            json.dump(state_data, f, indent=2, default=str)
        print(f"State saved to: {state_path}")
    except Exception as e:
        print(f"[WARN] Could not save state: {e}")

    return result["success"] if "success" in result else False


def main():
    """Entry point."""
    # Check required env vars
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set")
        sys.exit(1)
    if not os.getenv("TAVILY_API_KEY"):
        print("[ERROR] TAVILY_API_KEY not set")
        sys.exit(1)

    # Get query
    query = DEFAULT_QUERY
    if len(sys.argv) > 1:
        if sys.argv[1] == "--brief" and len(sys.argv) > 2:
            # Load brief from file
            brief_path = Path(sys.argv[2])
            if brief_path.exists():
                content = brief_path.read_text()
                # Extract just the brief content (skip markdown headers)
                query = content
                print(f"[INFO] Using brief from: {brief_path}")
        else:
            query = sys.argv[1]

    success = asyncio.run(test_research(query))

    if success:
        print("\nTEST PASSED")
    else:
        print("\nTEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
