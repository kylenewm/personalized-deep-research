#!/usr/bin/env python3
"""Search sandbox - analyze Tavily search and summarization quality.

This sandbox tests:
1. Raw search result quality
2. Summarization compression ratio
3. Extract API vs raw content
4. Relevance filtering accuracy

Usage:
    python scripts/search_sandbox.py --search "query"    # Run search and analyze
    python scripts/search_sandbox.py --analyze trace.json  # Analyze saved trace
    python scripts/search_sandbox.py --compare           # Compare raw vs extract
    python scripts/search_sandbox.py --stats             # Show all stats
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

TRACES_DIR = project_root / "tests/fixtures/search_traces"


def ensure_traces_dir():
    """Create traces directory if needed."""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)


async def run_search(query: str, max_results: int = 5) -> dict:
    """Run Tavily search and capture raw results.

    Returns:
        Dict with query, raw results, formatted output, and metrics
    """
    from tavily import AsyncTavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("Error: TAVILY_API_KEY not set")
        return {"error": "No API key"}

    client = AsyncTavilyClient(api_key=api_key)

    print(f"Searching: {query[:60]}...")

    # Run search
    raw_results = await client.search(
        query,
        max_results=max_results,
        include_raw_content=True,
        include_answer=False
    )

    # Extract metrics
    results = raw_results.get("results", [])

    trace = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "max_results": max_results,
        "results_count": len(results),
        "results": []
    }

    for r in results:
        content = r.get("content", "")
        raw_content = r.get("raw_content", "")

        trace["results"].append({
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content_len": len(content),
            "raw_content_len": len(raw_content) if raw_content else 0,
            "content": content[:500],  # Preview
            "raw_content_preview": raw_content[:500] if raw_content else "",
            "score": r.get("score", 0),
        })

    return trace


async def run_extract(urls: list) -> dict:
    """Run Tavily Extract API on URLs.

    Returns:
        Dict with extraction results
    """
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("Error: TAVILY_API_KEY not set")
        return {"error": "No API key"}

    client = TavilyClient(api_key=api_key)

    print(f"Extracting {len(urls)} URLs...")

    try:
        results = client.extract(urls=urls)

        extracted = []
        for r in results.get("results", []):
            content = r.get("raw_content", "")
            extracted.append({
                "url": r.get("url", ""),
                "content_len": len(content),
                "content_preview": content[:500],
            })

        return {
            "urls_requested": len(urls),
            "urls_extracted": len(extracted),
            "extracted": extracted,
            "failed_urls": results.get("failed_results", [])
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_trace(trace: dict) -> dict:
    """Analyze a search trace for quality metrics.

    Returns:
        Dict with:
        - avg_content_length: average content size
        - raw_content_ratio: % with raw content
        - avg_score: relevance score
        - content_quality: rough quality assessment
    """
    results = trace.get("results", [])

    if not results:
        return {"error": "No results"}

    content_lengths = [r["content_len"] for r in results]
    raw_lengths = [r["raw_content_len"] for r in results]
    scores = [r.get("score", 0) for r in results]

    has_raw = sum(1 for r in raw_lengths if r > 0)

    # Quality buckets
    short_content = sum(1 for l in content_lengths if l < 200)
    medium_content = sum(1 for l in content_lengths if 200 <= l < 1000)
    long_content = sum(1 for l in content_lengths if l >= 1000)

    return {
        "query": trace.get("query", "unknown")[:60],
        "results_count": len(results),
        "avg_content_len": round(sum(content_lengths) / len(content_lengths)),
        "min_content_len": min(content_lengths),
        "max_content_len": max(content_lengths),
        "has_raw_content": f"{has_raw}/{len(results)}",
        "avg_raw_len": round(sum(raw_lengths) / len(raw_lengths)) if has_raw else 0,
        "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "content_buckets": {
            "short_<200": short_content,
            "medium_200-1000": medium_content,
            "long_>=1000": long_content
        }
    }


def print_analysis(analysis: dict):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print(f"SEARCH TRACE ANALYSIS")
    print(f"{'='*60}")
    print(f"Query: {analysis.get('query', 'unknown')}...")
    print(f"\n  Results: {analysis['results_count']}")
    print(f"  Avg Score: {analysis['avg_score']}")

    print(f"\n  Content Length:")
    print(f"    Average: {analysis['avg_content_len']} chars")
    print(f"    Min: {analysis['min_content_len']} chars")
    print(f"    Max: {analysis['max_content_len']} chars")

    print(f"\n  Raw Content: {analysis['has_raw_content']}")
    if analysis['avg_raw_len'] > 0:
        print(f"    Avg Raw Length: {analysis['avg_raw_len']} chars")

    print(f"\n  Content Distribution:")
    buckets = analysis['content_buckets']
    total = analysis['results_count']
    for bucket, count in buckets.items():
        bar = "█" * int(count / total * 20) if total > 0 else ""
        print(f"    {bucket}: {count} {bar}")

    # Quality assessment
    if analysis['avg_content_len'] < 200:
        print(f"\n  ⚠️  Low content length - may need Extract API")
    elif analysis['avg_content_len'] > 1000:
        print(f"\n  ✅ Good content length")

    if analysis['avg_score'] < 0.5:
        print(f"  ⚠️  Low relevance scores")


async def compare_raw_vs_extract(query: str):
    """Compare raw search content vs Extract API content."""
    # Run search
    trace = await run_search(query, max_results=5)

    if "error" in trace:
        print(f"Search error: {trace['error']}")
        return

    # Get URLs
    urls = [r["url"] for r in trace["results"]]

    # Run extract
    extract_result = await run_extract(urls)

    if "error" in extract_result:
        print(f"Extract error: {extract_result['error']}")
        return

    # Compare
    print(f"\n{'='*60}")
    print("RAW SEARCH vs EXTRACT API COMPARISON")
    print(f"{'='*60}")
    print(f"Query: {query[:60]}...")

    extract_by_url = {e["url"]: e for e in extract_result.get("extracted", [])}

    for r in trace["results"]:
        url = r["url"]
        raw_len = r["content_len"]

        if url in extract_by_url:
            extract_len = extract_by_url[url]["content_len"]
            ratio = extract_len / raw_len if raw_len > 0 else 0

            print(f"\n  {r['title'][:50]}...")
            print(f"    Raw: {raw_len} chars")
            print(f"    Extract: {extract_len} chars")
            print(f"    Ratio: {ratio:.1f}x")

            if ratio > 5:
                print(f"    ✅ Extract got {ratio:.0f}x more content")
            elif ratio < 1:
                print(f"    ⚠️  Extract got less content")
        else:
            print(f"\n  {r['title'][:50]}...")
            print(f"    Raw: {raw_len} chars")
            print(f"    Extract: FAILED")


def show_all_stats():
    """Show aggregated stats from all captured traces."""
    traces = list(TRACES_DIR.glob("*.json"))

    if not traces:
        print("No captured traces found.")
        print(f"Run: python scripts/search_sandbox.py --search 'your query'")
        return

    print(f"\n{'='*60}")
    print(f"AGGREGATED SEARCH STATS ({len(traces)} traces)")
    print(f"{'='*60}")

    all_content_lens = []
    all_scores = []
    all_result_counts = []

    for trace_path in traces:
        with open(trace_path) as f:
            trace = json.load(f)

        analysis = analyze_trace(trace)
        if "error" not in analysis:
            all_content_lens.append(analysis["avg_content_len"])
            all_scores.append(analysis["avg_score"])
            all_result_counts.append(analysis["results_count"])

            print(f"\n  {trace_path.stem}:")
            print(f"    Results: {analysis['results_count']}, Avg Length: {analysis['avg_content_len']}")

    if all_content_lens:
        print(f"\n{'='*60}")
        print("AGGREGATE METRICS")
        print(f"{'='*60}")
        print(f"  Avg content length: {sum(all_content_lens)//len(all_content_lens)} chars")
        print(f"  Avg relevance score: {sum(all_scores)/len(all_scores):.3f}")
        print(f"  Avg results per query: {sum(all_result_counts)//len(all_result_counts)}")

        short_avg = sum(1 for l in all_content_lens if l < 300)
        if short_avg > len(all_content_lens) // 2:
            print(f"\n  💡 Consider enabling Extract API by default")


async def test_summarization_quality():
    """Test search summarization by comparing before/after.

    Tavily returns both 'content' (AI-summarized) and 'raw_content' (full page).
    This compares the compression ratio between them.
    """
    query = "best practices for Python async programming"

    print(f"\n{'='*60}")
    print("SUMMARIZATION QUALITY TEST")
    print(f"{'='*60}")

    # Get raw results
    trace = await run_search(query, max_results=3)

    if "error" in trace:
        print(f"Search error: {trace['error']}")
        return

    # Calculate raw size
    total_raw = sum(r["raw_content_len"] for r in trace["results"])
    total_summary = sum(r["content_len"] for r in trace["results"])

    print(f"\nQuery: {query}")
    print(f"Results: {len(trace['results'])}")
    print(f"\nContent Sizes:")
    print(f"  Total raw content: {total_raw:,} chars")
    print(f"  Summarized content: {total_summary:,} chars")

    if total_raw > 0:
        compression = (1 - total_summary / total_raw) * 100
        print(f"  Compression: {compression:.1f}%")

    print("\nPer-result breakdown:")
    for r in trace["results"]:
        title = r["title"][:40]
        raw = r["raw_content_len"]
        summary = r["content_len"]
        if raw > 0:
            ratio = summary / raw * 100
            print(f"  {title}... {raw:,} → {summary:,} ({ratio:.0f}%)")
        else:
            print(f"  {title}... (no raw content)")


async def main():
    args = sys.argv[1:]

    ensure_traces_dir()

    if "--search" in args:
        idx = args.index("--search")
        if idx + 1 < len(args):
            query = args[idx + 1]
            trace = await run_search(query)

            if "error" not in trace:
                # Save trace
                trace_path = TRACES_DIR / f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(trace_path, 'w') as f:
                    json.dump(trace, f, indent=2)
                print(f"\nTrace saved to: {trace_path}")

                # Analyze
                analysis = analyze_trace(trace)
                print_analysis(analysis)
            else:
                print(f"Error: {trace['error']}")
        else:
            print("Usage: --search 'query string'")

    elif "--analyze" in args:
        idx = args.index("--analyze")
        if idx + 1 < len(args):
            trace_path = Path(args[idx + 1])
            if trace_path.exists():
                with open(trace_path) as f:
                    trace = json.load(f)
                analysis = analyze_trace(trace)
                print_analysis(analysis)
            else:
                print(f"Trace not found: {trace_path}")
        else:
            print("Usage: --analyze trace.json")

    elif "--compare" in args:
        idx = args.index("--compare")
        query = args[idx + 1] if idx + 1 < len(args) else "Python async best practices"
        await compare_raw_vs_extract(query)

    elif "--summarize" in args:
        await test_summarization_quality()

    elif "--stats" in args:
        show_all_stats()

    else:
        print(__doc__)
        print("\nAvailable commands:")
        print("  --search 'query'     Run search and analyze")
        print("  --analyze file.json  Analyze saved trace")
        print("  --compare 'query'    Compare raw vs Extract API")
        print("  --summarize          Test summarization compression")
        print("  --stats              Show aggregated stats")


if __name__ == "__main__":
    asyncio.run(main())
