#!/usr/bin/env python3
"""Standalone pipeline runner for testing checkpoints.

Runs pipeline_v2 directly without full graph dependencies.

Usage:
    python scripts/run_pipeline_standalone.py "your query here"
    python scripts/run_pipeline_standalone.py --from-state run_state_123.json
"""

import asyncio
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from openai import AsyncOpenAI

from open_deep_research.pipeline_v2 import run_pipeline_v2
from open_deep_research.render import render_report, report_to_dict



async def run_from_sources(sources: dict, query: str):
    """Run pipeline on provided sources."""
    client = AsyncOpenAI()
    call_count = [0]

    async def llm_call(prompt: str) -> str:
        # Add delay every call to avoid rate limits
        call_count[0] += 1
        if call_count[0] > 1:
            await asyncio.sleep(2.0)  # 2 second delay between calls

        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3
        )
        return resp.choices[0].message.content

    def on_progress(stage: str, msg: str):
        print(f"[{stage}] {msg}")

    print(f"Running pipeline on {len(sources)} sources...")
    print(f"Query: {query[:100]}...")
    print("=" * 60)

    start = time.time()
    # Generate title from query (first 100 chars, truncate at word boundary)
    title = query[:100].rsplit(' ', 1)[0] if len(query) > 100 else query

    report = await run_pipeline_v2(
        sources=sources,
        topic=query,
        title=title,
        llm_call=llm_call,
        on_progress=on_progress
    )
    elapsed = time.time() - start

    print("=" * 60)
    print(f"Complete in {elapsed:.1f}s")
    print(f"Facts: {report.verified_count}")
    print(f"Themes: {len(report.sections)}")

    # Check checkpoints
    if report.checkpoints:
        print(f"\nCheckpoints captured:")
        print(f"  pre_dedup: {report.checkpoints.get('pre_dedup_count', 'N/A')} facts")
        print(f"  post_dedup: {report.checkpoints.get('post_dedup_count', 'N/A')} facts")
        print(f"  pre_arrangement: {report.checkpoints.get('pre_arrangement_count', 'N/A')} facts")
        post_arr = report.checkpoints.get('post_arrangement', {})
        print(f"  post_arrangement: {post_arr.get('grouped_count', 'N/A')} grouped, {post_arr.get('excluded_count', 'N/A')} excluded")
    else:
        print("\nWARN: No checkpoints captured")

    return report


async def main():
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python scripts/run_pipeline_standalone.py --from-state run_state_123.json")
        print("  python scripts/run_pipeline_standalone.py 'your query here'")
        return

    if args[0] == "--from-state":
        # Load sources from existing run_state
        state_path = Path(args[1])
        if not state_path.exists():
            print(f"Error: {state_path} not found")
            return

        with open(state_path) as f:
            state = json.load(f)

        source_store = state.get("source_store", [])
        if not source_store:
            print("Error: No sources in state")
            return

        # Convert to sources dict (limit to 5 for rate limit management)
        sources = {}
        for i, src in enumerate(source_store[:5]):
            content = src.get("content", "") or src.get("raw_content", "")
            if content:
                sources[f"src_{i:03d}"] = {
                    "content": content,
                    "url": src.get("url", ""),
                    "title": src.get("title", "Unknown"),
                }

        # Get query
        brief = state.get("research_brief", {})
        if isinstance(brief, dict):
            query = brief.get("query", "Research topic")
        else:
            query = str(brief)[:500] if brief else "Research topic"

        print(f"Loaded {len(sources)} sources from {state_path.name}")

    else:
        print("Error: Query-based run not supported yet. Use --from-state")
        return

    report = await run_from_sources(sources, query)

    # Save results
    timestamp = int(time.time())

    # Save HTML report
    html_path = project_root / f"report_{timestamp}.html"
    html_content = render_report(report)
    html_path.write_text(html_content)
    print(f"\nHTML saved: {html_path.name}")

    # Save run_state with checkpoints
    state_path = project_root / f"run_state_{timestamp}.json"
    state_data = {
        "research_brief": {"query": query},
        "source_store": source_store,
        "hybrid_report": report_to_dict(report),
    }
    with open(state_path, "w") as f:
        json.dump(state_data, f, indent=2, default=str)
    print(f"State saved: {state_path.name}")


if __name__ == "__main__":
    asyncio.run(main())
