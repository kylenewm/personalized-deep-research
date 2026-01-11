#!/usr/bin/env python3
"""Re-render report from saved research data to test render changes."""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from langchain.chat_models import init_chat_model
from open_deep_research.pipeline_v2 import run_pipeline_v2, render_html


def get_llm_call():
    """Create LLM call wrapper using Claude to avoid OpenAI rate limits."""
    model = init_chat_model(
        model="anthropic:claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4000
    )

    async def llm_call(prompt: str) -> str:
        response = await model.ainvoke(prompt)
        return response.content

    return llm_call


async def main():
    # Load saved research data
    data_path = project_root / "tests/fixtures/gold_queries/latest_research.json"
    with open(data_path) as f:
        data = json.load(f)

    query = data["query"]
    sources = data["sources"]

    print(f"[RERENDER] Query: {query}")
    print(f"[RERENDER] Sources: {len(sources)}")
    print("=" * 60)

    # Convert sources to pipeline format
    source_dict = {}
    for i, src in enumerate(sources):
        source_id = f"src_{i}"
        source_dict[source_id] = {
            "content": src.get("content", src.get("raw_content", "")),
            "url": src.get("url", ""),
            "title": src.get("title", ""),
        }

    # Filter out empty sources
    source_dict = {k: v for k, v in source_dict.items() if v["content"]}
    print(f"[RERENDER] Valid sources: {len(source_dict)}")

    # Run pipeline
    llm_call = get_llm_call()
    report = await run_pipeline_v2(
        sources=source_dict,
        topic=query,
        title="Voice Agent Orchestration Methods in 2026",
        llm_call=llm_call,
        prefer_authoritative_sources=True
    )

    print("\n" + "=" * 60)
    print(f"[REPORT] Sections: {len(report.sections)}")
    print(f"[REPORT] Verified facts: {report.verified_count}")
    print(f"[REPORT] Excluded facts: {len(report.excluded_facts)}")

    # Render HTML
    html = render_html(report)

    # Save
    output_path = project_root / f"report_rerendered_{int(datetime.now().timestamp())}.html"
    output_path.write_text(html)
    print(f"\n[SAVED] {output_path}")
    print(f"[SIZE] {len(html)} bytes")

    # Quick check for our fixes
    if "Additional Sources" in html:
        print("[CHECK] ✓ Additional Sources section found")
    else:
        print("[CHECK] ✗ Additional Sources section NOT found")

    # Check if evidence is truncated
    if '..."' in html or "..." in html:
        # Find how many truncations
        count = html.count("...")
        print(f"[CHECK] Found {count} '...' occurrences (some may be legitimate)")
    else:
        print("[CHECK] ✓ No truncation markers found")


if __name__ == "__main__":
    asyncio.run(main())
