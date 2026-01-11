#!/usr/bin/env python3
"""Stress test pipeline with real fixture data."""

import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from openai import AsyncOpenAI
from open_deep_research.pipeline_v2 import run_pipeline_v2
from open_deep_research.render import render_report


async def main():
    # Load fixture
    fixture_path = project_root / "tests/fixtures/gold_queries/voice_agent_eval.json"
    with open(fixture_path) as f:
        data = json.load(f)

    # Quick test with fewer sources
    sources = data["sources"][:8]  # 8 sources for faster run
    topic = data["query"]

    print(f"Stress testing with {len(sources)} sources")
    print(f"Topic: {topic[:80]}...")

    # Convert to expected format
    sources_dict = {}
    for i, s in enumerate(sources):
        sources_dict[f"source_{i}"] = {
            "url": s.get("url", ""),
            "title": s.get("title", ""),
            "content": s.get("content", "")
        }

    # Setup LLM
    client = AsyncOpenAI()

    async def llm_call(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3
        )
        return resp.choices[0].message.content

    def on_progress(stage: str, msg: str):
        print(f"[{stage}] {msg}")

    # Run pipeline
    report = await run_pipeline_v2(
        sources=sources_dict,
        topic=topic,
        title="Stress Test Report",
        llm_call=llm_call,
        batch_size=1,
        min_score=0.3,
        on_progress=on_progress,
        trust_level="med"
    )

    print(f"\n{'='*60}")
    print("STRESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total extracted: {report.total_extracted}")
    print(f"Total verified: {report.total_verified}")
    print(f"Total used: {report.total_used}")
    print(f"Sections: {len(report.sections)}")

    for i, section in enumerate(report.sections):
        print(f"\n  [{i+1}] {section.theme}")
        print(f"      Facts: {len(section.facts)}")
        print(f"      Citations: {len(section.citations)}")
        print(f"      Prose: {len(section.prose)} chars")

    # Render to HTML using existing render module
    html = render_report(report, trust_level="med")
    output_path = project_root / "stress_test_report.html"
    output_path.write_text(html)
    print(f"\n  Report saved to: {output_path}")

    # Open in browser
    import subprocess
    subprocess.run(["open", str(output_path)])


if __name__ == "__main__":
    asyncio.run(main())
