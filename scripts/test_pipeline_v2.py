#!/usr/bin/env python3
"""Test pipeline v2 with real sources.

Runs the three-stage safeguarded generation:
1. Batched pointer extraction
2. Arranger grouping + curation
3. Per-theme synthesis
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import AsyncOpenAI

# Direct imports to avoid package init
import importlib.util

src_dir = Path(__file__).parent.parent / "src" / "open_deep_research"

def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

pointer_extract = load_mod("pointer_extract", src_dir / "pointer_extract.py")
sys.modules["open_deep_research.pointer_extract"] = pointer_extract

pipeline_v2 = load_mod("pipeline_v2", src_dir / "pipeline_v2.py")


async def main():
    # Config
    MAX_SOURCES = 141  # All sources
    MODEL = "gpt-4.1-mini"

    print("=" * 70)
    print("PIPELINE V2: Full 141 Sources")
    print("=" * 70)

    # Load sources
    state_file = Path(__file__).parent.parent / "run_state_1767563291.json"
    with open(state_file) as f:
        state = json.load(f)

    sources = {}
    for i, src in enumerate(state.get("source_store", [])[:MAX_SOURCES]):
        content = src.get("content", "")
        if content:  # Only include sources with content
            sources[f"src_{i:03d}"] = {
                "content": content,
                "url": src.get("url", ""),
                "title": src.get("title", "Unknown"),
            }

    print(f"\nLoaded {len(sources)} sources with content")

    # Setup LLM
    client = AsyncOpenAI()

    async def llm_call(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3
        )
        return resp.choices[0].message.content

    # Run pipeline
    topic = "AI safety developments, governance, regulatory frameworks, and technical advances in 2025"
    title = "AI Safety Developments 2025"

    def on_progress(stage: str, msg: str):
        print(f"[{stage}] {msg}")

    try:
        report = await pipeline_v2.run_pipeline_v2(
            sources=sources,
            topic=topic,
            title=title,
            llm_call=llm_call,
            on_progress=on_progress
        )

        # Render and save
        print("\n" + "=" * 70)
        print("REPORT GENERATED")
        print("=" * 70)

        md_output = pipeline_v2.render_hybrid_report(report, use_color=True)

        output_path = Path(__file__).parent.parent / "pipeline_v2_output.md"
        output_path.write_text(md_output)
        print(f"\n[Saved to {output_path}]")

        # Stats
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        print(f"  Sources processed:  {report.total_extracted}")
        print(f"  Verified facts:     {report.total_verified}")
        print(f"  Facts in report:    {report.total_used}")
        print(f"  Themes:             {len(report.sections)}")
        for section in report.sections:
            print(f"    - {section.theme}: {len(section.facts)} facts")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
