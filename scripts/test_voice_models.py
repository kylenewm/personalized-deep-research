#!/usr/bin/env python3
"""Test pipeline v2 with voice models query - same as report_preview.html source."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import AsyncOpenAI

# Direct imports
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
    # Try to find existing voice model state, otherwise use AI safety
    state_file = Path(__file__).parent.parent / "run_state_1767563291.json"

    # Check for quality fix report state
    quality_fix = Path(__file__).parent.parent / "quality_fix_state.json"
    if quality_fix.exists():
        state_file = quality_fix

    print("=" * 70)
    print("PIPELINE V2 TEST WITH CLEANUP")
    print("=" * 70)

    with open(state_file) as f:
        state = json.load(f)

    # Load sources
    sources = {}
    source_list = state.get("source_store", state.get("sources", []))
    for i, src in enumerate(source_list[:100]):  # Limit for faster test
        content = src.get("content", "")
        if content:
            sources[f"src_{i:03d}"] = {
                "content": content,
                "url": src.get("url", ""),
                "title": src.get("title", "Unknown"),
            }

    print(f"\nLoaded {len(sources)} sources")

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

    # Run pipeline
    topic = "What are the best voice to voice models in 2025 and why"
    title = "Voice-to-Voice AI Models 2025"

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

        # Render HTML
        html_output = pipeline_v2.render_html(report)

        output_path = Path(__file__).parent.parent / "report_preview_v2.html"
        output_path.write_text(html_output)
        print(f"\n[Saved HTML to {output_path}]")

        # Also save markdown
        md_output = pipeline_v2.render_hybrid_report(report, use_color=False)
        md_path = Path(__file__).parent.parent / "report_preview_v2.md"
        md_path.write_text(md_output)
        print(f"[Saved MD to {md_path}]")

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

        # Show first few facts to verify cleanup worked
        print("\n" + "=" * 70)
        print("SAMPLE CLEANED FACTS")
        print("=" * 70)
        count = 0
        for section in report.sections:
            for fact in section.facts[:2]:
                count += 1
                print(f"\n[{count}] {fact.extracted_text[:150]}...")
                if count >= 6:
                    break
            if count >= 6:
                break

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
