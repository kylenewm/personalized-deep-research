#!/usr/bin/env python3
"""Quick test: Search for voice models + run pipeline v2."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import os
from openai import AsyncOpenAI
from tavily import TavilyClient

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
    query = "What are the best voice to voice AI models in 2025 and why"

    print("=" * 70)
    print("VOICE MODELS TEST: Fresh Search + Pipeline v2")
    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    # Search with Tavily
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    print("\n[SEARCH] Searching for voice model content...")

    # Multiple search queries for better coverage
    searches = [
        "best voice to voice AI models 2025",
        "top speech to speech models comparison 2025",
        "voice cloning TTS models 2025 review",
    ]

    all_results = []
    for search_query in searches:
        print(f"  Searching: {search_query}")
        results = tavily.search(
            query=search_query,
            search_depth="advanced",
            max_results=10,
            include_raw_content=True,
        )
        all_results.extend(results.get("results", []))

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    print(f"\n[SEARCH] Found {len(unique_results)} unique sources")

    # Build sources dict
    sources = {}
    for i, result in enumerate(unique_results):
        content = result.get("raw_content") or result.get("content", "")
        if content and len(content) > 100:
            sources[f"src_{i:03d}"] = {
                "content": content,
                "url": result.get("url", ""),
                "title": result.get("title", "Unknown"),
            }

    print(f"[SEARCH] {len(sources)} sources with content")

    if len(sources) < 5:
        print("[ERROR] Not enough sources found")
        return

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
    def on_progress(stage: str, msg: str):
        print(f"[{stage}] {msg}")

    try:
        report = await pipeline_v2.run_pipeline_v2(
            sources=sources,
            topic=query,
            title="Best Voice-to-Voice AI Models 2025",
            llm_call=llm_call,
            on_progress=on_progress
        )

        # Render HTML
        html_output = pipeline_v2.render_html(report)
        output_path = Path(__file__).parent.parent / "voice_models_report.html"
        output_path.write_text(html_output)
        print(f"\n[Saved to {output_path}]")

        # Stats
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        print(f"  Sources searched:   {len(sources)}")
        print(f"  Extractions:        {report.total_extracted}")
        print(f"  Verified facts:     {report.total_verified}")
        print(f"  Facts in report:    {report.total_used}")
        print(f"  Themes:             {len(report.sections)}")
        for section in report.sections:
            print(f"    - {section.theme}: {len(section.facts)} facts")
        print("=" * 70)

        # Open in browser
        import subprocess
        subprocess.run(["open", str(output_path)])

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
