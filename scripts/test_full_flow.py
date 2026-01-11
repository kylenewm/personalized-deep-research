#!/usr/bin/env python3
"""Full flow test: Pointer → Extract → Synthesize → Render

Standalone test - does NOT touch the main pipeline.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

from open_deep_research.pointer_extract import (
    POINTER_PROMPT,
    extract_from_pointer,
    format_sources_for_prompt,
    parse_pointer_response,
)
from open_deep_research.synthesis import (
    SYNTHESIS_PROMPT,
    format_facts_for_synthesis,
    parse_synthesis_response,
    synthesize_report,
    render_report_markdown,
    render_report_plain,
)


def load_sources(state_file: str, max_sources: int = 15) -> dict:
    """Load sources from state file."""
    with open(state_file) as f:
        state = json.load(f)

    sources = {}
    for i, src in enumerate(state.get("source_store", [])[:max_sources]):
        sources[f"src_{i:03d}"] = {
            "content": src.get("content", ""),
            "url": src.get("url", ""),
            "title": src.get("title", "Unknown"),
        }
    return sources


def main():
    client = OpenAI()
    state_file = Path(__file__).parent.parent / "run_state_1767563291.json"

    print("=" * 60)
    print("FULL FLOW TEST: Pointer → Extract → Synthesize")
    print("=" * 60)

    # 1. Load sources
    print("\n[1/5] Loading sources...")
    sources = load_sources(state_file, max_sources=15)
    print(f"       Loaded {len(sources)} sources")

    # 2. Get pointers from LLM
    print("\n[2/5] Getting pointers from LLM...")
    formatted_sources = format_sources_for_prompt(sources, max_chars=2500)
    pointer_prompt = POINTER_PROMPT.format(
        sources=formatted_sources,
        topic="AI safety developments, governance, and technical advances in 2025"
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": pointer_prompt}],
        max_tokens=3000,
        temperature=0.3
    )
    pointers = parse_pointer_response(resp.choices[0].message.content)
    print(f"       Generated {len(pointers)} pointers")

    # 3. Extract with code
    print("\n[3/5] Extracting with code...")
    extractions = []
    for p in pointers:
        result = extract_from_pointer(p, sources)
        extractions.append(result)
        icon = {"verified": "✓", "partial": "~", "not_found": "✗"}[result.status]
        print(f"       [{icon}] {p.context[:50]}...")

    verified = [e for e in extractions if e.status == "verified"]
    print(f"       Verified: {len(verified)}/{len(extractions)}")

    if not verified:
        print("       No verified extractions! Aborting.")
        return

    # 4. Synthesize with LLM
    print("\n[4/5] Generating synthesis...")
    facts_text = format_facts_for_synthesis(verified)
    synth_prompt = SYNTHESIS_PROMPT.format(verified_facts=facts_text)

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": synth_prompt}],
        max_tokens=1500,
        temperature=0.5
    )
    synthesis = parse_synthesis_response(resp.choices[0].message.content, len(verified))
    print(f"       Intro: {len(synthesis['intro'])} chars")
    print(f"       Transitions: {len(synthesis['transitions'])}")
    print(f"       Conclusion: {len(synthesis['conclusion'])} chars")

    # 5. Build report
    print("\n[5/5] Rendering report...")
    report = synthesize_report(
        verified,
        "AI Safety Developments 2025",
        synthesis
    )

    # Output
    print("\n" + "=" * 60)
    print("PLAIN TEXT OUTPUT")
    print("=" * 60 + "\n")
    print(render_report_plain(report))

    # Save markdown
    output_path = Path(__file__).parent.parent / "test_synthesis_output.md"
    md_content = render_report_markdown(report, use_color=True)
    output_path.write_text(md_content)
    print(f"\n[Saved markdown to {output_path}]")

    # Stats
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"  Sources loaded:     {len(sources)}")
    print(f"  Pointers generated: {len(pointers)}")
    print(f"  Verified facts:     {len(verified)}")
    print(f"  Synthesis blocks:   {report.synthesis_count}")
    print(f"  Verification rate:  {len(verified)/len(pointers)*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
