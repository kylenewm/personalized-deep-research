#!/usr/bin/env python3
"""Test the pointer extraction approach with real sources.

This script:
1. Loads sources from a saved state file
2. Uses LLM to generate pointers (what to extract)
3. Uses code to extract actual text
4. Reports verification rate
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

from open_deep_research.pointer_extract import (
    POINTER_PROMPT,
    Pointer,
    extract_from_pointer,
    format_extraction_markdown,
    format_sources_for_prompt,
    parse_pointer_response,
)


def load_sources(state_file: str, max_sources: int = 10) -> dict:
    """Load sources from state file."""
    with open(state_file) as f:
        state = json.load(f)

    sources = {}
    source_store = state.get("source_store", [])

    for i, src in enumerate(source_store[:max_sources]):
        src_id = f"src_{i:03d}"
        sources[src_id] = {
            "content": src.get("content", ""),
            "url": src.get("url", ""),
            "title": src.get("title", "Unknown"),
        }

    return sources


def get_pointers_from_llm(sources: dict, topic: str, model: str = "gpt-4.1-mini") -> list:
    """Use LLM to generate pointers."""
    client = OpenAI()

    formatted_sources = format_sources_for_prompt(sources, max_chars=3000)
    prompt = POINTER_PROMPT.format(sources=formatted_sources, topic=topic)

    print(f"[LLM] Generating pointers with {model}...")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3,  # Lower temp for more consistent extraction
    )

    raw_response = response.choices[0].message.content
    print(f"[LLM] Response length: {len(raw_response)} chars")

    pointers = parse_pointer_response(raw_response)
    print(f"[LLM] Parsed {len(pointers)} pointers")

    return pointers


def main():
    # Load sources
    state_file = Path(__file__).parent.parent / "run_state_1767563291.json"

    if not state_file.exists():
        print(f"State file not found: {state_file}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("POINTER EXTRACTION TEST")
    print(f"{'='*60}")

    # Load first 10 sources for test
    print("\n[1/4] Loading sources...")
    sources = load_sources(state_file, max_sources=10)
    print(f"       Loaded {len(sources)} sources")

    for src_id, src in list(sources.items())[:3]:
        print(f"       - {src_id}: {src['title'][:50]}...")

    # Get pointers from LLM
    print("\n[2/4] Getting pointers from LLM...")
    topic = "AI safety developments, regulations, and technical advances"
    pointers = get_pointers_from_llm(sources, topic)

    if not pointers:
        print("       No pointers generated!")
        sys.exit(1)

    print(f"       Generated {len(pointers)} pointers:")
    for p in pointers[:5]:
        print(f"       - {p.source_id}: {p.context} ({len(p.keywords)} keywords)")

    # Extract using code
    print("\n[3/4] Extracting with code...")
    extractions = []
    for pointer in pointers:
        result = extract_from_pointer(pointer, sources)
        extractions.append(result)
        status_icon = {"verified": "✓", "partial": "~", "not_found": "✗"}[result.status]
        print(f"       [{status_icon}] {pointer.context}: {result.status} ({result.match_score:.0%})")

    # Summary
    print("\n[4/4] Results...")
    verified = sum(1 for e in extractions if e.status == "verified")
    partial = sum(1 for e in extractions if e.status == "partial")
    not_found = sum(1 for e in extractions if e.status == "not_found")
    total = len(extractions)

    print(f"\n{'='*60}")
    print("EXTRACTION RESULTS")
    print(f"{'='*60}")
    print(f"  Verified:  {verified}/{total} ({verified/total*100:.1f}%)")
    print(f"  Partial:   {partial}/{total} ({partial/total*100:.1f}%)")
    print(f"  Not Found: {not_found}/{total} ({not_found/total*100:.1f}%)")
    print(f"{'='*60}")

    # Show sample output
    print("\n--- Sample Verified Extractions ---\n")
    verified_extractions = [e for e in extractions if e.status == "verified"][:5]
    print(format_extraction_markdown(verified_extractions, use_color=False))

    if not_found > 0:
        print("\n--- Failed Extractions ---\n")
        failed = [e for e in extractions if e.status == "not_found"][:3]
        for e in failed:
            print(f"  ✗ {e.pointer.context}")
            print(f"    Keywords: {e.pointer.keywords}")
            print()

    # Compare to baseline
    print(f"\n{'='*60}")
    print("COMPARISON TO BASELINE")
    print(f"{'='*60}")
    print(f"  Baseline (B+C):     27% citation validity")
    print(f"  Pointer Extract:    {verified/total*100:.0f}% verified extraction")
    print(f"{'='*60}")

    return verified / total if total > 0 else 0


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result > 0.5 else 1)
