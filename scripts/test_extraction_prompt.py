"""Fast extraction prompt testing - single batch, <30 seconds."""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_deep_research.configuration import Configuration
from open_deep_research.pointer_extract import parse_pointer_response
from langchain.chat_models import init_chat_model

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "gold_queries"

# Cache formatted batch to skip loading every time
BATCH_CACHE = Path(__file__).parent.parent / "sandbox_output" / "batch_cache.json"


def get_test_batch(fixture_name: str = "voice_pm", batch_idx: int = 0, batch_size: int = 10):
    """Load and cache a single batch for testing."""
    if BATCH_CACHE.exists():
        with open(BATCH_CACHE) as f:
            return json.load(f)

    # Load fixture
    fixture_path = FIXTURES_DIR / f"{fixture_name}.json"
    with open(fixture_path) as f:
        state = json.load(f)

    # Get one batch
    sources = state["source_store"][batch_idx * batch_size:(batch_idx + 1) * batch_size]

    # Format for prompt (same as format_sources_for_prompt)
    formatted_sources = []
    for i, src in enumerate(sources):
        src_id = f"src_{i:03d}"
        content = src.get("content", "")[:5000]  # MAX_CHARS_PER_SOURCE
        title = src.get("title", "Unknown")
        formatted_sources.append(f"[{src_id}] {title}\n{content}\n")

    batch_data = {
        "topic": state["query"],
        "formatted": "\n---\n".join(formatted_sources),
        "source_count": len(sources)
    }

    # Cache it
    BATCH_CACHE.parent.mkdir(exist_ok=True)
    with open(BATCH_CACHE, "w") as f:
        json.dump(batch_data, f)

    return batch_data


# Different prompts to test
PROMPTS = {
    "current": '''Extract facts from these sources that DIRECTLY answer: {topic}

RELEVANCE CHECK (critical):
- Only extract facts that help answer the specific question
- Skip sources that don't contain relevant information
- Skip generic/promotional content ("We'll show you how to...", "In this article...")
- Skip tutorial intros, marketing claims, and filler text

For each RELEVANT fact, output:
- source_id: Match exactly (e.g., "src_001")
- keywords: 3-5 SINGLE words that appear in that source (not phrases)
- context: What this fact is about (3-5 words)
- relevance: 1-5 score (5=directly answers question, 3=somewhat relevant, 1=tangential)

ONLY include facts with relevance >= 3.

CRITICAL: Use single distinctive words, not phrases. Example:
- Good: ["Biden", "October", "2023", "Executive", "Order"]
- Bad: ["Executive Order", "October 2023"] (these are phrases)

Sources:
{sources}

Output JSON array:
[
  {{"source_id": "src_001", "keywords": ["latency", "200ms", "L40S"], "context": "Speech model latency", "relevance": 5}},
  {{"source_id": "src_002", "keywords": ["ElevenLabs", "cloning", "accuracy"], "context": "Voice cloning quality", "relevance": 4}}
]

Output ONLY the JSON array. Skip sources with no relevant facts.''',

    "aggressive": '''Extract ALL factual claims from these sources about: {topic}

For EACH source, extract 3-5 distinct facts. Be thorough - we want comprehensive coverage.

For each fact:
- source_id: Match exactly (e.g., "src_001")
- keywords: 3-5 SINGLE words from that exact sentence
- context: What this fact covers

Sources:
{sources}

Output JSON array. Extract MORE facts - aim for 3-5 per source:
[{{"source_id": "src_001", "keywords": ["word1", "word2", "word3"], "context": "description"}}]''',

    "numbered": '''Extract facts from these sources about: {topic}

REQUIREMENT: Extract AT LEAST 2 facts per source. Each source has useful information.

For each fact output:
- source_id: e.g., "src_001"
- keywords: 3-5 single words from that sentence
- context: 3-5 word description

Sources:
{sources}

Output JSON array with 2+ facts per source:
[{{"source_id": "src_001", "keywords": ["word1", "word2"], "context": "desc"}}]'''
}


async def test_prompt(prompt_name: str):
    """Test a single prompt variant."""
    batch = get_test_batch()
    prompt_template = PROMPTS[prompt_name]
    prompt = prompt_template.format(topic=batch["topic"], sources=batch["formatted"])

    config = Configuration()
    model = init_chat_model(
        model=config.research_model,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4000
    )

    print(f"\n[{prompt_name}] Testing ({batch['source_count']} sources)...")
    response = await model.ainvoke(prompt)

    pointers = parse_pointer_response(response.content, min_relevance=1)  # Get all

    print(f"[{prompt_name}] Pointers: {len(pointers)} ({len(pointers)/batch['source_count']:.1f} per source)")

    # Show breakdown by source
    by_source = {}
    for p in pointers:
        by_source[p.source_id] = by_source.get(p.source_id, 0) + 1

    print(f"[{prompt_name}] By source: {dict(sorted(by_source.items()))}")

    return len(pointers)


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="all", help="Prompt to test (current/aggressive/numbered/all)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear batch cache")
    args = parser.parse_args()

    if args.clear_cache and BATCH_CACHE.exists():
        BATCH_CACHE.unlink()
        print("[CACHE] Cleared")

    if args.prompt == "all":
        results = {}
        for name in PROMPTS:
            results[name] = await test_prompt(name)
        print(f"\n{'='*50}")
        print("RESULTS:")
        for name, count in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count} pointers")
    else:
        await test_prompt(args.prompt)


if __name__ == "__main__":
    asyncio.run(main())
