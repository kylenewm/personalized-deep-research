#!/usr/bin/env python3
"""
TEMP TEST: Direct quote extraction vs keyword/Jaccard approach

Hypothesis: LLM outputs exact quote, code verifies substring = simpler, same safety

Delete this file after testing.
"""

import json
import os
from pathlib import Path

# Use OpenAI API (available in this project)
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Load fixture data
FIXTURE_PATH = Path(__file__).parent.parent / "tests/fixtures/gold_queries/voice_pm.json"


def load_sources(limit: int = 10) -> list[dict]:
    """Load sources from fixture."""
    with open(FIXTURE_PATH) as f:
        data = json.load(f)

    sources = []
    source_list = data.get("source_store", [])
    for i, src_data in enumerate(source_list[:limit * 2]):  # Check more to find good ones
        content = src_data.get("content", "")
        if len(content) > 500 and len(sources) < limit:  # Skip tiny sources
            sources.append({
                "id": f"src_{i:03d}",
                "url": src_data.get("url", ""),
                "title": src_data.get("title", ""),
                "content": content[:8000]  # Truncate for API
            })
    return sources


# =============================================================================
# APPROACH 1: Current (keyword + Jaccard) - simplified recreation
# =============================================================================

KEYWORD_PROMPT = """Extract key facts from this source about voice/speech AI models.

For each fact, output keywords that identify WHERE in the source to find it.

Source ({source_id}):
{content}

Output JSON array:
[
  {{"keywords": ["keyword1", "keyword2", "keyword3"], "description": "what this fact is about"}}
]

Rules:
- Use 3-5 single keywords per fact
- Keywords must appear in the source
- Max 3 facts per source
"""


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def find_by_keywords(content: str, keywords: list[str], window_size: int = 100) -> tuple[str, float]:
    """Find best matching window using keyword/Jaccard approach."""
    words = content.lower().split()
    keyword_set = set(k.lower() for k in keywords)

    best_window = ""
    best_score = 0.0

    for i in range(len(words) - window_size + 1):
        window_words = set(words[i:i + window_size])
        score = jaccard_similarity(keyword_set, window_words)
        if score > best_score:
            best_score = score
            # Reconstruct approximate window from original
            start_idx = content.lower().find(words[i])
            if start_idx >= 0:
                best_window = content[start_idx:start_idx + 500]

    return best_window, best_score


def extract_keyword_approach(sources: list[dict]) -> list[dict]:
    """Current approach: LLM outputs keywords, code searches."""
    results = []

    for src in sources:
        prompt = KEYWORD_PROMPT.format(source_id=src["id"], content=src["content"][:4000])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            # Parse JSON from response
            text = response.choices[0].message.content
            # Find JSON array in response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                extractions = json.loads(text[start:end])

                for ext in extractions[:3]:
                    keywords = ext.get("keywords", [])
                    window, score = find_by_keywords(src["content"], keywords)

                    # Verify: is the extracted window actually in source?
                    verified = window[:100] in src["content"] if window else False

                    results.append({
                        "source_id": src["id"],
                        "approach": "keyword_jaccard",
                        "keywords": keywords,
                        "extracted": window[:200] if window else "",
                        "score": score,
                        "verified": verified
                    })
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"  Error parsing keywords for {src['id']}: {e}")

    return results


# =============================================================================
# APPROACH 2: Direct quote extraction (simpler)
# =============================================================================

DIRECT_QUOTE_PROMPT = """Extract key facts from this source about voice/speech AI models.

For each fact, output the EXACT text from the source (verbatim quote).

Source ({source_id}):
{content}

Output JSON array:
[
  {{"quote": "exact text copied from source", "topic": "what this is about"}}
]

Rules:
- Copy text EXACTLY as it appears (including punctuation)
- Each quote should be 1-3 sentences (50-200 chars)
- Max 3 quotes per source
- Only include factual information, not navigation/metadata
"""


def extract_direct_quote_approach(sources: list[dict]) -> list[dict]:
    """Simpler approach: LLM outputs exact quote, code verifies substring."""
    results = []

    for src in sources:
        prompt = DIRECT_QUOTE_PROMPT.format(source_id=src["id"], content=src["content"][:4000])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.choices[0].message.content
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                extractions = json.loads(text[start:end])

                for ext in extractions[:3]:
                    quote = ext.get("quote", "")

                    # Simple verification: is quote a substring of source?
                    verified = quote in src["content"]

                    # If not exact match, try normalized (whitespace)
                    if not verified:
                        normalized_quote = " ".join(quote.split())
                        normalized_content = " ".join(src["content"].split())
                        verified = normalized_quote in normalized_content

                    results.append({
                        "source_id": src["id"],
                        "approach": "direct_quote",
                        "quote": quote[:200],
                        "verified": verified,
                        "exact_match": quote in src["content"]
                    })
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"  Error parsing quotes for {src['id']}: {e}")

    return results


# =============================================================================
# Compare approaches
# =============================================================================

def main():
    print("=" * 60)
    print("TEMP TEST: Direct Quote vs Keyword/Jaccard")
    print("=" * 60)

    if not FIXTURE_PATH.exists():
        print(f"ERROR: Fixture not found at {FIXTURE_PATH}")
        print("Run sandbox capture first: python scripts/sandbox_pipeline.py --capture 'voice models' --name voice_pm")
        return

    sources = load_sources(limit=5)  # Small test
    print(f"\nLoaded {len(sources)} sources from fixture\n")

    # Test both approaches
    print("-" * 40)
    print("APPROACH 1: Keyword + Jaccard")
    print("-" * 40)
    keyword_results = extract_keyword_approach(sources)

    keyword_verified = sum(1 for r in keyword_results if r["verified"])
    print(f"\nResults: {keyword_verified}/{len(keyword_results)} verified ({100*keyword_verified/len(keyword_results):.0f}%)")

    for r in keyword_results[:3]:
        print(f"  - Keywords: {r['keywords']}")
        print(f"    Score: {r['score']:.2f}, Verified: {r['verified']}")
        print(f"    Extract: {r['extracted'][:80]}...")
        print()

    print("-" * 40)
    print("APPROACH 2: Direct Quote")
    print("-" * 40)
    quote_results = extract_direct_quote_approach(sources)

    quote_verified = sum(1 for r in quote_results if r["verified"])
    exact_match = sum(1 for r in quote_results if r.get("exact_match"))
    print(f"\nResults: {quote_verified}/{len(quote_results)} verified ({100*quote_verified/len(quote_results):.0f}%)")
    print(f"Exact matches: {exact_match}/{len(quote_results)}")

    for r in quote_results[:3]:
        print(f"  - Quote: {r['quote'][:80]}...")
        print(f"    Verified: {r['verified']}, Exact: {r.get('exact_match')}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Keyword/Jaccard: {keyword_verified}/{len(keyword_results)} verified")
    print(f"Direct Quote:    {quote_verified}/{len(quote_results)} verified (exact: {exact_match})")
    print()

    if quote_verified >= keyword_verified:
        print("✓ Direct quote approach works at least as well")
        print("  → Simpler code, same safety guarantee")
    else:
        print("✗ Keyword approach had better verification rate")
        print("  → May need prompt tuning for direct quote")

    # Save results for inspection
    output_path = Path(__file__).parent / "temp_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "keyword_results": keyword_results,
            "quote_results": quote_results,
            "summary": {
                "keyword_verified": keyword_verified,
                "keyword_total": len(keyword_results),
                "quote_verified": quote_verified,
                "quote_total": len(quote_results),
                "quote_exact": exact_match
            }
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
