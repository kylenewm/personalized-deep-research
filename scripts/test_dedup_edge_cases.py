#!/usr/bin/env python3
"""Test dedup with specific edge cases."""

import asyncio
import json
import sys
from pathlib import Path
from dataclasses import dataclass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@dataclass
class MockPointer:
    context: str = "example.com"


@dataclass
class MockExtraction:
    extracted_text: str
    match_score: float = 1.0
    source_url: str = "http://example.com"
    status: str = "verified"
    pointer: MockPointer = None

    def __post_init__(self):
        if self.pointer is None:
            self.pointer = MockPointer()


# Edge cases for dedup
EDGE_CASES = [
    # Numbers that look similar but are different
    ("10,000 concurrent users with 200ms latency", "10000 concurrent users with 200 millisecond latency", True, "Same numbers, different format"),
    ("10,000 concurrent users with 200ms latency", "5,000 concurrent users with 180ms latency", False, "Different numbers"),

    # Same metric, different subjects (CRITICAL - must be different)
    ("Claude achieved 95% accuracy on MMLU", "GPT-4 achieved 95% accuracy on MMLU", False, "Same metric, different models"),
    ("LiveKit delivers sub-100ms latency", "Agora delivers sub-100ms latency", False, "Same metric, different companies"),
    ("Hamming runs 10k tests", "Coval runs 10k tests", False, "Same metric, different companies"),

    # Paraphrased duplicates (should be caught)
    ("Latency should stay under 500ms for natural conversation", "Response time must remain below 500 milliseconds for natural dialogue", True, "Paraphrased"),
    ("SOC 2 Type II certified with HIPAA compliance", "SOC 2 Type II certified, HIPAA-ready", True, "Same certifications"),

    # Similar but different features
    ("Supports 12 languages including Mandarin", "Supports 8 languages including Spanish", False, "Different language counts"),
    ("Runs 1000 tests per minute", "Runs 500 tests per hour", False, "Different rates"),

    # Marketing vs substance
    ("We're excited to announce our new platform", "Our revolutionary new solution is here", True, "Both marketing fluff"),
    ("Platform processes 50k requests/second", "System handles 50000 requests per second", True, "Same metric"),
]


async def test_edge_cases():
    from openai import AsyncOpenAI
    from open_deep_research.pipeline_v2 import (
        DEDUP_PROMPT,
        format_facts_for_dedup,
        parse_dedup_response,
        deduplicate_extractions_llm
    )

    print("Testing dedup edge cases...")
    print("="*60)

    # Create extractions from edge cases
    extractions = []
    for i, (fact_a, fact_b, expected, reason) in enumerate(EDGE_CASES):
        extractions.append(MockExtraction(fact_a, match_score=1.0))
        extractions.append(MockExtraction(fact_b, match_score=0.9))

    print(f"Created {len(extractions)} facts from {len(EDGE_CASES)} edge case pairs")

    client = AsyncOpenAI()

    async def llm_call(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1
        )
        return resp.choices[0].message.content

    # Run dedup
    print("\nCalling LLM dedup...")
    deduped = await deduplicate_extractions_llm(extractions, llm_call, batch_size=30)

    print(f"Deduped: {len(extractions)} -> {len(deduped)} facts")

    # Check which pairs were marked as duplicates
    kept_texts = {e.extracted_text for e in deduped}

    print("\n" + "="*60)
    print("EDGE CASE RESULTS")
    print("="*60)

    correct = 0
    total = len(EDGE_CASES)

    for i, (fact_a, fact_b, expected_dup, reason) in enumerate(EDGE_CASES):
        a_kept = fact_a in kept_texts
        b_kept = fact_b in kept_texts

        # If expected duplicate: only one should be kept (or both if different scores)
        # If expected different: both should be kept
        if expected_dup:
            # Duplicate: at least one should be removed
            actual_dup = not (a_kept and b_kept)
        else:
            # Different: both should be kept
            actual_dup = not (a_kept and b_kept)

        # For proper check: if expected_dup=True, we expect exactly one removed
        # If expected_dup=False, we expect both kept
        if expected_dup:
            is_correct = not (a_kept and b_kept)  # At least one removed
        else:
            is_correct = a_kept and b_kept  # Both kept

        status = "✅" if is_correct else "❌"
        if is_correct:
            correct += 1

        print(f"\n{status} Case {i+1}: {reason}")
        print(f"   Expected: {'DUPLICATE' if expected_dup else 'DIFFERENT'}")
        print(f"   A kept: {a_kept}, B kept: {b_kept}")
        if not is_correct:
            print(f"   A: {fact_a[:50]}...")
            print(f"   B: {fact_b[:50]}...")

    print(f"\n{'='*60}")
    print(f"Score: {correct}/{total} ({100*correct/total:.0f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(test_edge_cases())
