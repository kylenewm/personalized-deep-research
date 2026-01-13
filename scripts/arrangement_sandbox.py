#!/usr/bin/env python3
"""Arrangement sandbox - test fact grouping and curation quality.

Measures:
1. Theme coherence: do facts match their assigned theme?
2. Exclusion rate: what % of facts are dropped?
3. Theme balance: std dev of theme sizes
4. Coverage: all facts accounted for (grouped or excluded)?

Usage:
    python scripts/arrangement_sandbox.py                                      # Test arrangement
    python scripts/arrangement_sandbox.py --dry                                # Show test facts without LLM
    python scripts/arrangement_sandbox.py --fixture tests/fixtures/arrangement/sample.json
"""

import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


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


# Mix of good facts, marketing fluff, and irrelevant content
TEST_FACTS = [
    # Good facts - specific, useful
    "Hamming runs 10,000 concurrent test calls with sub-200ms latency",
    "Coval integrates with GitHub Actions for automated regression testing on every commit",
    "Roark captures actual production calls and replays them against updated agent logic",
    "Future AGI Simulate creates synthetic customers that interrupt, change topics, and express frustration",
    "Braintrust offers SOC 2 Type II certification with HIPAA-ready compliance",
    "Evalion supports testing in 12 languages including Mandarin, Arabic, and Hindi",
    "The average latency threshold for natural conversation is 500 milliseconds",

    # Marketing fluff - should be excluded
    "We're excited to announce our revolutionary new platform",
    "Learn how to build better voice agents in this comprehensive guide",
    "Our solution offers many powerful features for enterprise teams",
    "Click here to start your free trial today",

    # Vague claims - should be excluded
    "Voice AI is transforming customer service",
    "Testing is important for production systems",
    "The platform provides robust evaluation capabilities",

    # Tangential - might be excluded
    "The global voice AI market is expected to reach $30B by 2026",
    "OpenAI released GPT-4 in March 2023",
]

TEST_TOPIC = "How can I run simulation and evaluations for production grade voice agents?"


async def test_arrangement(facts: list, topic: str, dry_run: bool = False) -> dict:
    """Test arrangement quality."""
    from openai import AsyncOpenAI
    from open_deep_research.pipeline_v2 import ARRANGER_PROMPT

    # Create mock extractions
    extractions = [MockExtraction(fact) for fact in facts]

    if dry_run:
        print(f"\n--- Test Facts ({len(facts)} total) ---")
        for i, f in enumerate(facts, 1):
            print(f"[{i}] {f[:80]}...")
        print(f"\nTopic: {topic}")
        return {"dry_run": True}

    # Format facts for arranger
    facts_text = "\n\n".join([
        f"[{i}] {ext.extracted_text}\n    Source: {ext.pointer.context}"
        for i, ext in enumerate(extractions, 1)
    ])

    prompt = ARRANGER_PROMPT.format(
        topic=topic,
        num_facts=len(facts),
        facts=facts_text,
        source_quality_guidance=""  # Empty for sandbox testing
    )

    client = AsyncOpenAI()

    print(f"Testing arrangement with {len(facts)} facts...")
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.3
    )
    response = resp.choices[0].message.content

    # Parse response
    groups = []
    excluded = []
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            data = json.loads(match.group())
            groups = data.get("groups", [])
            excluded = data.get("excluded", [])
    except json.JSONDecodeError:
        print(f"Failed to parse response: {response[:200]}")

    # Calculate metrics
    total_facts = len(facts)

    # Grouped facts
    grouped_ids = set()
    theme_sizes = []
    for g in groups:
        ids = set(g.get("fact_ids", []))
        grouped_ids.update(ids)
        theme_sizes.append(len(ids))

    # Excluded facts
    excluded_ids = set(e.get("id") for e in excluded)

    # Coverage check
    all_accounted = grouped_ids | excluded_ids
    missing = set(range(1, total_facts + 1)) - all_accounted
    duplicates = len(grouped_ids & excluded_ids)

    # Theme balance (std dev of sizes)
    if theme_sizes:
        mean_size = sum(theme_sizes) / len(theme_sizes)
        variance = sum((s - mean_size) ** 2 for s in theme_sizes) / len(theme_sizes)
        std_dev = variance ** 0.5
    else:
        std_dev = 0

    return {
        "total_facts": total_facts,
        "num_themes": len(groups),
        "grouped_count": len(grouped_ids),
        "excluded_count": len(excluded_ids),
        "exclusion_rate": len(excluded_ids) / total_facts if total_facts > 0 else 0,
        "missing_count": len(missing),
        "missing_ids": sorted(missing),
        "duplicate_count": duplicates,
        "theme_sizes": theme_sizes,
        "theme_balance_std": std_dev,
        "themes": [{"name": g.get("theme"), "count": len(g.get("fact_ids", []))} for g in groups],
        "excluded_reasons": [{"id": e.get("id"), "reason": e.get("reason", "")[:50]} for e in excluded[:5]],
        "response_preview": response[:300]
    }


def print_results(results: dict):
    """Print arrangement quality results."""
    print(f"\n{'='*60}")
    print("ARRANGEMENT QUALITY RESULTS")
    print(f"{'='*60}")

    print(f"\n  Total Facts: {results['total_facts']}")
    print(f"  Themes Created: {results['num_themes']}")
    print(f"  Facts Grouped: {results['grouped_count']}")
    print(f"  Facts Excluded: {results['excluded_count']} ({results['exclusion_rate']:.1%})")

    # Quality checks
    checks = []

    # Coverage
    if results["missing_count"] == 0:
        checks.append(("Coverage", "✅", "All facts accounted for"))
    else:
        checks.append(("Coverage", "❌", f"Missing: {results['missing_ids']}"))

    # No duplicates
    if results["duplicate_count"] == 0:
        checks.append(("No Duplicates", "✅", "Each fact in exactly one place"))
    else:
        checks.append(("No Duplicates", "❌", f"{results['duplicate_count']} duplicates"))

    # Exclusion rate (target: 20-50% for this test set)
    rate = results["exclusion_rate"]
    if 0.2 <= rate <= 0.6:
        checks.append(("Exclusion Rate", "✅", f"{rate:.1%} (expected 20-60% for test data)"))
    else:
        checks.append(("Exclusion Rate", "⚠️", f"{rate:.1%} (expected 20-60%)"))

    # Theme balance
    std = results["theme_balance_std"]
    if std < 3:
        checks.append(("Theme Balance", "✅", f"std dev = {std:.1f}"))
    else:
        checks.append(("Theme Balance", "⚠️", f"std dev = {std:.1f} (unbalanced)"))

    print(f"\n  Quality Checks:")
    for name, status, detail in checks:
        print(f"    {status} {name}: {detail}")

    print(f"\n  Themes:")
    for t in results["themes"]:
        print(f"    - {t['name']}: {t['count']} facts")

    if results["excluded_reasons"]:
        print(f"\n  Sample Exclusions:")
        for e in results["excluded_reasons"]:
            print(f"    [{e['id']}] {e['reason']}")

    # Overall pass/fail
    passed = (
        results["missing_count"] == 0 and
        results["duplicate_count"] == 0 and
        0.1 <= results["exclusion_rate"] <= 0.7
    )
    print(f"\n  Status: {'✅ PASS' if passed else '❌ FAIL'}")


async def run_sandbox(dry_run: bool = False, fixture_path: str = None):
    """Run arrangement sandbox."""
    print("Arrangement Quality Sandbox")
    print("="*60)

    # Load facts - from fixture or default
    if fixture_path:
        fixture_file = Path(fixture_path)
        if not fixture_file.exists():
            print(f"Error: {fixture_path} not found")
            return
        with open(fixture_file) as f:
            data = json.load(f)
        facts = data.get("facts_before_arrangement", data.get("facts", []))
        # Handle facts as list of strings or list of dicts
        if facts and isinstance(facts[0], dict):
            facts = [f.get("extracted_text", "") for f in facts]
        topic = data.get("topic", data.get("query", TEST_TOPIC))
        print(f"Loaded {len(facts)} facts from {fixture_file.name}")
    else:
        facts = TEST_FACTS
        topic = TEST_TOPIC

    result = await test_arrangement(facts, topic, dry_run)

    if not result.get("dry_run"):
        print_results(result)

        # Log results
        log_path = project_root / "arrangement_sandbox_log.jsonl"
        log_entry = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "total_facts": result["total_facts"],
            "exclusion_rate": result["exclusion_rate"],
            "num_themes": result["num_themes"],
            "missing": result["missing_count"],
            "duplicates": result["duplicate_count"]
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Arrangement sandbox")
    parser.add_argument("--dry", action="store_true", help="Show test facts without LLM")
    parser.add_argument("--fixture", "-f", help="Path to fixture file")

    args = parser.parse_args()
    asyncio.run(run_sandbox(dry_run=args.dry, fixture_path=args.fixture))


if __name__ == "__main__":
    main()
