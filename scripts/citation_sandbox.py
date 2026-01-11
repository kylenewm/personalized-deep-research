#!/usr/bin/env python3
"""Citation sandbox - test synthesis citation quality.

Measures:
1. Citation rate: % of provided facts that get cited
2. Citation correctness: do [N] markers reference the right fact?

Usage:
    python scripts/citation_sandbox.py           # Test citation quality
    python scripts/citation_sandbox.py --dry     # Show test facts without LLM call
"""

import asyncio
import json
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


# Sample facts for testing citation - diverse topics to test alignment
TEST_FACTS = [
    {
        "theme": "Performance Metrics",
        "facts": [
            "Hamming achieves sub-200ms latency for voice agent testing at scale",
            "Coval processes 10,000 concurrent test simulations with 99.9% uptime",
            "Roark captures 95% of edge cases through adversarial scenario generation",
            "Braintrust integrates with CI/CD pipelines for automated regression testing",
            "Future AGI Simulate runs a month of interactions in 5 minutes",
        ]
    },
    {
        "theme": "Security & Compliance",
        "facts": [
            "Hamming is SOC 2 Type II certified with HIPAA-ready compliance",
            "Roark offers enterprise SSO and role-based access controls",
            "All data is encrypted at rest using AES-256 and in transit using TLS 1.3",
            "Coval provides audit logs with 90-day retention for compliance reviews",
        ]
    },
    {
        "theme": "Multilingual Support",
        "facts": [
            "Evalion supports testing in 12 languages including Mandarin and Arabic",
            "Voice accent simulation covers 8 regional variants of English",
            "Real-time translation maintains under 50ms additional latency",
        ]
    }
]


async def test_synthesis(theme: str, facts: list, dry_run: bool = False) -> dict:
    """Test synthesis for a theme and measure citation quality."""
    from openai import AsyncOpenAI
    from open_deep_research.pipeline_v2 import THEME_SYNTHESIS_PROMPT

    if dry_run:
        print(f"\n--- {theme} ({len(facts)} facts) ---")
        for i, f in enumerate(facts, 1):
            print(f"[{i}] {f}")
        return {"dry_run": True}

    # Format facts as they would appear in synthesis prompt
    facts_text = "\n\n".join([
        f"[{i}] {fact}\n    Source: example.com"
        for i, fact in enumerate(facts, 1)
    ])

    prompt = THEME_SYNTHESIS_PROMPT.format(
        theme=theme,
        topic="voice agent evaluation and testing",
        facts=facts_text
    )

    client = AsyncOpenAI()

    print(f"\nTesting theme: {theme} ({len(facts)} facts)")
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3
    )
    response = resp.choices[0].message.content

    # Parse response
    prose = ""
    cited_ids = []
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            data = json.loads(match.group())
            prose = data.get("prose", "")
            cited_ids = data.get("cited_ids", [])
    except json.JSONDecodeError:
        prose = response

    # Find all citations in prose
    found_citations = re.findall(r'\[(\d+)\]', prose)
    unique_cited = set(int(c) for c in found_citations if c.isdigit())

    # Calculate metrics
    total_facts = len(facts)
    cited_count = len(unique_cited)
    citation_rate = cited_count / total_facts if total_facts > 0 else 0

    # Check for invalid citations (citing facts that don't exist)
    valid_range = set(range(1, total_facts + 1))
    invalid_citations = unique_cited - valid_range
    alignment_errors = len(invalid_citations)

    return {
        "theme": theme,
        "total_facts": total_facts,
        "cited_count": cited_count,
        "citation_rate": citation_rate,
        "unique_cited": sorted(unique_cited),
        "uncited": sorted(valid_range - unique_cited),
        "invalid_citations": sorted(invalid_citations),
        "alignment_errors": alignment_errors,
        "prose_length": len(prose),
        "prose_preview": prose[:200] + "..." if len(prose) > 200 else prose
    }


def print_results(results: list):
    """Print citation quality results."""
    print(f"\n{'='*60}")
    print("CITATION QUALITY RESULTS")
    print(f"{'='*60}")

    total_facts = sum(r["total_facts"] for r in results)
    total_cited = sum(r["cited_count"] for r in results)
    total_errors = sum(r["alignment_errors"] for r in results)

    overall_rate = total_cited / total_facts if total_facts > 0 else 0

    print(f"\n  Overall Citation Rate: {overall_rate:.1%} ({total_cited}/{total_facts} facts)")
    print(f"  Alignment Errors: {total_errors}")
    print(f"  Target: >80% citation rate, 0 alignment errors")

    passed = overall_rate >= 0.8 and total_errors == 0
    print(f"\n  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    print(f"\n{'='*60}")
    print("PER-THEME BREAKDOWN")
    print(f"{'='*60}")

    for r in results:
        status = "✅" if r["citation_rate"] >= 0.8 else "❌"
        print(f"\n{status} {r['theme']}")
        print(f"   Citation Rate: {r['citation_rate']:.1%} ({r['cited_count']}/{r['total_facts']})")
        if r["uncited"]:
            print(f"   Uncited facts: {r['uncited']}")
        if r["invalid_citations"]:
            print(f"   ⚠️ Invalid citations: {r['invalid_citations']}")
        print(f"   Prose: {r['prose_preview'][:100]}...")


async def run_sandbox(dry_run: bool = False):
    """Run citation sandbox."""
    print("Citation Quality Sandbox")
    print("="*60)

    if dry_run:
        for test in TEST_FACTS:
            await test_synthesis(test["theme"], test["facts"], dry_run=True)
        return

    results = []
    for test in TEST_FACTS:
        result = await test_synthesis(test["theme"], test["facts"])
        if not result.get("dry_run"):
            results.append(result)

    print_results(results)

    # Log results
    log_path = project_root / "citation_sandbox_log.jsonl"
    log_entry = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "themes_tested": len(results),
        "overall_rate": sum(r["citation_rate"] for r in results) / len(results) if results else 0,
        "alignment_errors": sum(r["alignment_errors"] for r in results)
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def main():
    dry_run = "--dry" in sys.argv
    asyncio.run(run_sandbox(dry_run))


if __name__ == "__main__":
    main()
