#!/usr/bin/env python3
"""Re-run synthesis on saved run states to test prompt changes.

Skip the 20+ minute research phase, test synthesis in ~30 seconds.

Usage:
    # Re-synthesize and check citation accuracy
    python scripts/resynthesis_test.py run_state_1768263187.json
    python scripts/resynthesis_test.py run_state_1768263187.json --section 0

    # Analyze source authority distribution
    python scripts/resynthesis_test.py run_state_1768263187.json --analyze

    # Save as fixture for future testing
    python scripts/resynthesis_test.py --save-fixture run_state_1768263187.json

    # Test against existing fixture
    python scripts/resynthesis_test.py tests/fixtures/synthesis/example.json
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from openai import AsyncOpenAI

# Copied from pipeline_v2.py to avoid import chain dependencies
THEME_SYNTHESIS_PROMPT = '''You are writing a section of a research report.

Theme: {theme}
Research Topic: {topic}

VERIFIED FACTS (cite using the number in brackets):
{facts}

Write 2-4 paragraphs of flowing prose that synthesizes these facts.

CITATION RULES (STRICT):
1. EVERY factual claim MUST have a citation: "latency is 200ms[1]"
2. Use bracket numbers exactly as shown: [1], [2], [3]
3. Multiple citations allowed: "tools offer fast testing[2][5]"
4. Cite at least 90% of facts - uncited facts are WASTED
5. Transitions/opinions need NO citation: "Overall," "In contrast," "This suggests..."
6. Marketing claims: "claims to be..." or "according to the vendor..."[N]
{source_quality_guidance}
BAD (uncited claim):
  "Hamming achieves sub-200ms latency for voice testing."

GOOD (cited claim):
  "Hamming achieves sub-200ms latency for voice testing[1]."

BAD (over-cited transition):
  "In contrast[1][2], these tools differ significantly[3]."

GOOD (natural transition):
  "In contrast, these tools differ significantly in latency[1] and throughput[2]."

CRITICAL: Use EXACT bracket numbers from the facts. Uncited sentences = unverified opinion.

STYLE:
- Write like a research analyst, not a list maker
- Group related points into paragraphs
- Start with the most important findings

Output ONLY prose paragraphs with [N] citations. No JSON, no headers.'''


CITATION_ACCURACY_PROMPT = '''For each citation below, determine if the claim in the prose accurately reflects the referenced fact.

A citation is ACCURATE if:
- The claim matches the fact's meaning (paraphrasing is OK)
- Numbers/metrics match exactly
- The entity being described matches

A citation is INACCURATE if:
- The claim says something different from the fact
- Numbers/metrics are wrong
- The claim is about a different entity than the fact

Respond with JSON only:
{{"results": [{{"num": 1, "accurate": true}}, {{"num": 2, "accurate": false, "reason": "brief explanation"}}]}}

Citations to check:
{citations}'''


async def check_citation_accuracy(prose: str, facts: list) -> dict:
    """Check if citations in prose accurately reference the facts.

    Returns:
        dict with:
        - accuracy: float (0-1) or None if check failed
        - checked: int (number of citations checked)
        - mismatches: list of inaccurate citations with reasons
    """
    # Extract all citations with surrounding sentence
    citations = []
    # Match sentences containing [N] - handle multiple citations per sentence
    sentences = re.split(r'(?<=[.!?])\s+', prose)

    for sentence in sentences:
        # Find all citation numbers in this sentence
        cite_nums = [int(c) for c in re.findall(r'\[(\d+)\]', sentence) if c.isdigit()]
        for fact_num in cite_nums:
            if 1 <= fact_num <= len(facts):
                citations.append({
                    "sentence": sentence.strip(),
                    "fact_num": fact_num,
                    "fact_text": facts[fact_num - 1].get("extracted_text", "")
                })

    if not citations:
        return {"accuracy": 1.0, "checked": 0, "mismatches": []}

    # Build prompt with citations to check
    citations_text = ""
    for i, c in enumerate(citations, 1):
        citations_text += f"\n{i}. Claim: \"{c['sentence']}\"\n   Fact [{c['fact_num']}]: \"{c['fact_text']}\"\n"

    prompt = CITATION_ACCURACY_PROMPT.format(citations=citations_text)

    try:
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0
        )

        response_text = resp.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in response_text:
            response_text = re.search(r'```(?:json)?\s*(.*?)```', response_text, re.DOTALL)
            response_text = response_text.group(1) if response_text else "{}"

        result = json.loads(response_text)

        accurate_count = sum(1 for r in result.get("results", []) if r.get("accurate", True))
        mismatches = [
            {"citation_num": i + 1, "fact_num": citations[i]["fact_num"], "reason": r.get("reason", "no reason given")}
            for i, r in enumerate(result.get("results", []))
            if not r.get("accurate", True)
        ]
        accuracy = accurate_count / len(citations) if citations else 1.0

        return {
            "accuracy": accuracy,
            "checked": len(citations),
            "mismatches": mismatches
        }
    except Exception as e:
        # If accuracy check fails, return None but don't crash
        return {
            "accuracy": None,
            "checked": len(citations),
            "mismatches": [],
            "error": str(e)
        }


async def resynthesize_section(theme: str, facts: list, topic: str) -> dict:
    """Re-run synthesis for a single section."""
    # Format facts like pipeline does
    facts_text = "\n\n".join([
        f"[{i}] {f['extracted_text']}\n    Source: {f.get('source_url', 'unknown')}"
        for i, f in enumerate(facts, 1)
    ])

    prompt = THEME_SYNTHESIS_PROMPT.format(
        theme=theme,
        topic=topic,
        facts=facts_text,
        source_quality_guidance=""
    )

    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3
    )
    prose = resp.choices[0].message.content.strip()

    # Calculate fact usage metrics
    found_citations = set(int(c) for c in re.findall(r'\[(\d+)\]', prose) if c.isdigit())
    total_facts = len(facts)
    valid_range = set(range(1, total_facts + 1))
    valid_cited = found_citations & valid_range
    invalid_cited = found_citations - valid_range
    fact_usage_rate = len(valid_cited) / total_facts if total_facts > 0 else 0
    uncited = valid_range - found_citations

    # Check citation accuracy (are citations correct?)
    accuracy_result = await check_citation_accuracy(prose, facts)

    return {
        "theme": theme,
        "total_facts": total_facts,
        "fact_usage_rate": fact_usage_rate,  # % of input facts mentioned
        "citation_accuracy": accuracy_result["accuracy"],  # % of citations that are correct
        "citations_checked": accuracy_result["checked"],
        "mismatches": accuracy_result.get("mismatches", []),
        "accuracy_error": accuracy_result.get("error"),
        "cited": sorted(valid_cited),
        "uncited": sorted(uncited),
        "invalid_citations": sorted(invalid_cited),
        "prose": prose
    }


def analyze_source_authority(facts: list) -> dict:
    """Analyze source authority distribution across facts.

    Tiers:
    - Tier 1 (Authoritative): Official docs, .gov, .edu, arxiv, major vendor docs
    - Tier 2 (Established): Major tech sites, news outlets, established blogs
    - Tier 3 (Community): Medium, dev.to, personal blogs, forums

    Returns dict with tier counts and domain breakdown.
    """
    from urllib.parse import urlparse

    TIER_1_DOMAINS = {
        'arxiv.org', 'github.com', 'docs.python.org', 'docs.langchain.com',
        'platform.openai.com', 'anthropic.com', 'cloud.google.com',
        'aws.amazon.com', 'azure.microsoft.com', 'developer.nvidia.com',
        'huggingface.co', 'pytorch.org', 'tensorflow.org'
    }
    TIER_1_PATTERNS = ['.gov', '.edu', '/docs/', '/documentation/', '/api/']

    TIER_2_DOMAINS = {
        'techcrunch.com', 'wired.com', 'arstechnica.com', 'theverge.com',
        'venturebeat.com', 'zdnet.com', 'infoworld.com', 'thenewstack.io',
        'towardsdatascience.com', 'analyticsvidhya.com', 'kdnuggets.com'
    }

    TIER_3_DOMAINS = {
        'medium.com', 'dev.to', 'hashnode.dev', 'substack.com',
        'reddit.com', 'stackoverflow.com', 'quora.com'
    }

    tier_counts = {1: 0, 2: 0, 3: 0, 'unknown': 0}
    domain_counts = {}

    for fact in facts:
        url = fact.get('source_url', '')
        if not url:
            tier_counts['unknown'] += 1
            continue

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
        except:
            tier_counts['unknown'] += 1
            continue

        # Count domains
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Classify tier
        if domain in TIER_1_DOMAINS or any(p in url.lower() for p in TIER_1_PATTERNS):
            tier_counts[1] += 1
        elif domain in TIER_2_DOMAINS:
            tier_counts[2] += 1
        elif domain in TIER_3_DOMAINS or 'blog' in domain or 'medium' in domain:
            tier_counts[3] += 1
        else:
            # Default to tier 2 for unknown domains (assume established)
            tier_counts[2] += 1

    total = sum(tier_counts.values())
    tier_pcts = {
        k: (v / total * 100 if total > 0 else 0)
        for k, v in tier_counts.items()
    }

    # Top domains
    top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:10]

    return {
        'tier_counts': tier_counts,
        'tier_percentages': tier_pcts,
        'total_facts': total,
        'top_domains': top_domains,
        'unique_domains': len(domain_counts)
    }


def load_data(path: str) -> tuple:
    """Load sections from run_state or fixture file."""
    with open(path) as f:
        data = json.load(f)

    # Detect format: run_state has "hybrid_report", fixture has "sections" at top level
    if "hybrid_report" in data:
        # run_state format
        sections = data["hybrid_report"]["sections"]
        # research_brief can be string or dict depending on version
        brief = data.get("research_brief", "")
        if isinstance(brief, dict):
            topic = brief.get("topic", "research topic")
            query = brief.get("query", "unknown")
        else:
            # String brief - extract from first section theme or use placeholder
            topic = "research topic"
            query = str(brief)[:100] if brief else "unknown"
    elif "sections" in data:
        # fixture format
        sections = data["sections"]
        topic = data.get("topic", "research topic")
        query = data.get("query", "unknown")
    else:
        raise ValueError(f"Unknown file format: {path}")

    return sections, topic, query


async def run_resynthesis(path: str, section_idx: int = None, verbose: bool = False):
    """Re-synthesize all or one section from a run state or fixture."""
    sections, topic, query = load_data(path)

    if section_idx is not None:
        if section_idx >= len(sections):
            print(f"Error: section {section_idx} doesn't exist (only {len(sections)} sections)")
            return []
        sections = [sections[section_idx]]

    print(f"Re-synthesizing: {query}")
    print(f"Sections: {len(sections)}")
    print("=" * 60)

    results = []
    for section in sections:
        result = await resynthesize_section(
            theme=section["theme"],
            facts=section["facts"],
            topic=topic
        )
        results.append(result)

        # Status based on accuracy (the metric that matters)
        acc = result["citation_accuracy"]
        if acc is None:
            status = "?"
        elif acc >= 0.9:
            status = "+"
        else:
            status = "-"

        print(f"[{status}] {section['theme'][:50]}...")
        print(f"    Fact usage: {result['fact_usage_rate']:.0%} ({len(result['cited'])}/{result['total_facts']} facts mentioned)")

        if acc is not None:
            print(f"    Citation accuracy: {acc:.0%} ({result['citations_checked']} citations checked)")
        else:
            print(f"    Citation accuracy: ERROR - {result.get('accuracy_error', 'unknown')}")

        if result["mismatches"]:
            print(f"    Mismatches:")
            for m in result["mismatches"][:3]:  # Show max 3
                print(f"      - [{m['fact_num']}]: {m['reason']}")

        if result["invalid_citations"]:
            print(f"    Invalid refs: {result['invalid_citations']}")

        if verbose:
            print(f"    Prose preview: {result['prose'][:150]}...")
        print()

    # Summary
    total_facts = sum(r["total_facts"] for r in results)
    total_cited = sum(len(r["cited"]) for r in results)
    total_checked = sum(r["citations_checked"] for r in results)
    total_mismatches = sum(len(r["mismatches"]) for r in results)

    # Calculate overall accuracy (only from sections that succeeded)
    accuracy_results = [r["citation_accuracy"] for r in results if r["citation_accuracy"] is not None]
    overall_accuracy = sum(accuracy_results) / len(accuracy_results) if accuracy_results else None

    print("=" * 60)
    print(f"Fact usage: {total_cited}/{total_facts} facts mentioned")

    if overall_accuracy is not None:
        print(f"Citation accuracy: {overall_accuracy:.1%} ({total_checked - total_mismatches}/{total_checked} correct)")
        passed = overall_accuracy >= 0.9
        print(f"Status: {'PASS' if passed else 'FAIL'} (target: 90% accuracy)")
    else:
        print("Citation accuracy: FAILED TO CHECK")
        passed = False

    return results


def save_fixture(run_state_path: str, output_dir: str = None):
    """Extract synthesis fixture from run_state."""
    with open(run_state_path) as f:
        data = json.load(f)

    if "hybrid_report" not in data:
        print(f"Error: {run_state_path} is not a run_state file")
        return

    # Extract relevant data
    hybrid = data["hybrid_report"]
    brief = data.get("research_brief", "")

    # Handle string vs dict brief
    if isinstance(brief, dict):
        query = brief.get("query", "unknown")
        topic = brief.get("topic", "research topic")
    else:
        query = str(brief)[:100] if brief else "unknown"
        topic = "research topic"

    fixture = {
        "query": query,
        "topic": topic,
        "created_from": Path(run_state_path).name,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "sections": []
    }

    for section in hybrid["sections"]:
        # Calculate original citation rate
        facts = section["facts"]
        prose = section.get("prose", "")
        found = set(int(c) for c in re.findall(r'\[(\d+)\]', prose) if c.isdigit())
        valid_cited = found & set(range(1, len(facts) + 1))
        original_rate = len(valid_cited) / len(facts) if facts else 0

        fixture["sections"].append({
            "theme": section["theme"],
            "facts": [
                {
                    "extracted_text": f.get("extracted_text", ""),
                    "source_url": f.get("source_url", ""),
                }
                for f in facts
            ],
            "original_prose": prose,
            "original_citation_rate": round(original_rate, 3)
        })

    # Determine output path
    if output_dir is None:
        output_dir = project_root / "tests" / "fixtures" / "synthesis"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from query
    safe_name = re.sub(r'[^\w\s-]', '', fixture["query"].lower())
    safe_name = re.sub(r'[\s]+', '_', safe_name)[:50]
    output_path = output_dir / f"{safe_name}.json"

    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)

    print(f"Fixture saved: {output_path}")
    print(f"  Query: {fixture['query']}")
    print(f"  Sections: {len(fixture['sections'])}")
    print(f"  Total facts: {sum(len(s['facts']) for s in fixture['sections'])}")

    return output_path


def run_authority_analysis(path: str):
    """Run source authority analysis on a run state or fixture."""
    sections, topic, query = load_data(path)

    # Collect all facts from all sections
    all_facts = []
    for section in sections:
        all_facts.extend(section["facts"])

    print(f"Analyzing source authority: {query}")
    print(f"Total facts: {len(all_facts)}")
    print("=" * 60)

    result = analyze_source_authority(all_facts)

    # Tier breakdown
    tier_names = {1: "Authoritative", 2: "Established", 3: "Community", 'unknown': "Unknown"}
    print("\nTier Distribution:")
    for tier, count in sorted(result['tier_counts'].items(), key=lambda x: (isinstance(x[0], str), x[0])):
        pct = result['tier_percentages'][tier]
        bar = "█" * int(pct / 2)
        print(f"  Tier {tier} ({tier_names[tier]}): {count:3d} ({pct:5.1f}%) {bar}")

    # Top domains
    print(f"\nTop Domains ({result['unique_domains']} unique):")
    for domain, count in result['top_domains'][:10]:
        print(f"  {domain}: {count}")

    # Assessment
    print("\n" + "=" * 60)
    tier1_pct = result['tier_percentages'].get(1, 0)
    tier3_pct = result['tier_percentages'].get(3, 0)

    if tier1_pct >= 30:
        status = "GOOD"
        msg = f"Strong authoritative sources ({tier1_pct:.0f}% tier 1)"
    elif tier3_pct > 30:
        status = "WARN"
        msg = f"High community source ratio ({tier3_pct:.0f}% tier 3)"
    else:
        status = "OK"
        msg = f"Balanced source mix (tier 1: {tier1_pct:.0f}%, tier 3: {tier3_pct:.0f}%)"

    print(f"Assessment: {status} - {msg}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Re-run synthesis on saved run states")
    parser.add_argument("path", nargs="?", help="Path to run_state or fixture JSON")
    parser.add_argument("--fixture", "-f", help="Path to fixture file (alias for path)")
    parser.add_argument("--section", "-s", type=int, help="Re-synth only this section (0-indexed)")
    parser.add_argument("--save-fixture", action="store_true", help="Save as fixture instead of re-synth")
    parser.add_argument("--analyze", "-a", action="store_true", help="Analyze source authority distribution")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show prose previews")
    parser.add_argument("--output-dir", "-o", help="Output directory for fixtures")

    args = parser.parse_args()

    # --fixture overrides positional path
    if args.fixture:
        args.path = args.fixture

    if not args.path:
        parser.print_help()
        return

    if args.save_fixture:
        save_fixture(args.path, args.output_dir)
    elif args.analyze:
        run_authority_analysis(args.path)
    else:
        asyncio.run(run_resynthesis(args.path, args.section, args.verbose))


if __name__ == "__main__":
    main()
