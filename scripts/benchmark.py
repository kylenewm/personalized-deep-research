#!/usr/bin/env python3
"""Benchmark runner for report quality evaluation.

Evaluates quality metrics from saved gold dataset JSON files.
No pipeline imports required - works standalone.

Usage:
    python scripts/benchmark.py tests/fixtures/gold_queries/agentic_coding_2026.json
    python scripts/benchmark.py --all
    python scripts/benchmark.py --json  # Output as JSON
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


# =============================================================================
# Quality Metrics Functions
# =============================================================================

def count_specific(facts: list[str]) -> int:
    """Count facts with specific numbers/metrics."""
    pattern = r'\d+[%kmKMGB]?|\d+\.\d+|99\.\d+%|200[,\s]?000'
    return sum(1 for f in facts if re.search(pattern, f))


def count_vague(facts: list[str]) -> int:
    """Count facts with vague language."""
    patterns = ['can be', 'may be', 'could be', 'enables', 'allows', 'provides', 'supports']
    return sum(1 for f in facts if any(p in f.lower() for p in patterns))


def count_fluff(facts: list[str]) -> int:
    """Count facts with marketing fluff."""
    patterns = ['revolutionary', 'cutting-edge', 'best-in-class', 'world-class',
                'leading', 'powerful', 'robust', 'seamless', 'game-changing']
    return sum(1 for f in facts if any(p in f.lower() for p in patterns))


def avg_words(facts: list[str]) -> float:
    """Average word count per fact."""
    if not facts:
        return 0
    return sum(len(f.split()) for f in facts) / len(facts)


def count_over_40(facts: list[str]) -> int:
    """Count facts over 40 words."""
    return sum(1 for f in facts if len(f.split()) > 40)


def count_unique_domains(footnotes: list[dict]) -> int:
    """Count unique source domains."""
    domains = set()
    for f in footnotes:
        domain = f.get('source_domain', '')
        if domain:
            domains.add(domain)
    return len(domains)


def count_authoritative(footnotes: list[dict]) -> int:
    """Count facts from authoritative sources."""
    authoritative_patterns = [
        'anthropic.com', 'openai.com', 'google.com', 'microsoft.com',
        'github.com', 'arxiv.org', 'acm.org', 'ieee.org',
        '.edu', '.gov', 'docs.', 'official'
    ]
    count = 0
    for f in footnotes:
        domain = f.get('source_domain', '').lower()
        url = f.get('source_url', '').lower()
        if any(p in domain or p in url for p in authoritative_patterns):
            count += 1
    return count


def count_cited_sentences(prose: str) -> int:
    """Count sentences with citations."""
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', prose)
    sentences = [s.strip() for s in sentences if s.strip()]
    cited = sum(1 for s in sentences if re.search(r'\[\d+\]', s))
    return cited


def count_sentences(prose: str) -> int:
    """Count total sentences."""
    sentences = re.split(r'[.!?]+', prose)
    return len([s for s in sentences if s.strip()])


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_from_json(data: dict) -> dict:
    """Evaluate quality from saved JSON (hybrid_report format)."""

    report = data.get('hybrid_report', {})
    footnotes = report.get('footnotes', [])

    # Extract fact texts
    facts = [f.get('extracted_text', '') for f in footnotes]
    facts = [f for f in facts if f]  # Filter empty

    if not facts:
        return {"error": "No facts found in report"}

    # Extract prose from sections
    sections = report.get('sections', [])
    all_prose = ' '.join(s.get('prose', '') for s in sections)

    # Calculate metrics
    metrics = {
        "extraction": {
            "total_facts": len(facts),
            "specificity_rate": round(count_specific(facts) / len(facts) * 100, 1),
            "vague_rate": round(count_vague(facts) / len(facts) * 100, 1),
            "fluff_rate": round(count_fluff(facts) / len(facts) * 100, 1),
            "avg_word_count": round(avg_words(facts), 1),
            "over_40_words_pct": round(count_over_40(facts) / len(facts) * 100, 1)
        },
        "source": {
            "domain_diversity": count_unique_domains(footnotes),
            "authoritative_count": count_authoritative(footnotes),
            "authoritative_rate": round(count_authoritative(footnotes) / len(facts) * 100, 1) if facts else 0
        },
        "synthesis": {
            "theme_count": len(sections),
            "total_sentences": count_sentences(all_prose),
            "cited_sentences": count_cited_sentences(all_prose),
            "citation_rate": round(count_cited_sentences(all_prose) / max(count_sentences(all_prose), 1) * 100, 1)
        }
    }

    return metrics


def evaluate_sources_only(data: dict) -> dict:
    """Evaluate source quality when no hybrid_report exists."""

    sources = data.get('sources', [])

    # Domain distribution
    domains = []
    for s in sources:
        url = s.get('url', '')
        if url.startswith('http'):
            parts = url.split('/')
            if len(parts) > 2:
                domains.append(parts[2])

    domain_counts = Counter(domains)

    # Authoritative source detection
    authoritative_patterns = [
        'anthropic.com', 'openai.com', 'google.com', 'microsoft.com',
        'github.com', 'arxiv.org', '.edu', '.gov', 'docs.'
    ]
    auth_count = sum(1 for d in domains if any(p in d.lower() for p in authoritative_patterns))

    return {
        "sources": {
            "total": len(sources),
            "unique_domains": len(domain_counts),
            "authoritative_count": auth_count,
            "authoritative_rate": round(auth_count / len(sources) * 100, 1) if sources else 0,
            "top_domains": dict(domain_counts.most_common(10))
        }
    }


# =============================================================================
# Main
# =============================================================================

def run_evaluation(path: str) -> dict:
    """Run evaluation on a gold dataset."""

    data = json.loads(Path(path).read_text())

    result = {
        "dataset": path,
        "query": data.get('query', '')[:100] + '...',
        "timestamp": datetime.now().isoformat(),
        "source_count": len(data.get('sources', []))
    }

    # Try full evaluation if hybrid_report exists
    if 'hybrid_report' in data:
        result["metrics"] = evaluate_from_json(data)
    else:
        result["metrics"] = evaluate_sources_only(data)

    return result


def print_results(result: dict):
    """Pretty print evaluation results."""

    print(f"\n{'='*60}")
    print(f"Dataset: {Path(result['dataset']).name}")
    print(f"Query: {result['query']}")
    print(f"Sources: {result['source_count']}")
    print(f"{'='*60}\n")

    metrics = result.get('metrics', {})

    if 'extraction' in metrics:
        ext = metrics['extraction']
        print("EXTRACTION QUALITY:")
        print(f"  Total facts: {ext['total_facts']}")
        print(f"  Specificity rate: {ext['specificity_rate']}% (target: >30%)")
        print(f"  Vague rate: {ext['vague_rate']}% (target: <20%)")
        print(f"  Fluff rate: {ext['fluff_rate']}% (target: 0%)")
        print(f"  Avg words: {ext['avg_word_count']} (target: 20-30)")
        print(f"  Over 40 words: {ext['over_40_words_pct']}% (target: <15%)")

    if 'source' in metrics:
        src = metrics['source']
        print("\nSOURCE QUALITY:")
        print(f"  Domain diversity: {src['domain_diversity']} unique domains")
        print(f"  Authoritative: {src.get('authoritative_count', 'N/A')} ({src['authoritative_rate']}%)")

    if 'synthesis' in metrics:
        syn = metrics['synthesis']
        print("\nSYNTHESIS QUALITY:")
        print(f"  Themes: {syn['theme_count']}")
        print(f"  Citation rate: {syn['citation_rate']}% (target: >85%)")

    if 'sources' in metrics:
        src = metrics['sources']
        print("\nSOURCE ANALYSIS:")
        print(f"  Total sources: {src['total']}")
        print(f"  Unique domains: {src['unique_domains']}")
        print(f"  Authoritative: {src['authoritative_count']} ({src['authoritative_rate']}%)")
        print(f"  Top domains: {list(src['top_domains'].items())[:5]}")

    print()


def main():
    parser = argparse.ArgumentParser(description='Benchmark report quality')
    parser.add_argument('dataset', nargs='?', help='Path to gold dataset JSON')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    gold_dir = Path(__file__).parent.parent / 'tests' / 'fixtures' / 'gold_queries'

    if args.all:
        datasets = list(gold_dir.glob('*.json'))
    elif args.dataset:
        datasets = [Path(args.dataset)]
    else:
        print("Usage: benchmark.py <dataset.json> or benchmark.py --all")
        sys.exit(1)

    all_results = []
    for dataset in datasets:
        if not dataset.exists():
            print(f"Warning: {dataset} not found, skipping")
            continue

        result = run_evaluation(str(dataset))
        all_results.append(result)

        if not args.json:
            print_results(result)

    if args.json:
        print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()
