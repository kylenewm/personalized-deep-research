#!/usr/bin/env python3
"""Extract component-specific fixtures from run_state files.

Takes a run_state_*.json and extracts fixtures for each component's sandbox testing.

Usage:
    # Extract all fixture types
    python scripts/extract_fixtures.py run_state_123.json --all

    # Extract specific type
    python scripts/extract_fixtures.py run_state_123.json --type synthesis
    python scripts/extract_fixtures.py run_state_123.json --type extraction
    python scripts/extract_fixtures.py run_state_123.json --type sources

    # Force extraction even if at limit
    python scripts/extract_fixtures.py run_state_123.json --all --force
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Fixture directory limits (Part 6 of plan)
FIXTURE_LIMITS = {
    'gold_queries': 10,
    'extraction': 20,
    'dedup': 15,
    'synthesis': 20,
    'arrangement': 15,
    'sources': 10,
}

FIXTURES_ROOT = Path(__file__).parent.parent / "tests" / "fixtures"


def get_fixture_count(fixture_type: str) -> int:
    """Count existing fixtures in a directory."""
    fixture_dir = FIXTURES_ROOT / fixture_type
    if not fixture_dir.exists():
        return 0
    return len(list(fixture_dir.glob("*.json")))


def check_limit(fixture_type: str, force: bool = False) -> bool:
    """Check if we're at the fixture limit. Returns True if OK to add."""
    count = get_fixture_count(fixture_type)
    limit = FIXTURE_LIMITS.get(fixture_type, 20)

    if count >= limit:
        if force:
            print(f"  WARN: {fixture_type}/ at limit ({count}/{limit}), forcing...")
            return True
        else:
            print(f"  SKIP: {fixture_type}/ at limit ({count}/{limit}). Use --force to override.")
            return False
    return True


def generate_filename(query: str, source_file: str, suffix: str = "") -> str:
    """Generate a safe filename from query."""
    # Slugify query
    safe_name = re.sub(r'[^\w\s-]', '', query.lower())
    safe_name = re.sub(r'[\s]+', '_', safe_name)[:40]

    # Add date
    date = datetime.now().strftime("%Y-%m-%d")

    if suffix:
        return f"{safe_name}_{date}_{suffix}.json"
    return f"{safe_name}_{date}.json"


def create_meta(source_file: str, purpose: str) -> dict:
    """Create standard metadata for a fixture."""
    return {
        "_meta": {
            "tier": "extended",  # New fixtures start as extended
            "created": datetime.now().strftime("%Y-%m-%d"),
            "last_used": None,
            "purpose": purpose,
            "passing": None,
            "extracted_from": Path(source_file).name
        }
    }


def extract_synthesis(run_state: dict, source_file: str) -> dict:
    """Extract synthesis fixture from run_state.

    Source: hybrid_report.sections[] → theme + facts + prose
    """
    if "hybrid_report" not in run_state:
        return None

    hybrid = run_state["hybrid_report"]
    brief = run_state.get("research_brief", {})

    # Handle string vs dict brief
    if isinstance(brief, dict):
        query = brief.get("query", "unknown")
        topic = brief.get("topic", "research topic")
    else:
        query = str(brief)[:100] if brief else "unknown"
        topic = "research topic"

    fixture = {
        **create_meta(source_file, "synthesis testing"),
        "query": query,
        "topic": topic,
        "sections": []
    }

    for section in hybrid.get("sections", []):
        facts = section.get("facts", [])
        prose = section.get("prose", "")

        # Calculate original metrics
        found = set(int(c) for c in re.findall(r'\[(\d+)\]', prose) if c.isdigit())
        valid_cited = found & set(range(1, len(facts) + 1))
        original_rate = len(valid_cited) / len(facts) if facts else 0

        fixture["sections"].append({
            "theme": section.get("theme", ""),
            "facts": [
                {
                    "extracted_text": f.get("extracted_text", ""),
                    "source_url": f.get("source_url", ""),
                }
                for f in facts
            ],
            "original_prose": prose,
            "original_metrics": {
                "citation_rate": round(original_rate, 3),
                "facts_count": len(facts)
            }
        })

    return fixture


def extract_extraction(run_state: dict, source_file: str) -> dict:
    """Extract extraction fixture from run_state.

    Source: source_store[] → raw content + URL (trimmed to 10 sources)
    """
    source_store = run_state.get("source_store", [])
    if not source_store:
        return None

    brief = run_state.get("research_brief", {})
    query = brief.get("query", "unknown") if isinstance(brief, dict) else str(brief)[:100]

    # Take first 10 sources to keep fixture manageable
    sources = source_store[:10]

    fixture = {
        **create_meta(source_file, "extraction testing"),
        "query": query,
        "sources": [
            {
                "url": s.get("url", ""),
                "title": s.get("title", ""),
                "raw_content": (s.get("raw_content") or s.get("content", ""))[:5000],  # Truncate long content
            }
            for s in sources
        ]
    }

    return fixture


def extract_sources(run_state: dict, source_file: str) -> dict:
    """Extract source authority fixture from run_state.

    Source: All fact URLs from hybrid_report
    """
    if "hybrid_report" not in run_state:
        return None

    brief = run_state.get("research_brief", {})
    query = brief.get("query", "unknown") if isinstance(brief, dict) else str(brief)[:100]

    # Collect all URLs from facts
    urls = []
    for section in run_state["hybrid_report"].get("sections", []):
        for fact in section.get("facts", []):
            url = fact.get("source_url", "")
            if url:
                urls.append(url)

    # Dedupe and count
    url_counts = {}
    for url in urls:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
        except:
            domain = "unknown"
        url_counts[domain] = url_counts.get(domain, 0) + 1

    fixture = {
        **create_meta(source_file, "source authority testing"),
        "query": query,
        "total_facts": len(urls),
        "unique_domains": len(url_counts),
        "domain_distribution": dict(sorted(url_counts.items(), key=lambda x: -x[1]))
    }

    return fixture


def extract_dedup(run_state: dict, source_file: str) -> dict:
    """Extract dedup fixture from run_state.

    Source: checkpoints.pre_dedup and checkpoints.post_dedup from hybrid_report
    """
    # Check for checkpoints in hybrid_report
    hybrid_report = run_state.get("hybrid_report", {})
    checkpoints = hybrid_report.get("checkpoints", {})

    if checkpoints.get("pre_dedup") and checkpoints.get("post_dedup"):
        brief = run_state.get("research_brief", {})
        query = brief.get("query", "unknown") if isinstance(brief, dict) else str(brief)[:100]

        pre_dedup = checkpoints["pre_dedup"]
        post_dedup = checkpoints["post_dedup"]

        # Build labeled pairs from what was removed
        pre_texts = {f["extracted_text"] for f in pre_dedup}
        post_texts = {f["extracted_text"] for f in post_dedup}
        removed_texts = pre_texts - post_texts

        fixture = {
            **create_meta(source_file, "dedup testing"),
            "query": query,
            "pre_dedup": pre_dedup,
            "post_dedup": post_dedup,
            "pre_dedup_count": checkpoints.get("pre_dedup_count", len(pre_dedup)),
            "post_dedup_count": checkpoints.get("post_dedup_count", len(post_dedup)),
            "dedup_removed": checkpoints.get("dedup_removed", len(removed_texts)),
            "labeled_pairs": []  # Would need manual labeling
        }
        return fixture

    print("  WARN: No checkpoints in hybrid_report - run pipeline with latest code")
    return None


def extract_arrangement(run_state: dict, source_file: str) -> dict:
    """Extract arrangement fixture from run_state.

    Source: checkpoints.pre_arrangement and checkpoints.post_arrangement from hybrid_report
    """
    hybrid_report = run_state.get("hybrid_report", {})
    checkpoints = hybrid_report.get("checkpoints", {})

    if checkpoints.get("pre_arrangement") and checkpoints.get("post_arrangement"):
        brief = run_state.get("research_brief", {})
        query = brief.get("query", "unknown") if isinstance(brief, dict) else str(brief)[:100]

        post_arr = checkpoints["post_arrangement"]

        fixture = {
            **create_meta(source_file, "arrangement testing"),
            "query": query,
            "facts_before_arrangement": checkpoints["pre_arrangement"],
            "pre_arrangement_count": checkpoints.get("pre_arrangement_count", len(checkpoints["pre_arrangement"])),
            "themes": post_arr.get("themes", []),
            "excluded_ids": post_arr.get("excluded_ids", []),
            "excluded_count": post_arr.get("excluded_count", 0),
            "grouped_count": post_arr.get("grouped_count", 0),
        }
        return fixture

    print("  WARN: No checkpoints in hybrid_report - run pipeline with latest code")
    return None


EXTRACTORS = {
    'synthesis': extract_synthesis,
    'extraction': extract_extraction,
    'sources': extract_sources,
    'dedup': extract_dedup,
    'arrangement': extract_arrangement,
}


def extract_fixture(run_state_path: str, fixture_type: str, force: bool = False) -> Path:
    """Extract a specific fixture type from run_state."""
    with open(run_state_path) as f:
        run_state = json.load(f)

    if fixture_type not in EXTRACTORS:
        print(f"  ERROR: Unknown fixture type: {fixture_type}")
        return None

    # Check limit
    if not check_limit(fixture_type, force):
        return None

    # Extract fixture
    extractor = EXTRACTORS[fixture_type]
    fixture = extractor(run_state, run_state_path)

    if fixture is None:
        return None

    # Determine output path
    output_dir = FIXTURES_ROOT / fixture_type
    output_dir.mkdir(parents=True, exist_ok=True)

    query = fixture.get("query", "unknown")
    filename = generate_filename(query, run_state_path)
    output_path = output_dir / filename

    # Avoid overwriting
    counter = 1
    while output_path.exists():
        filename = generate_filename(query, run_state_path, f"v{counter}")
        output_path = output_dir / filename
        counter += 1

    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract fixtures from run_state files")
    parser.add_argument("path", help="Path to run_state JSON file")
    parser.add_argument("--all", action="store_true", help="Extract all fixture types")
    parser.add_argument("--type", "-t", choices=list(EXTRACTORS.keys()), help="Extract specific fixture type")
    parser.add_argument("--force", "-f", action="store_true", help="Force extraction even at limit")

    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"Error: {args.path} not found")
        return

    print(f"Extracting fixtures from: {Path(args.path).name}")
    print("=" * 60)

    if args.all:
        types_to_extract = list(EXTRACTORS.keys())
    elif args.type:
        types_to_extract = [args.type]
    else:
        parser.print_help()
        return

    results = {}
    for fixture_type in types_to_extract:
        print(f"\n[{fixture_type}]")
        output_path = extract_fixture(args.path, fixture_type, args.force)
        if output_path:
            print(f"  ✓ Saved: {output_path.relative_to(FIXTURES_ROOT.parent.parent)}")
            results[fixture_type] = output_path
        else:
            print(f"  ✗ Skipped")

    print("\n" + "=" * 60)
    print(f"Extracted: {len(results)}/{len(types_to_extract)} fixture types")

    # Show current fixture counts
    print("\nFixture counts:")
    for ft in FIXTURE_LIMITS:
        count = get_fixture_count(ft)
        limit = FIXTURE_LIMITS[ft]
        bar = "█" * (count * 10 // limit) if limit > 0 else ""
        status = "FULL" if count >= limit else ""
        print(f"  {ft:15} {count:3}/{limit:3} {bar} {status}")


if __name__ == "__main__":
    main()
