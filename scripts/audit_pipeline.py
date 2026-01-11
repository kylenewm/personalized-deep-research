#!/usr/bin/env python3
"""Audit pipeline - run extraction/arrangement/synthesis with full visibility.

Shows exactly what facts are extracted, how they're grouped, what's excluded.
Uses saved fixtures to iterate without API costs on research phase.

Usage:
    python scripts/audit_pipeline.py                    # Use latest fixture
    python scripts/audit_pipeline.py realtime_api      # Use specific fixture
    python scripts/audit_pipeline.py --verbose         # Show all facts
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from openai import AsyncOpenAI
from open_deep_research.pipeline_v2 import (
    extract_all_batched,
    deduplicate_extractions,
    arrange_facts,
    synthesize_theme,
    assemble_report,
    HybridReport,
    ThemedSection,
)
from open_deep_research.render import render_html, report_to_dict


async def run_audit(fixture_name: str = None, verbose: bool = False):
    """Run pipeline with full audit output."""

    # Load fixture
    fixture_dir = project_root / "tests/fixtures/gold_queries"
    if fixture_name:
        fixture_path = fixture_dir / f"{fixture_name}.json"
    else:
        # Use latest
        fixtures = sorted(fixture_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not fixtures:
            print("No fixtures found. Run a test first to create one.")
            return
        fixture_path = fixtures[0]

    print(f"Loading: {fixture_path.name}")
    with open(fixture_path) as f:
        fixture = json.load(f)

    query = fixture.get("query", "Research topic")
    sources_list = fixture.get("sources", [])

    # Limit sources for testing
    quick_mode = "--quick" in sys.argv or "-q" in sys.argv
    max_sources = 3 if quick_mode else 15
    sources_list = sources_list[:max_sources]
    if quick_mode:
        print("⚡ QUICK MODE: 3 sources only")

    # Convert to dict format
    sources = {}
    for i, src in enumerate(sources_list):
        content = src.get("content", "") or src.get("raw_content", "")
        if content:
            sources[f"src_{i:03d}"] = {
                "content": content,
                "url": src.get("url", ""),
                "title": src.get("title", "Unknown"),
            }

    print(f"Query: {query[:80]}...")
    print(f"Sources: {len(sources)} (limited from {len(fixture.get('sources', []))})")
    print("=" * 80)

    # Setup LLMs - 4.1 for extraction (needs good instruction following), 4.1-mini for rest
    client = AsyncOpenAI()

    async def llm_call_strong(prompt: str) -> str:
        """GPT-4.1 for extraction - better at following complex instructions."""
        resp = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3
        )
        return resp.choices[0].message.content

    async def llm_call_fast(prompt: str) -> str:
        """GPT-4.1-mini for arrangement/synthesis - good enough, 10x cheaper."""
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3
        )
        return resp.choices[0].message.content

    # =========================================================================
    # STAGE 1: EXTRACTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: EXTRACTION")
    print("=" * 80)

    extractions = await extract_all_batched(
        sources=sources,
        topic=query,
        llm_call=llm_call_fast,  # 4.1-mini for extraction (big input, needs to be cheap)
        batch_size=12,
        min_score=0.4,
        on_batch_complete=lambda i, t, e: print(f"  Batch {i}/{t}: {len([x for x in e if x.status == 'verified'])} verified")
    )

    verified = [e for e in extractions if e.status == "verified"]
    print(f"\nExtracted: {len(verified)} verified facts from {len(extractions)} attempts")

    # =========================================================================
    # STAGE 1.5: QUALITY FILTER (4.1 on small set)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1.5: QUALITY FILTER (GPT-4.1 on extracted facts)")
    print("=" * 80)

    # Build batch of facts to check
    facts_to_check = [ext.extracted_text for ext in verified if ext.extracted_text]

    quality_prompt = f"""Review these extracted facts. For each one, respond KEEP or REJECT.

REJECT if:
- Question (ends with ? or asks something)
- Header/title ("Key Features:", "What to Expect", etc.)
- Too long (more than ~50 words) - we want concise claims with context
- Has formatting artifacts (bullet points, "####", "Key strengths:", numbered lists)
- Promotional fluff without specific data
- Incomplete fragment
- Repeats same info as another fact (mark duplicates as REJECT)

KEEP if:
- Declarative statement (under 50 words)
- Contains specific info (numbers, names, dates, metrics)
- Clean formatting, reads as a standalone sentence

Facts to review:
{chr(10).join(f'[{i}] {text[:300]}' for i, text in enumerate(facts_to_check))}

Respond with one line per fact: "0: KEEP" or "0: REJECT"
For duplicates, keep the best-worded one and reject others."""

    filter_response = await llm_call_strong(quality_prompt)

    # Parse response
    keep_indices = set()
    for line in filter_response.strip().split('\n'):
        if ':' in line:
            try:
                idx = int(line.split(':')[0].strip())
                if 'KEEP' in line.upper():
                    keep_indices.add(idx)
            except ValueError:
                continue

    # Filter verified list
    quality_filtered = [ext for i, ext in enumerate(verified) if i in keep_indices]
    rejected_count = len(verified) - len(quality_filtered)
    print(f"Quality filtered: {len(verified)} → {len(quality_filtered)} ({rejected_count} rejected by GPT-4.1)")

    verified = quality_filtered  # Use filtered list going forward

    if verbose:
        print("\n--- QUALITY-FILTERED FACTS ---")
        for i, ext in enumerate(verified):
            print(f"\n[{i}] {ext.extracted_text[:200]}...")
            print(f"    Source: {ext.source_url}")
            print(f"    Context: {ext.pointer.context}")
            print(f"    Score: {ext.match_score:.0%}")

    # =========================================================================
    # STAGE 2: DEDUPLICATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: DEDUPLICATION")
    print("=" * 80)

    deduped = deduplicate_extractions(verified)
    removed = len(verified) - len(deduped)
    print(f"Deduplicated: {len(verified)} → {len(deduped)} ({removed} duplicates removed)")

    # =========================================================================
    # STAGE 3: ARRANGEMENT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 3: ARRANGEMENT (Grouping by theme)")
    print("=" * 80)

    arranged = await arrange_facts(deduped, query, llm_call_fast)  # 4.1-mini is fine for arrangement

    print(f"\nThemes created: {len(arranged.groups)}")
    print(f"Facts included: {sum(len(g.fact_ids) for g in arranged.groups)}")
    print(f"Facts excluded: {len(arranged.excluded_ids)}")

    print("\n--- THEMES ---")
    for i, group in enumerate(arranged.groups):
        print(f"\n[Theme {i+1}] {group.theme}")
        print(f"  Facts: {len(group.fact_ids)}")
        if verbose:
            for fid in group.fact_ids:
                fact = deduped[fid]
                print(f"    - {fact.extracted_text[:100]}...")

    if arranged.excluded_ids:
        print(f"\n--- EXCLUDED FACTS ({len(arranged.excluded_ids)}) ---")
        if verbose:
            for fid in arranged.excluded_ids[:10]:  # Show first 10
                fact = deduped[fid]
                print(f"  [{fid}] {fact.extracted_text[:100]}...")
            if len(arranged.excluded_ids) > 10:
                print(f"  ... and {len(arranged.excluded_ids) - 10} more")

    # =========================================================================
    # STAGE 4: SYNTHESIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 4: SYNTHESIS (Writing prose with citations)")
    print("=" * 80, flush=True)
    sections = []
    for group in arranged.groups:
        print(f"  Synthesizing '{group.theme}' ({len(group.fact_ids)} facts)...", flush=True)
        section = await synthesize_theme(group.theme, group.fact_ids, deduped, query, llm_call_fast)
        sections.append(section)
        print(f"    -> {len(section.prose)} chars, {len(section.citations)} citations")

    for section in sections:
        print(f"\n[{section.theme}]")
        print(f"  Prose: {section.prose[:150]}..." if section.prose else "  (no prose)")
        print(f"  Facts: {len(section.facts)}")
        print(f"  Citations: {len(section.citations)}")

    # =========================================================================
    # STAGE 5: ASSEMBLY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 5: ASSEMBLY (Executive summary, analysis, conclusion)")
    print("=" * 80)
    report = await assemble_report(sections, query, "Research Report", llm_call_fast, len(sources), len(verified))
    print(f"Executive summary: {len(report.executive_summary)} chars")
    print(f"Analysis: {len(report.analysis)} chars")
    print(f"Conclusion: {len(report.conclusion)} chars")

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    # Save audit log
    audit_log = {
        "query": query,
        "sources_count": len(sources),
        "extraction": {
            "total_attempts": len(extractions),
            "verified": len(verified),
            "facts": [
                {
                    "id": i,
                    "text": e.extracted_text,
                    "source_url": e.source_url,
                    "context": e.pointer.context,
                    "score": e.match_score,
                }
                for i, e in enumerate(verified)
            ]
        },
        "deduplication": {
            "before": len(verified),
            "after": len(deduped),
            "removed": removed,
        },
        "arrangement": {
            "themes": [
                {
                    "theme": g.theme,
                    "fact_ids": g.fact_ids,
                    "fact_count": len(g.fact_ids),
                }
                for g in arranged.groups
            ],
            "excluded_ids": arranged.excluded_ids,
        },
        "synthesis": {
            "sections": [
                {
                    "theme": s.theme,
                    "intro": s.intro,
                    "fact_count": len(s.facts),
                    "transition_count": len(s.transitions),
                }
                for s in sections
            ]
        },
        "report_stats": {
            "total_extracted": report.total_extracted,
            "total_verified": report.total_verified,
            "total_used": report.total_used,
            "themes": len(report.sections),
        }
    }

    audit_path = project_root / "audit_output.json"
    with open(audit_path, "w") as f:
        json.dump(audit_log, f, indent=2)
    print(f"Audit log: {audit_path}")

    # Save HTML report
    data = report_to_dict(report)
    html = render_html(data)
    html_path = project_root / "audit_report.html"
    html_path.write_text(html)
    print(f"HTML report: {html_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


def main():
    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args

    # Filter out flags to get fixture name
    args = [a for a in args if not a.startswith("-")]
    fixture_name = args[0] if args else None

    asyncio.run(run_audit(fixture_name, verbose))


if __name__ == "__main__":
    main()
