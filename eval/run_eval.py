#!/usr/bin/env python3
"""Run evaluation on gold datasets - standalone, no pipeline imports.

Usage:
    python eval/run_eval.py tests/fixtures/gold_queries/agentic_coding_2026.json
    python eval/run_eval.py --mode mini
    python eval/run_eval.py --mode full
    python eval/run_eval.py --all --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from llm import call_llm, estimate_cost
from metrics import EvalResult


# =============================================================================
# Prompt Loading
# =============================================================================

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load prompt template from file."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text()


# =============================================================================
# Data Extraction (from JSON only)
# =============================================================================

def extract_facts_from_json(data: dict) -> list[dict]:
    """Extract facts from gold dataset JSON."""
    facts = []

    # Try hybrid_report.footnotes first (has full data)
    report = data.get("hybrid_report", {})
    footnotes = report.get("footnotes", [])

    if footnotes:
        for i, f in enumerate(footnotes):
            facts.append({
                "id": i + 1,
                "text": f.get("extracted_text", ""),
                "source_url": f.get("source_url", ""),
                "source_domain": f.get("source_domain", ""),
                "match_score": f.get("match_score", None),
                "theme": f.get("theme", "")
            })
    return facts


def extract_themes_from_json(data: dict) -> list[str]:
    """Extract themes from research brief or report sections."""
    # Try report sections first
    report = data.get("hybrid_report", {})
    sections = report.get("sections", [])

    if sections:
        return [s.get("theme", "") for s in sections if s.get("theme")]

    # Fallback: extract from brief (would need LLM, skip for now)
    return []


def extract_prose_from_json(data: dict) -> list[dict]:
    """Extract prose sections with citations."""
    report = data.get("hybrid_report", {})
    sections = report.get("sections", [])

    prose_sections = []
    for s in sections:
        prose = s.get("prose", "")
        if prose:
            prose_sections.append({
                "theme": s.get("theme", ""),
                "prose": prose
            })
    return prose_sections


# =============================================================================
# Brief Evaluation (Query → Brief transformation quality)
# =============================================================================

def run_brief_eval(
    query: str,
    brief: str,
    dry_run: bool = False
) -> dict:
    """Run brief evaluation (checks for dilution, preservation, assumptions)."""

    if not brief:
        return {"skip": True, "reason": "No research brief in dataset"}

    # Load and fill prompt
    prompt_template = load_prompt("brief_eval")
    safe_query = query.replace("{", "{{").replace("}", "}}")
    safe_brief = brief.replace("{", "{{").replace("}", "}}")
    prompt = prompt_template.format(
        query=safe_query,
        brief=safe_brief
    )

    if dry_run:
        cost = estimate_cost(prompt)
        return {
            "dry_run": True,
            "prompt_chars": len(prompt),
            "estimated_cost": cost
        }

    # Call LLM with error handling
    try:
        result = call_llm(prompt)
        if not result:
            raise ValueError("Empty response from LLM")
        return result
    except Exception as e:
        print(f"[ERROR] Brief LLM call failed: {e}")
        raise


# =============================================================================
# Upstream Evaluation
# =============================================================================

def run_upstream_eval(
    query: str,
    facts: list[dict],
    themes: list[str],
    dry_run: bool = False
) -> dict:
    """Run upstream evaluation (fact quality + coverage)."""

    # Format facts for prompt
    numbered_facts = "\n".join([
        f"[{f['id']}] {f['text']}"
        for f in facts
    ])

    # Format themes
    themes_str = "\n".join([f"- {t}" for t in themes])

    # Load and fill prompt (escape curly braces to prevent format string injection)
    prompt_template = load_prompt("upstream_eval")
    safe_query = query.replace("{", "{{").replace("}", "}}")
    safe_themes = themes_str.replace("{", "{{").replace("}", "}}")
    safe_facts = numbered_facts.replace("{", "{{").replace("}", "}}")
    prompt = prompt_template.format(
        query=safe_query,
        themes=safe_themes,
        numbered_facts=safe_facts
    )

    if dry_run:
        cost = estimate_cost(prompt)
        return {
            "dry_run": True,
            "prompt_chars": len(prompt),
            "estimated_cost": cost,
            "fact_count": len(facts),
            "theme_count": len(themes)
        }

    # Call LLM with error handling
    try:
        result = call_llm(prompt)
        if not result:
            raise ValueError("Empty response from LLM")
        return result
    except Exception as e:
        print(f"[ERROR] Upstream LLM call failed: {e}")
        raise


# =============================================================================
# Downstream Evaluation
# =============================================================================

def run_downstream_eval(
    query: str,
    prose_sections: list[dict],
    facts: list[dict],
    dry_run: bool = False
) -> dict:
    """Run downstream evaluation (citation accuracy + synthesis)."""

    if not prose_sections:
        return {"skip": True, "reason": "No prose sections in report"}

    # Format prose with citations
    prose_text = "\n\n".join([
        f"### {p['theme']}\n{p['prose']}"
        for p in prose_sections
    ])

    # Format facts for reference
    numbered_facts = "\n".join([
        f"[{f['id']}] {f['text']}"
        for f in facts
    ])

    # Load and fill prompt (escape curly braces to prevent format string injection)
    prompt_template = load_prompt("downstream_eval")
    safe_query = query.replace("{", "{{").replace("}", "}}")
    safe_prose = prose_text.replace("{", "{{").replace("}", "}}")
    safe_facts = numbered_facts.replace("{", "{{").replace("}", "}}")
    prompt = prompt_template.format(
        query=safe_query,
        prose_with_citations=safe_prose,
        numbered_facts=safe_facts
    )

    if dry_run:
        cost = estimate_cost(prompt)
        return {
            "dry_run": True,
            "prompt_chars": len(prompt),
            "estimated_cost": cost,
            "section_count": len(prose_sections)
        }

    # Call LLM with error handling
    try:
        result = call_llm(prompt)
        if not result:
            raise ValueError("Empty response from LLM")
        return result
    except Exception as e:
        print(f"[ERROR] Downstream LLM call failed: {e}")
        raise


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_dataset(
    path: str,
    mode: str = "medium",
    dry_run: bool = False,
    custom_limit: Optional[int] = None
) -> EvalResult:
    """Run full evaluation on a dataset."""

    # Parse JSON with error handling
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    query = data.get("query", "")
    brief = data.get("research_brief", "")

    # Extract data from JSON
    facts = extract_facts_from_json(data)
    themes = extract_themes_from_json(data)
    prose_sections = extract_prose_from_json(data)

    # Reject empty datasets
    if not facts:
        raise ValueError(f"No facts found in dataset {path}. Cannot evaluate empty dataset.")

    # Limit facts based on mode or custom limit
    if custom_limit:
        limit = custom_limit
    else:
        # "full" mode uses all facts, others have limits
        fact_limits = {"mini": 15, "medium": 50, "full": None}
        limit = fact_limits.get(mode, 50)
    if limit:
        facts = facts[:limit]

    print(f"[EVAL] Dataset: {Path(path).name}")
    print(f"[EVAL] Mode: {mode}, Facts: {len(facts)}, Themes: {len(themes)}")

    # Run brief eval (if brief exists)
    brief_result = {}
    if brief:
        print("[EVAL] Running brief eval...")
        brief_result = run_brief_eval(query, brief, dry_run)
        if dry_run:
            print(f"[DRY RUN] Brief: {brief_result.get('prompt_chars', 0)} chars, ~${brief_result.get('estimated_cost', 0):.4f}")
        elif brief_result.get("skip"):
            print(f"[EVAL] Brief: SKIP ({brief_result.get('reason')})")
        else:
            rec = brief_result.get("recommendation", "N/A")
            pres = brief_result.get("preservation_score", "N/A")
            dil = brief_result.get("dilution_score", "N/A")
            print(f"[EVAL] Brief: {rec} (preservation={pres}, dilution={dil})")
    else:
        print("[EVAL] Brief: SKIP (no research_brief in dataset)")

    # Run upstream eval
    print("[EVAL] Running upstream eval...")
    upstream = run_upstream_eval(query, facts, themes, dry_run)

    if dry_run:
        print(f"[DRY RUN] Upstream: {upstream['prompt_chars']} chars, ~${upstream['estimated_cost']:.4f}")
    else:
        print(f"[EVAL] Upstream: avg_quality={upstream.get('summary', {}).get('avg_quality', 'N/A')}")

    # Run downstream eval
    print("[EVAL] Running downstream eval...")
    downstream = run_downstream_eval(query, prose_sections, facts, dry_run)

    if dry_run:
        if downstream.get("skip"):
            print(f"[DRY RUN] Downstream: SKIP ({downstream.get('reason')})")
        else:
            print(f"[DRY RUN] Downstream: {downstream['prompt_chars']} chars, ~${downstream['estimated_cost']:.4f}")
    else:
        if downstream.get("skip"):
            print(f"[EVAL] Downstream: SKIP ({downstream.get('reason')})")
        else:
            print(f"[EVAL] Downstream: avg_accuracy={downstream.get('summary', {}).get('avg_accuracy', 'N/A')}")

    # Calculate match_score avg from existing data (coerce to float for type safety)
    match_scores = []
    for f in facts:
        score = f.get("match_score")
        if score is not None:
            try:
                match_scores.append(float(score))
            except (ValueError, TypeError):
                pass  # Skip non-numeric values
    match_score_avg = sum(match_scores) / len(match_scores) if match_scores else None

    # Build result
    if dry_run:
        total_cost = (
            brief_result.get("estimated_cost", 0) +
            upstream.get("estimated_cost", 0) +
            downstream.get("estimated_cost", 0)
        )
        return EvalResult(
            dataset=path,
            mode=mode,
            brief_preservation=None,
            brief_dilution=None,
            brief_assumptions=None,
            brief_recommendation=None,
            avg_fact_quality=0,
            avg_theme_coverage=0,
            duplicate_rate=0,
            low_quality_rate=0,
            match_score_avg=match_score_avg,
            avg_citation_accuracy=None,
            avg_synthesis_quality=None,
            uncited_rate=None,
            total_facts=len(facts),
            total_themes=len(themes),
            cost_estimate=total_cost
        )

    # Validate upstream response structure
    if "summary" not in upstream:
        raise ValueError(f"LLM upstream response missing 'summary' field. Got keys: {list(upstream.keys())}")
    up_summary = upstream["summary"]
    if "avg_quality" not in up_summary:
        raise ValueError(f"LLM upstream summary missing 'avg_quality'. Got: {up_summary}")

    fact_scores = upstream.get("fact_scores", [])
    low_quality = sum(1 for f in fact_scores if f.get("quality", 5) <= 2)

    # Validate downstream response structure (if not skipped)
    if downstream.get("skip"):
        down_summary = {}
    elif "summary" not in downstream:
        raise ValueError(f"LLM downstream response missing 'summary' field. Got keys: {list(downstream.keys())}")
    else:
        down_summary = downstream["summary"]

    # Extract brief metrics (if available)
    brief_preservation = None
    brief_dilution = None
    brief_assumptions = None
    brief_recommendation = None
    if brief_result and not brief_result.get("skip"):
        brief_preservation = brief_result.get("preservation_score")
        brief_dilution = brief_result.get("dilution_score")
        brief_assumptions = brief_result.get("assumptions_score")
        brief_recommendation = brief_result.get("recommendation")

    return EvalResult(
        dataset=path,
        mode=mode,
        brief_preservation=brief_preservation,
        brief_dilution=brief_dilution,
        brief_assumptions=brief_assumptions,
        brief_recommendation=brief_recommendation,
        avg_fact_quality=up_summary.get("avg_quality", 0),
        avg_theme_coverage=up_summary.get("avg_coverage", 0),
        duplicate_rate=up_summary.get("duplicate_count", 0) / max(len(facts), 1),
        low_quality_rate=low_quality / max(len(facts), 1),
        match_score_avg=match_score_avg,
        avg_citation_accuracy=down_summary.get("avg_accuracy"),
        avg_synthesis_quality=down_summary.get("avg_synthesis"),
        uncited_rate=down_summary.get("uncited_rate"),
        total_facts=len(facts),
        total_themes=len(themes),
        cost_estimate=0  # Actual cost from API response would go here
    )


def print_result(result: EvalResult):
    """Pretty print evaluation result."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULT: {Path(result.dataset).name}")
    print(f"{'='*60}")

    # Brief section
    brief_status = result.brief_status()
    if "SKIP" not in brief_status:
        print(f"\nBRIEF ({brief_status}):")
        if result.brief_preservation is not None:
            print(f"  Preservation:  {result.brief_preservation:.1f}/5 (target: ≥4)")
        if result.brief_dilution is not None:
            print(f"  Dilution:      {result.brief_dilution:.1f}/5 (target: ≥4, higher=less dilution)")
        if result.brief_assumptions is not None:
            print(f"  Assumptions:   {result.brief_assumptions:.1f}/5 (target: ≥4, higher=fewer assumptions)")
    else:
        print(f"\nBRIEF: {brief_status}")

    print(f"\nUPSTREAM ({result.upstream_status()}):")
    print(f"  Fact quality:    {result.avg_fact_quality:.2f} (target: ≥3.5)")
    print(f"  Theme coverage:  {result.avg_theme_coverage:.2f} (target: ≥3.5)")
    print(f"  Duplicate rate:  {result.duplicate_rate:.1%} (target: ≤15%)")
    print(f"  Low quality:     {result.low_quality_rate:.1%} (target: ≤10%)")
    if result.match_score_avg:
        print(f"  Match score:     {result.match_score_avg:.2f} (target: ≥0.8)")

    if result.avg_citation_accuracy is not None:
        print(f"\nDOWNSTREAM ({result.downstream_status()}):")
        print(f"  Citation accuracy: {result.avg_citation_accuracy:.2f} (target: ≥4.0)")
        if result.avg_synthesis_quality:
            print(f"  Synthesis quality: {result.avg_synthesis_quality:.2f} (target: ≥3.5)")
        if result.uncited_rate:
            print(f"  Uncited rate:      {result.uncited_rate:.1%} (target: ≤5%)")
    else:
        print(f"\nDOWNSTREAM: {result.downstream_status()}")

    print(f"\nOVERALL: {result.overall_status()}")
    print(f"Facts: {result.total_facts}, Themes: {result.total_themes}")
    if result.cost_estimate:
        print(f"Est. cost: ${result.cost_estimate:.4f}")
    print()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on gold datasets")
    parser.add_argument("dataset", nargs="?", help="Path to gold dataset JSON")
    parser.add_argument("--mode", choices=["mini", "medium", "full"], default="medium")
    parser.add_argument("--limit", type=int, help="Custom fact limit (overrides mode)")
    parser.add_argument("--all", action="store_true", help="Run on all gold datasets")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without calling LLM")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    gold_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "gold_queries"

    if args.all:
        if not gold_dir.exists():
            print(f"[ERROR] Gold datasets directory not found: {gold_dir}")
            sys.exit(1)
        datasets = sorted(gold_dir.glob("*.json"))
        if not datasets:
            print(f"[ERROR] No JSON files found in {gold_dir}")
            sys.exit(1)
        # Skip baseline files
        datasets = [d for d in datasets if "baseline" not in d.name]
    elif args.dataset:
        datasets = [Path(args.dataset)]
    else:
        print("Usage: run_eval.py <dataset.json> or run_eval.py --all")
        sys.exit(1)

    results = []
    for dataset in datasets:
        if not dataset.exists():
            print(f"[WARN] {dataset} not found, skipping")
            continue

        result = evaluate_dataset(str(dataset), args.mode, args.dry_run, args.limit)
        results.append(result)

        if not args.json:
            print_result(result)

    if args.json:
        output = [r.to_dict() for r in results]
        print(json.dumps(output, indent=2))

    # Exit with error if any failures (skip for dry run)
    if not args.dry_run and any(r.overall_status() == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
