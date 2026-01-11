#!/usr/bin/env python3
"""Dedup sandbox - test LLM deduplication against labeled pairs.

Measures false positive/negative rates using LLM semantic matching.
Uses labeled pairs from tests/fixtures/dedup_labeled_pairs.json.

Usage:
    python scripts/dedup_sandbox.py           # Test LLM dedup
    python scripts/dedup_sandbox.py --dry     # Show pairs without LLM call
"""

import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MockExtraction:
    """Mock extraction for testing."""
    extracted_text: str
    match_score: float = 1.0
    source_url: str = "http://example.com"

    @property
    def pointer(self):
        return None

    @property
    def status(self):
        return "verified"


async def test_llm_dedup(pairs: list, dry_run: bool = False) -> dict:
    """Test LLM dedup against labeled pairs.

    Creates mock extractions from pairs and runs LLM dedup.
    """
    from openai import AsyncOpenAI
    from open_deep_research.pipeline_v2 import (
        DEDUP_PROMPT,
        format_facts_for_dedup,
        parse_dedup_response
    )

    # Build list of all unique facts
    facts = []
    fact_to_pair = {}  # Map fact index to pair info

    for pair in pairs:
        # Add fact_a if not already present
        if pair["fact_a"] not in [f.extracted_text for f in facts]:
            facts.append(MockExtraction(pair["fact_a"]))
            fact_to_pair[len(facts)] = {"pair_id": pair["id"], "is_a": True}

        # Add fact_b
        if pair["fact_b"] not in [f.extracted_text for f in facts]:
            facts.append(MockExtraction(pair["fact_b"]))
            fact_to_pair[len(facts)] = {"pair_id": pair["id"], "is_a": False}

    print(f"Testing {len(facts)} unique facts from {len(pairs)} pairs")

    if dry_run:
        print("\n--- Facts ---")
        for i, f in enumerate(facts, 1):
            print(f"[{i}] {f.extracted_text[:80]}...")
        return {"dry_run": True}

    # Call LLM
    client = AsyncOpenAI()

    facts_text = format_facts_for_dedup(facts)
    prompt = DEDUP_PROMPT.format(facts=facts_text)

    print("Calling LLM for dedup...")
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.1
    )
    response = resp.choices[0].message.content

    print(f"\nLLM Response:\n{response}\n")

    duplicate_groups = parse_dedup_response(response)
    print(f"Parsed {len(duplicate_groups)} duplicate groups: {duplicate_groups}")

    # Evaluate against labeled pairs
    results = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "details": []
    }

    # Build set of pairs that LLM thinks are duplicates
    llm_dup_pairs = set()
    for group in duplicate_groups:
        # All pairs within this group are considered duplicates
        for i, idx1 in enumerate(group):
            for idx2 in group[i+1:]:
                llm_dup_pairs.add((min(idx1, idx2), max(idx1, idx2)))

    # Map each fact text to its index
    text_to_idx = {f.extracted_text: i+1 for i, f in enumerate(facts)}

    # Evaluate each labeled pair
    for pair in pairs:
        idx_a = text_to_idx.get(pair["fact_a"])
        idx_b = text_to_idx.get(pair["fact_b"])

        if idx_a is None or idx_b is None:
            continue

        # Handle identical text (same index = definitely duplicate)
        if idx_a == idx_b:
            predicted_dup = True  # Identical text = duplicate
        else:
            pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
            predicted_dup = pair_key in llm_dup_pairs

        actual_dup = pair["is_duplicate"]

        if predicted_dup and actual_dup:
            results["true_positive"] += 1
            status = "TP"
        elif not predicted_dup and not actual_dup:
            results["true_negative"] += 1
            status = "TN"
        elif predicted_dup and not actual_dup:
            results["false_positive"] += 1
            status = "FP"
        else:
            results["false_negative"] += 1
            status = "FN"

        results["details"].append({
            "id": pair["id"],
            "indices": pair_key,
            "predicted": predicted_dup,
            "actual": actual_dup,
            "status": status,
            "reason": pair.get("reason", "")
        })

    # Calculate metrics
    tp = results["true_positive"]
    tn = results["true_negative"]
    fp = results["false_positive"]
    fn = results["false_negative"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = f1
    results["accuracy"] = accuracy
    results["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print("LLM DEDUP RESULTS")
    print(f"{'='*60}")
    print(f"  True Positives:  {results['true_positive']}")
    print(f"  True Negatives:  {results['true_negative']}")
    print(f"  False Positives: {results['false_positive']} (different facts marked as duplicate)")
    print(f"  False Negatives: {results['false_negative']} (duplicates missed)")
    print()
    print(f"  Precision: {results['precision']:.1%}")
    print(f"  Recall:    {results['recall']:.1%}")
    print(f"  F1 Score:  {results['f1']:.1%}")
    print(f"  Accuracy:  {results['accuracy']:.1%}")
    print(f"  FP Rate:   {results['false_positive_rate']:.1%} (target: <5%)")

    # Quality check
    passed = results['false_positive_rate'] < 0.05 and results['recall'] > 0.8
    status = "PASS" if passed else "FAIL"
    print(f"\n  Status: {'✅' if passed else '❌'} {status}")

    print(f"\n--- Details ---")
    for d in results["details"]:
        marker = "⚠️" if d["status"] in ["FP", "FN"] else "  "
        print(f"{marker} [{d['id']:2d}] {d['status']} - {d['reason'][:50]}")


async def run_sandbox(dry_run: bool = False):
    """Run dedup sandbox."""
    pairs_path = project_root / "tests/fixtures/dedup_labeled_pairs.json"
    if not pairs_path.exists():
        print(f"Error: {pairs_path} not found")
        return

    with open(pairs_path) as f:
        data = json.load(f)

    pairs = data["pairs"]
    print(f"Loaded {len(pairs)} labeled pairs")
    print(f"  Duplicates: {sum(1 for p in pairs if p['is_duplicate'])}")
    print(f"  Different:  {sum(1 for p in pairs if not p['is_duplicate'])}")

    results = await test_llm_dedup(pairs, dry_run)

    if not dry_run:
        print_results(results)

        # Log results
        log_path = project_root / "dedup_sandbox_log.jsonl"
        log_entry = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "method": "llm",
            "pairs_count": len(pairs),
            "accuracy": results.get("accuracy", 0),
            "f1": results.get("f1", 0),
            "fp_rate": results.get("false_positive_rate", 0)
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def main():
    dry_run = "--dry" in sys.argv
    asyncio.run(run_sandbox(dry_run))


if __name__ == "__main__":
    main()
