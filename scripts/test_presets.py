#!/usr/bin/env python3
"""
Test Presets for Deep Research

Tests the fast, balanced, and thorough presets with a research query.
Measures time, sources found, and report quality for comparison.

Usage:
    python scripts/test_presets.py --preset fast
    python scripts/test_presets.py --preset balanced
    python scripts/test_presets.py --preset thorough
    python scripts/test_presets.py --all  # Run all presets
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env file BEFORE any other imports that might need API keys
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}")

# Verify API keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment")
    sys.exit(1)
if not os.getenv("TAVILY_API_KEY"):
    print("Warning: TAVILY_API_KEY not found - will use default search")


async def run_research_with_preset(query: str, preset: str) -> dict:
    """Run research with a specific preset and return metrics."""
    from langchain_core.messages import HumanMessage
    from open_deep_research.graph import deep_researcher

    print(f"\n{'='*70}")
    print(f"RUNNING PRESET: {preset.upper()}")
    print(f"QUERY: {query}")
    print(f"{'='*70}\n")

    config = {
        "configurable": {
            "preset": preset,
            "allow_clarification": False,  # Skip clarification for testing
        }
    }

    start_time = time.time()

    try:
        # Run the research pipeline - input must be messages list
        result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )

        end_time = time.time()
        duration = end_time - start_time

        # Extract metrics
        metrics = {
            "preset": preset,
            "query": query,
            "duration_seconds": round(duration, 2),
            "success": True,
            "error": None,
            "report_length": len(result.get("final_report", "") or ""),
            "sources_stored": len(result.get("source_store", []) or []),
            "notes_length": len(result.get("notes", "") or ""),
            "brief_length": len(result.get("research_brief", "") or ""),
        }

        # Print summary
        print(f"\n{'='*70}")
        print(f"PRESET '{preset.upper()}' COMPLETE")
        print(f"{'='*70}")
        print(f"Duration: {duration:.1f}s")
        print(f"Report Length: {metrics['report_length']:,} chars")
        print(f"Sources: {metrics['sources_stored']}")
        print(f"Notes: {metrics['notes_length']:,} chars")
        print(f"{'='*70}\n")

        # Save report
        output_dir = Path(__file__).parent.parent / "metrics" / f"preset_{preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        with open(output_dir / "report.md", 'w') as f:
            f.write(result.get("final_report", "") or "No report generated")

        print(f"Results saved to: {output_dir}")

        return metrics

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{'='*70}")
        print(f"PRESET '{preset.upper()}' FAILED")
        print(f"{'='*70}")
        print(f"Error: {e}")
        print(f"Duration before failure: {duration:.1f}s")
        print(f"{'='*70}\n")

        return {
            "preset": preset,
            "query": query,
            "duration_seconds": round(duration, 2),
            "success": False,
            "error": str(e),
            "report_length": 0,
            "sources_stored": 0,
            "notes_length": 0,
            "brief_length": 0,
        }


async def run_all_presets(query: str) -> list:
    """Run all presets sequentially and compare results."""
    results = []

    for preset in ["fast", "balanced", "thorough"]:
        result = await run_research_with_preset(query, preset)
        results.append(result)

        # Small delay between runs to avoid rate limits
        if preset != "thorough":
            print("Waiting 10 seconds before next preset...")
            await asyncio.sleep(10)

    # Print comparison
    print("\n" + "="*70)
    print("PRESET COMPARISON")
    print("="*70)
    print(f"{'Preset':<12} {'Time (s)':<12} {'Report':<12} {'Sources':<10} {'Success':<10}")
    print("-"*70)

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{r['preset']:<12} {r['duration_seconds']:<12} {r['report_length']:<12,} {r['sources_stored']:<10} {status:<10}")

    print("="*70 + "\n")

    # Save comparison
    output_path = Path(__file__).parent.parent / "metrics" / f"preset_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Comparison saved to: {output_path}")

    return results


async def main():
    parser = argparse.ArgumentParser(description='Test Deep Research presets')
    parser.add_argument('--preset', '-p', choices=['fast', 'balanced', 'thorough'],
                       help='Preset to test')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all presets for comparison')
    parser.add_argument('--query', '-q',
                       default="What are the top ways of preventing hallucinations with deep research",
                       help='Research query to test')

    args = parser.parse_args()

    if not args.preset and not args.all:
        parser.print_help()
        print("\nERROR: Must specify --preset or --all")
        sys.exit(1)

    if args.all:
        await run_all_presets(args.query)
    else:
        await run_research_with_preset(args.query, args.preset)


if __name__ == "__main__":
    asyncio.run(main())
