#!/usr/bin/env python3
"""Run all sandboxes and report unified status.

Runs each sandbox against fixtures and reports pass/fail status.

Usage:
    python scripts/run_all_sandboxes.py           # Runs CORE fixtures only (~30s)
    python scripts/run_all_sandboxes.py --full    # Runs all fixtures (~2min)
    python scripts/run_all_sandboxes.py --dry     # Show what would run without executing
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
FIXTURES_ROOT = project_root / "tests" / "fixtures"
HEALTH_FILE = FIXTURES_ROOT / "HEALTH.json"


def get_fixtures(fixture_dir: str, full: bool = False) -> list:
    """Get fixtures to run, filtered by tier."""
    dir_path = FIXTURES_ROOT / fixture_dir
    if not dir_path.exists():
        return []

    fixtures = []
    for f in dir_path.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)

        meta = data.get("_meta", {})
        tier = meta.get("tier", "extended")

        # In default mode, only run core fixtures
        if not full and tier != "core":
            continue

        fixtures.append({
            "path": f,
            "tier": tier,
            "name": f.stem
        })

    return fixtures


def run_sandbox(sandbox_script: str, fixture_path: Path = None, timeout: int = 120) -> dict:
    """Run a single sandbox and capture result."""
    cmd = [sys.executable, str(project_root / "scripts" / sandbox_script)]
    if fixture_path:
        cmd.extend(["--fixture", str(fixture_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root)
        )
        return {
            "passed": result.returncode == 0,
            "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
            "error": result.stderr[-200:] if result.stderr else None,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "output": "",
            "error": f"Timeout after {timeout}s",
            "returncode": -1
        }
    except Exception as e:
        return {
            "passed": False,
            "output": "",
            "error": str(e),
            "returncode": -1
        }


def extract_metrics(output: str, sandbox_type: str) -> dict:
    """Extract key metrics from sandbox output."""
    import re

    metrics = {}

    if sandbox_type == "resynthesis":
        # Look for accuracy percentage
        match = re.search(r'Citation accuracy:\s*([\d.]+)%', output)
        if match:
            metrics["accuracy"] = f"{match.group(1)}%"
        match = re.search(r'(\d+)/(\d+)\s+correct', output)
        if match:
            metrics["correct"] = f"{match.group(1)}/{match.group(2)}"

    elif sandbox_type == "dedup":
        match = re.search(r'FP Rate:\s*([\d.]+)%', output)
        if match:
            metrics["fp_rate"] = f"{match.group(1)}%"
        match = re.search(r'Precision:\s*([\d.]+)%', output)
        if match:
            metrics["precision"] = f"{match.group(1)}%"

    elif sandbox_type == "arrangement":
        match = re.search(r'Facts Excluded:\s*(\d+)\s*\(([\d.]+)%\)', output)
        if match:
            metrics["exclusion"] = f"{match.group(2)}%"

    elif sandbox_type == "prompt":
        match = re.search(r'Avg words:\s*([\d.]+)', output)
        if match:
            metrics["avg_words"] = match.group(1)
        match = re.search(r'Artifacts:\s*(\d+)', output)
        if match:
            metrics["artifacts"] = match.group(1)

    elif sandbox_type == "source_authority":
        match = re.search(r'Tier 1.*?:\s*\d+\s*\(\s*([\d.]+)%\)', output)
        if match:
            metrics["tier1"] = f"{match.group(1)}%"
        match = re.search(r'Tier 3.*?:\s*\d+\s*\(\s*([\d.]+)%\)', output)
        if match:
            metrics["tier3"] = f"{match.group(1)}%"

    return metrics


def update_health(results: dict, full: bool):
    """Update HEALTH.json with latest run info."""
    # Count fixtures per directory
    fixture_counts = {}
    for ft in ["gold_queries", "extraction", "dedup", "synthesis", "arrangement", "sources"]:
        dir_path = FIXTURES_ROOT / ft
        if dir_path.exists():
            files = list(dir_path.glob("*.json"))
            core_count = 0
            for f in files:
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    if data.get("_meta", {}).get("tier") == "core":
                        core_count += 1
                except:
                    pass
            fixture_counts[ft] = {
                "count": len(files),
                "core": core_count,
                "extended": len(files) - core_count
            }

    # Calculate total size
    total_size = sum(
        f.stat().st_size
        for ft in fixture_counts
        for f in (FIXTURES_ROOT / ft).glob("*.json")
        if f.exists()
    )

    health = {
        "last_updated": datetime.now().isoformat(),
        "last_run_mode": "full" if full else "core",
        "total_fixtures": sum(fc["count"] for fc in fixture_counts.values()),
        "by_directory": fixture_counts,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "results_summary": {
            name: {"passed": r["passed"]}
            for name, r in results.items()
        }
    }

    FIXTURES_ROOT.mkdir(parents=True, exist_ok=True)
    with open(HEALTH_FILE, "w") as f:
        json.dump(health, f, indent=2)


async def run_all(full: bool = False, dry_run: bool = False):
    """Run all sandboxes."""
    print(f"Sandbox Results ({datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 70)

    mode = "FULL" if full else "CORE"
    print(f"Mode: {mode}")
    print()

    # Define sandboxes and their fixture directories
    sandboxes = [
        ("prompt_sandbox.py", "extraction", "prompt"),
        ("dedup_sandbox.py", "dedup", "dedup"),
        ("resynthesis_test.py", "synthesis", "resynthesis"),
        ("arrangement_sandbox.py", "arrangement", "arrangement"),
    ]

    # Add source authority as a special case (uses --analyze flag)
    # It uses synthesis fixtures for facts

    results = {}
    total_time = 0

    for script, fixture_dir, sandbox_type in sandboxes:
        fixtures = get_fixtures(fixture_dir, full)

        if not fixtures:
            # Run with default fixtures if none found
            if dry_run:
                print(f"[ ] {script:25} | Would run with defaults")
                continue

            start = datetime.now()
            result = run_sandbox(script)
            elapsed = (datetime.now() - start).total_seconds()
            total_time += elapsed

            status = "[+]" if result["passed"] else "[-]"
            metrics = extract_metrics(result["output"], sandbox_type)
            metrics_str = " | ".join(f"{k}: {v}" for k, v in metrics.items()) if metrics else "no metrics"

            print(f"{status} {script:25} | {metrics_str} | {elapsed:.1f}s")
            results[script] = result
        else:
            # Run with each fixture
            for fixture in fixtures:
                if dry_run:
                    print(f"[ ] {script:25} | {fixture['name']} ({fixture['tier']})")
                    continue

                start = datetime.now()
                result = run_sandbox(script, fixture["path"])
                elapsed = (datetime.now() - start).total_seconds()
                total_time += elapsed

                status = "[+]" if result["passed"] else "[-]"
                metrics = extract_metrics(result["output"], sandbox_type)
                metrics_str = " | ".join(f"{k}: {v}" for k, v in metrics.items()) if metrics else ""

                key = f"{script}:{fixture['name']}"
                print(f"{status} {key:40} | {metrics_str} | {elapsed:.1f}s")
                results[key] = result

    # Run source authority on synthesis fixtures
    synthesis_fixtures = get_fixtures("synthesis", full)
    for fixture in synthesis_fixtures:
        if dry_run:
            print(f"[ ] source_authority:25   | {fixture['name']}")
            continue

        start = datetime.now()
        cmd = [sys.executable, str(project_root / "scripts" / "resynthesis_test.py"),
               "--analyze", str(fixture["path"])]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(project_root))
            result = {"passed": proc.returncode == 0, "output": proc.stdout}
        except Exception as e:
            result = {"passed": False, "output": "", "error": str(e)}

        elapsed = (datetime.now() - start).total_seconds()
        total_time += elapsed

        status = "[+]" if result["passed"] else "[-]"
        metrics = extract_metrics(result.get("output", ""), "source_authority")
        metrics_str = " | ".join(f"{k}: {v}" for k, v in metrics.items()) if metrics else ""

        key = f"source_authority:{fixture['name']}"
        print(f"{status} {key:40} | {metrics_str} | {elapsed:.1f}s")
        results[key] = result

    if dry_run:
        print("\n(dry run - no tests executed)")
        return

    # Summary
    print("=" * 70)
    passed = sum(1 for r in results.values() if r.get("passed", False))
    total = len(results)
    all_passed = passed == total

    print(f"Overall: {passed}/{total} {'PASS' if all_passed else 'FAIL'}")
    print(f"Total time: {total_time:.1f}s")

    # Update health file
    update_health(results, full)
    print(f"\nHealth updated: {HEALTH_FILE.relative_to(project_root)}")

    if not all_passed:
        print("\nFailed sandboxes:")
        for name, result in results.items():
            if not result.get("passed", False):
                print(f"  - {name}")
                if result.get("error"):
                    print(f"    Error: {result['error']}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run all sandboxes")
    parser.add_argument("--full", action="store_true", help="Run all fixtures (not just core)")
    parser.add_argument("--dry", action="store_true", help="Show what would run")

    args = parser.parse_args()

    passed = asyncio.run(run_all(full=args.full, dry_run=args.dry))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
