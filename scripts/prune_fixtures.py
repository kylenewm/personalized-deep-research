#!/usr/bin/env python3
"""Prune and maintain fixture directories.

Identifies stale, redundant, and oversized fixture sets.

Usage:
    python scripts/prune_fixtures.py --stale        # Show fixtures older than 30 days
    python scripts/prune_fixtures.py --redundant    # Show potentially redundant fixtures
    python scripts/prune_fixtures.py --clean        # Remove stale extended fixtures
    python scripts/prune_fixtures.py --clean --dry-run  # Show what would be removed
    python scripts/prune_fixtures.py --promote <path>   # Promote fixture to core tier
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
FIXTURES_ROOT = project_root / "tests" / "fixtures"

STALE_DAYS = 30  # Fixtures not used in 30 days are stale

FIXTURE_DIRS = [
    "gold_queries",
    "extraction",
    "dedup",
    "synthesis",
    "arrangement",
    "sources"
]


def load_fixture_meta(path: Path) -> dict:
    """Load fixture and return metadata."""
    try:
        with open(path) as f:
            data = json.load(f)
        meta = data.get("_meta", {})
        return {
            "path": path,
            "name": path.stem,
            "size_kb": path.stat().st_size / 1024,
            "tier": meta.get("tier", "extended"),
            "created": meta.get("created"),
            "last_used": meta.get("last_used"),
            "purpose": meta.get("purpose", ""),
            "passing": meta.get("passing"),
            "has_meta": "_meta" in data
        }
    except Exception as e:
        return {
            "path": path,
            "name": path.stem,
            "error": str(e)
        }


def get_all_fixtures() -> list:
    """Get all fixtures with metadata."""
    fixtures = []
    for dir_name in FIXTURE_DIRS:
        dir_path = FIXTURES_ROOT / dir_name
        if not dir_path.exists():
            continue
        for f in dir_path.glob("*.json"):
            meta = load_fixture_meta(f)
            meta["directory"] = dir_name
            fixtures.append(meta)
    return fixtures


def find_stale_fixtures(fixtures: list, days: int = STALE_DAYS) -> list:
    """Find fixtures not used in the last N days."""
    cutoff = datetime.now() - timedelta(days=days)
    stale = []

    for f in fixtures:
        if f.get("error"):
            continue

        # Check last_used date
        last_used = f.get("last_used")
        if last_used:
            try:
                used_date = datetime.fromisoformat(last_used.replace("Z", "+00:00"))
                if used_date.replace(tzinfo=None) < cutoff:
                    stale.append({**f, "reason": f"Last used {last_used}"})
                continue
            except:
                pass

        # Fall back to created date
        created = f.get("created")
        if created:
            try:
                created_date = datetime.fromisoformat(created)
                if created_date < cutoff:
                    stale.append({**f, "reason": f"Created {created}, never used"})
                continue
            except:
                pass

        # Check file modification time
        mtime = datetime.fromtimestamp(f["path"].stat().st_mtime)
        if mtime < cutoff:
            stale.append({**f, "reason": f"Modified {mtime.strftime('%Y-%m-%d')}"})

    return stale


def find_redundant_fixtures(fixtures: list) -> list:
    """Find potentially redundant fixtures (same directory, similar queries)."""
    from collections import defaultdict

    by_dir = defaultdict(list)
    for f in fixtures:
        if not f.get("error"):
            by_dir[f["directory"]].append(f)

    redundant = []
    for dir_name, dir_fixtures in by_dir.items():
        if len(dir_fixtures) < 2:
            continue

        # Compare each pair
        for i, f1 in enumerate(dir_fixtures):
            for f2 in dir_fixtures[i+1:]:
                # Check name similarity (basic heuristic)
                name1 = f1["name"].lower().replace("_", " ").replace("-", " ")
                name2 = f2["name"].lower().replace("_", " ").replace("-", " ")

                words1 = set(name1.split())
                words2 = set(name2.split())

                common = words1 & words2
                if len(common) >= 3:  # At least 3 words in common
                    redundant.append({
                        "dir": dir_name,
                        "fixture1": f1["name"],
                        "fixture2": f2["name"],
                        "common_words": common,
                        "suggestion": "Consider consolidating"
                    })

    return redundant


def clean_stale(fixtures: list, dry_run: bool = True) -> list:
    """Remove stale extended fixtures."""
    stale = find_stale_fixtures(fixtures)

    # Never remove core fixtures
    to_remove = [f for f in stale if f.get("tier") != "core"]

    if dry_run:
        return to_remove

    removed = []
    for f in to_remove:
        try:
            f["path"].unlink()
            removed.append(f)
        except Exception as e:
            f["remove_error"] = str(e)

    return removed


def promote_to_core(fixture_path: str) -> bool:
    """Promote a fixture to core tier."""
    path = Path(fixture_path)
    if not path.exists():
        print(f"Error: {fixture_path} not found")
        return False

    try:
        with open(path) as f:
            data = json.load(f)

        if "_meta" not in data:
            data["_meta"] = {}

        data["_meta"]["tier"] = "core"
        data["_meta"]["promoted_at"] = datetime.now().strftime("%Y-%m-%d")

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Promoted to core: {path.name}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def print_fixture_summary(fixtures: list):
    """Print summary of fixture state."""
    print(f"\n{'='*60}")
    print("FIXTURE SUMMARY")
    print(f"{'='*60}")

    from collections import defaultdict
    by_dir = defaultdict(lambda: {"core": 0, "extended": 0, "total": 0, "size": 0})

    for f in fixtures:
        if f.get("error"):
            continue
        d = f["directory"]
        by_dir[d]["total"] += 1
        by_dir[d]["size"] += f.get("size_kb", 0)
        if f.get("tier") == "core":
            by_dir[d]["core"] += 1
        else:
            by_dir[d]["extended"] += 1

    print(f"\n{'Directory':<15} {'Core':<6} {'Extended':<10} {'Total':<8} {'Size'}")
    print("-" * 60)

    total_fixtures = 0
    total_size = 0
    for dir_name in FIXTURE_DIRS:
        stats = by_dir[dir_name]
        if stats["total"] > 0:
            print(f"{dir_name:<15} {stats['core']:<6} {stats['extended']:<10} {stats['total']:<8} {stats['size']:.1f} KB")
            total_fixtures += stats["total"]
            total_size += stats["size"]

    print("-" * 60)
    print(f"{'TOTAL':<15} {'':<6} {'':<10} {total_fixtures:<8} {total_size:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Prune and maintain fixtures")
    parser.add_argument("--stale", action="store_true", help=f"Show fixtures not used in {STALE_DAYS} days")
    parser.add_argument("--redundant", action="store_true", help="Show potentially redundant fixtures")
    parser.add_argument("--clean", action="store_true", help="Remove stale extended fixtures")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    parser.add_argument("--promote", metavar="PATH", help="Promote fixture to core tier")
    parser.add_argument("--summary", action="store_true", help="Show fixture summary")

    args = parser.parse_args()

    if args.promote:
        promote_to_core(args.promote)
        return

    fixtures = get_all_fixtures()

    if args.stale or (not args.redundant and not args.clean and not args.summary):
        stale = find_stale_fixtures(fixtures)
        print(f"\nStale Fixtures (>{STALE_DAYS} days):")
        print("-" * 60)
        if not stale:
            print("  None found")
        else:
            for f in stale:
                tier_badge = "[CORE]" if f.get("tier") == "core" else ""
                print(f"  {f['directory']}/{f['name']}.json {tier_badge}")
                print(f"    {f.get('reason', 'unknown')}")

    if args.redundant:
        redundant = find_redundant_fixtures(fixtures)
        print(f"\nPotentially Redundant Fixtures:")
        print("-" * 60)
        if not redundant:
            print("  None found")
        else:
            for r in redundant:
                print(f"  {r['dir']}/")
                print(f"    - {r['fixture1']}")
                print(f"    - {r['fixture2']}")
                print(f"    Common words: {', '.join(r['common_words'])}")
                print()

    if args.clean:
        removed = clean_stale(fixtures, dry_run=args.dry_run)
        action = "Would remove" if args.dry_run else "Removed"
        print(f"\n{action} {len(removed)} stale extended fixtures:")
        print("-" * 60)
        if not removed:
            print("  None to remove")
        else:
            for f in removed:
                print(f"  {f['directory']}/{f['name']}.json")
                if f.get("remove_error"):
                    print(f"    Error: {f['remove_error']}")

    if args.summary or (not args.stale and not args.redundant and not args.clean):
        print_fixture_summary(fixtures)


if __name__ == "__main__":
    main()
