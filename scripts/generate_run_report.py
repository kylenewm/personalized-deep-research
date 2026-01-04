"""Generate detailed run report from final pipeline state.

This script analyzes the final state of a research run and generates
a human-readable Markdown report showing what happened, extraction quality,
verification results, and recommendations for optimization.

Usage:
    # As a module (called from test scripts):
    from generate_run_report import generate_run_report
    generate_run_report(final_state, "run_report.md")

    # Standalone (with JSON state file):
    python scripts/generate_run_report.py state.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis if too long."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or url[:30]
    except Exception:
        return url[:30]


def analyze_sources(sources: List[Dict]) -> Dict[str, Any]:
    """Analyze source extraction statistics."""
    if not sources:
        return {
            "total": 0,
            "by_method": {},
            "by_domain": {},
            "content_sizes": [],
            "successful": 0,
            "failed": 0,
        }

    by_method = {}
    by_domain = {}
    content_sizes = []
    successful = 0
    failed = 0

    for source in sources:
        # Count by extraction method
        method = source.get("extraction_method", "unknown")
        by_method[method] = by_method.get(method, 0) + 1

        # Count by domain
        url = source.get("url", "")
        domain = get_domain(url)
        by_domain[domain] = by_domain.get(domain, 0) + 1

        # Track content sizes
        content = source.get("content", "")
        content_len = len(content) if content else 0
        content_sizes.append(content_len)

        if content_len > 0:
            successful += 1
        else:
            failed += 1

    return {
        "total": len(sources),
        "by_method": by_method,
        "by_domain": by_domain,
        "content_sizes": content_sizes,
        "successful": successful,
        "failed": failed,
        "avg_content_size": sum(content_sizes) / len(content_sizes) if content_sizes else 0,
    }


def analyze_snippets(snippets: List[Dict]) -> Dict[str, Any]:
    """Analyze evidence snippet statistics."""
    if not snippets:
        return {
            "total": 0,
            "by_status": {"PASS": 0, "FAIL": 0, "SKIP": 0, "PENDING": 0},
            "by_source": {},
            "failed_samples": [],
            "passed_samples": [],
        }

    by_status = {"PASS": 0, "FAIL": 0, "SKIP": 0, "PENDING": 0}
    by_source = {}
    failed_samples = []
    passed_samples = []

    for snippet in snippets:
        # Count by status
        status = snippet.get("status", "PENDING")
        if status in by_status:
            by_status[status] += 1
        else:
            by_status[status] = 1

        # Count by source URL
        url = snippet.get("source_url", snippet.get("url", "unknown"))
        domain = get_domain(url)
        by_source[domain] = by_source.get(domain, 0) + 1

        # Collect samples
        quote = snippet.get("quote", snippet.get("text", ""))
        if status == "FAIL" and len(failed_samples) < 10:
            failed_samples.append({
                "quote": quote,
                "source": domain,
                "reason": snippet.get("verification_reason", "No reason provided"),
            })
        elif status == "PASS" and len(passed_samples) < 5:
            passed_samples.append({
                "quote": quote,
                "source": domain,
            })

    return {
        "total": len(snippets),
        "by_status": by_status,
        "by_source": by_source,
        "failed_samples": failed_samples,
        "passed_samples": passed_samples,
    }


def analyze_report(report: str) -> Dict[str, Any]:
    """Analyze the final report structure."""
    if not report:
        return {
            "length_chars": 0,
            "length_words": 0,
            "sections": [],
            "has_citations": False,
        }

    words = report.split()

    # Try to detect sections (markdown headers)
    sections = []
    for line in report.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            sections.append(("h1", line[2:].strip()))
        elif line.startswith("## "):
            sections.append(("h2", line[3:].strip()))
        elif line.startswith("### "):
            sections.append(("h3", line[4:].strip()))

    # Check for citations
    has_citations = "[" in report and "]" in report

    return {
        "length_chars": len(report),
        "length_words": len(words),
        "sections": sections,
        "has_citations": has_citations,
    }


def generate_recommendations(
    source_stats: Dict,
    snippet_stats: Dict,
    verification: Dict,
) -> List[str]:
    """Generate optimization recommendations based on stats."""
    recommendations = []

    # Source extraction recommendations
    if source_stats.get("failed", 0) > 0:
        fail_rate = source_stats["failed"] / source_stats["total"] * 100 if source_stats["total"] > 0 else 0
        if fail_rate > 20:
            recommendations.append(
                f"High extraction failure rate ({fail_rate:.0f}%). Consider adding retry logic or filtering problematic domains."
            )

    by_method = source_stats.get("by_method", {})
    if by_method.get("fallback", 0) > by_method.get("extract_api", 0):
        recommendations.append(
            "More fallback extractions than API extractions. Check Tavily Extract API availability/limits."
        )

    # Verification recommendations
    by_status = snippet_stats.get("by_status", {})
    total_verified = by_status.get("PASS", 0) + by_status.get("FAIL", 0)
    if total_verified > 0:
        fail_rate = by_status.get("FAIL", 0) / total_verified * 100
        if fail_rate > 30:
            recommendations.append(
                f"High verification failure rate ({fail_rate:.0f}%). Consider:\n"
                "  - Lowering Jaccard threshold (currently 0.8) to 0.7\n"
                "  - Checking if quotes are being paraphrased during research\n"
                "  - Verifying source content hasn't changed"
            )

    if by_status.get("SKIP", 0) > by_status.get("PASS", 0):
        recommendations.append(
            "Many snippets skipped. Check if sources are being stored correctly."
        )

    # Content size recommendations
    avg_size = source_stats.get("avg_content_size", 0)
    if avg_size < 1000:
        recommendations.append(
            f"Average content size is low ({avg_size:.0f} chars). Extraction may be truncating content."
        )
    elif avg_size > 50000:
        recommendations.append(
            f"Average content size is very high ({avg_size:.0f} chars). Consider limiting to reduce processing time."
        )

    if not recommendations:
        recommendations.append("No major issues detected. Run completed successfully.")

    return recommendations


def generate_run_report(
    state: Dict[str, Any],
    output_path: str = "run_report.md",
    run_duration: Optional[float] = None,
) -> str:
    """Generate a detailed run report from final state.

    Args:
        state: Final pipeline state dictionary
        output_path: Path to write the report (use None to skip writing)
        run_duration: Optional run duration in seconds

    Returns:
        The generated Markdown report string
    """
    # Extract state components
    sources = state.get("source_store", [])
    snippets = state.get("evidence_snippets", [])
    verification = state.get("verification_result", {})
    report = state.get("final_report", "")
    brief = state.get("research_brief", "")
    messages = state.get("messages", [])

    # Get original query from messages
    query = ""
    for msg in messages:
        if isinstance(msg, tuple) and msg[0] == "user":
            query = msg[1]
            break
        elif hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    # Analyze components
    source_stats = analyze_sources(sources)
    snippet_stats = analyze_snippets(snippets)
    report_stats = analyze_report(report)
    recommendations = generate_recommendations(source_stats, snippet_stats, verification)

    # Build report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# Run Report: {timestamp}

## Overview

| Metric | Value |
|--------|-------|
| Query | {truncate(query, 60)} |
| Duration | {f'{run_duration:.1f}s' if run_duration else 'N/A'} |
| Sources Extracted | {source_stats['total']} ({source_stats['successful']} successful) |
| Evidence Snippets | {snippet_stats['total']} |
| Report Length | {report_stats['length_words']} words |

---

## 1. Research Brief

{truncate(brief, 500) if brief else '(No brief generated)'}

---

## 2. Source Extraction

### Summary
- **Total sources:** {source_stats['total']}
- **Successful:** {source_stats['successful']} ({source_stats['successful']/source_stats['total']*100:.0f}% success rate)
- **Failed:** {source_stats['failed']}
- **Average content size:** {source_stats['avg_content_size']:.0f} chars

### By Extraction Method

| Method | Count |
|--------|-------|
"""

    for method, count in sorted(source_stats["by_method"].items(), key=lambda x: -x[1]):
        md += f"| {method} | {count} |\n"

    md += """
### Top Domains

| Domain | Sources |
|--------|---------|
"""

    for domain, count in sorted(source_stats["by_domain"].items(), key=lambda x: -x[1])[:10]:
        md += f"| {domain} | {count} |\n"

    md += f"""
---

## 3. Evidence Extraction

### Verification Status

| Status | Count | Percentage |
|--------|-------|------------|
"""

    total_snippets = snippet_stats["total"] or 1  # Avoid division by zero
    for status in ["PASS", "FAIL", "SKIP", "PENDING"]:
        count = snippet_stats["by_status"].get(status, 0)
        pct = count / total_snippets * 100
        emoji = {"PASS": "✓", "FAIL": "✗", "SKIP": "○", "PENDING": "…"}.get(status, "?")
        md += f"| {emoji} {status} | {count} | {pct:.1f}% |\n"

    md += """
### By Source Domain

| Domain | Quotes |
|--------|--------|
"""

    for domain, count in sorted(snippet_stats["by_source"].items(), key=lambda x: -x[1])[:10]:
        md += f"| {domain} | {count} |\n"

    # Failed samples
    if snippet_stats["failed_samples"]:
        md += """
### Failed Verification Samples

"""
        for i, sample in enumerate(snippet_stats["failed_samples"][:5], 1):
            md += f"{i}. **Source:** {sample['source']}\n"
            md += f"   **Quote:** \"{truncate(sample['quote'], 100)}\"\n"
            md += f"   **Reason:** {truncate(sample['reason'], 80)}\n\n"

    # Passed samples
    if snippet_stats["passed_samples"]:
        md += """
### Verified Quote Samples

"""
        for i, sample in enumerate(snippet_stats["passed_samples"][:3], 1):
            md += f"{i}. \"{truncate(sample['quote'], 100)}\" — *{sample['source']}*\n\n"

    md += f"""
---

## 4. Final Report Analysis

- **Length:** {report_stats['length_chars']:,} characters, {report_stats['length_words']:,} words
- **Has citations:** {'Yes' if report_stats['has_citations'] else 'No'}

### Structure
"""

    if report_stats["sections"]:
        for level, title in report_stats["sections"]:
            indent = {"h1": "", "h2": "  ", "h3": "    "}.get(level, "")
            md += f"{indent}- {title}\n"
    else:
        md += "- (No markdown headers detected)\n"

    md += """
### Report Preview (first 500 chars)

```
"""
    md += truncate(report, 500) if report else "(No report generated)"
    md += """
```

---

## 5. Verification Summary

"""

    if verification:
        summary = verification.get("summary", {})
        md += f"""| Metric | Value |
|--------|-------|
| Total Claims | {summary.get('total_claims', 'N/A')} |
| Supported | {summary.get('supported', 'N/A')} |
| Partially Supported | {summary.get('partially_supported', 'N/A')} |
| Unsupported | {summary.get('unsupported', 'N/A')} |
| Overall Confidence | {summary.get('overall_confidence', 0):.0%} |

"""
        warnings = summary.get("warnings", [])
        if warnings:
            md += "### Warnings\n\n"
            for w in warnings[:5]:
                md += f"- {truncate(w, 100)}\n"
            md += "\n"
    else:
        md += "(No verification data available)\n\n"

    md += """---

## 6. Recommendations

"""

    for i, rec in enumerate(recommendations, 1):
        md += f"{i}. {rec}\n\n"

    md += f"""
---

*Report generated at {timestamp}*
"""

    # Write to file if path provided
    if output_path:
        output = Path(output_path)
        output.write_text(md)
        print(f"[REPORT] Generated: {output.absolute()}")

    return md


def main():
    """CLI entry point for standalone usage."""
    if len(sys.argv) < 2:
        print("Usage: python generate_run_report.py <state.json> [output.md]")
        print("\nReads a JSON file containing the final pipeline state and generates a report.")
        sys.exit(1)

    state_path = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else "run_report.md"

    if not state_path.exists():
        print(f"Error: State file not found: {state_path}")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    report = generate_run_report(state, output_path)
    print(f"\n{'='*60}")
    print(report)


if __name__ == "__main__":
    main()
