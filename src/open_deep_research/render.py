"""Report rendering module.

Renders HybridReport objects to HTML using external templates.
Keeps rendering logic separate from pipeline logic.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

# Handle imports for both module and standalone usage
try:
    from .pipeline_v2 import HybridReport, ThemedSection
    from .pointer_extract import Extraction
except ImportError:
    from pipeline_v2 import HybridReport, ThemedSection
    from pointer_extract import Extraction


# =============================================================================
# Template Loading
# =============================================================================

TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"
DEFAULT_TEMPLATE = "report.html"


def get_template_path(template_name: str = DEFAULT_TEMPLATE) -> Path:
    """Get path to a template file."""
    return TEMPLATE_DIR / template_name


def load_template(template_name: str = DEFAULT_TEMPLATE) -> str:
    """Load template from templates directory."""
    path = get_template_path(template_name)
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text()


# =============================================================================
# Helpers
# =============================================================================

def extract_domain(url: Optional[str]) -> str:
    """Extract domain from URL (e.g., 'https://nature.com/article' -> 'nature.com')."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove 'www.' prefix if present
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def fact_to_dict(fact: Extraction) -> dict:
    """Convert Extraction object to template-friendly dict."""
    return {
        "extracted_text": fact.extracted_text or "",
        "source_url": fact.source_url or "",
        "source_title": fact.pointer.context if fact.pointer else "",
        "source_domain": extract_domain(fact.source_url),
        "match_score": fact.match_score,
    }


def section_to_dict(section: ThemedSection) -> dict:
    """Convert ThemedSection object to template-friendly dict."""
    return {
        "theme": section.theme,
        "intro": section.intro,
        "facts": [fact_to_dict(f) for f in section.facts],
        "transitions": section.transitions,
    }


def report_to_dict(report: HybridReport) -> dict:
    """Convert HybridReport object to template-friendly dict."""
    return {
        "title": report.title,
        "executive_summary": report.executive_summary,
        "sections": [section_to_dict(s) for s in report.sections],
        "analysis": report.analysis,
        "conclusion": report.conclusion,
        "stats": {
            "total_extracted": report.total_extracted,
            "total_verified": report.total_verified,
            "total_used": report.total_used,
            "themes": len(report.sections),
        },
    }


# =============================================================================
# Rendering
# =============================================================================

def render_fact_html(fact: dict) -> str:
    """Render a single fact card to HTML."""
    confidence_html = ""
    if fact.get("match_score"):
        score_pct = int(fact["match_score"] * 100)
        confidence_html = f'''
                <span class="source-sep">/</span>
                <span class="confidence">{score_pct}%</span>'''

    return f'''<div class="fact">
            <p class="fact-text">{fact["extracted_text"]}</p>
            <div class="fact-source">
                <a href="{fact["source_url"]}" target="_blank" rel="noopener">{fact["source_title"]}</a>
                <span class="source-sep">/</span>
                <span>{fact.get("source_domain", "")}</span>{confidence_html}
            </div>
        </div>'''


def render_section_html(section: dict) -> str:
    """Render a section with its facts to HTML."""
    parts = [f'<h3>{section["theme"]}</h3>']

    if section.get("intro"):
        parts.append(f'<p class="synthesis">{section["intro"]}</p>')

    for fact in section.get("facts", []):
        parts.append(render_fact_html(fact))

    return "\n".join(parts)


def render_html(data: dict, template: Optional[str] = None) -> str:
    """Render report data dict to HTML.

    Args:
        data: Report data in template-friendly dict format
        template: Optional template string. If not provided, loads default template.

    Returns:
        Rendered HTML string
    """
    if template is None:
        template = load_template()

    # Extract CSS from template
    css_match = re.search(r'<style>(.*?)</style>', template, re.DOTALL)
    css = css_match.group(1) if css_match else ""

    # Split sections for two columns
    sections = data.get("sections", [])
    mid = (len(sections) + 1) // 2
    left_sections = sections[:mid]
    right_sections = sections[mid:]

    # Render columns
    left_html = ['<h2>Verified Findings</h2>']
    for section in left_sections:
        left_html.append(render_section_html(section))

    right_html = ['<h2>&nbsp;</h2>']
    for section in right_sections:
        right_html.append(render_section_html(section))

    # Render analysis paragraphs
    analysis_paras = []
    for para in data.get("analysis", "").split('\n\n'):
        if para.strip():
            analysis_paras.append(f'<p>{para.strip()}</p>')

    stats = data.get("stats", {})

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{data.get("title", "Research Report")}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>{css}</style>
</head>
<body>

<header>
    <p class="overline">Research Report</p>
    <h1>{data.get("title", "")}</h1>
    <p class="summary">{data.get("executive_summary", "")}</p>
</header>

<div class="columns">

    <div class="column">
        {chr(10).join(left_html)}
    </div>

    <div class="column">
        {chr(10).join(right_html)}
    </div>

    <div class="full-width">
        <h2>Analysis</h2>
        <div class="prose">
            {chr(10).join(analysis_paras)}
        </div>
    </div>

    <div class="full-width" style="margin-top: 0; border-top: none; padding-top: 1.5rem;">
        <h2>Conclusion</h2>
        <div class="prose">
            <p>{data.get("conclusion", "")}</p>
        </div>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{stats.get("total_extracted", 0)}</div>
            <div class="stat-label">Sources</div>
        </div>
        <div class="stat">
            <div class="stat-value">{stats.get("total_verified", 0)}</div>
            <div class="stat-label">Verified</div>
        </div>
        <div class="stat">
            <div class="stat-value">{stats.get("total_used", 0)}</div>
            <div class="stat-label">Cited</div>
        </div>
        <div class="stat">
            <div class="stat-value">{stats.get("themes", 0)}</div>
            <div class="stat-label">Themes</div>
        </div>
    </div>

</div>

</body>
</html>'''


def render_report(report: HybridReport, template_name: str = DEFAULT_TEMPLATE) -> str:
    """Render a HybridReport object to HTML.

    This is the main entry point for rendering reports from the pipeline.

    Args:
        report: HybridReport object from pipeline_v2
        template_name: Template file to use (default: report.html)

    Returns:
        Rendered HTML string
    """
    template = load_template(template_name)
    data = report_to_dict(report)
    return render_html(data, template)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import json
    import sys

    # Load sample data and render
    sample_path = TEMPLATE_DIR / "sample_report.json"
    if sample_path.exists():
        with open(sample_path) as f:
            data = json.load(f)
        html = render_html(data)

        output_path = Path(__file__).parent.parent.parent / "report_preview.html"
        output_path.write_text(html)
        print(f"Rendered: {output_path}")
    else:
        print(f"Sample data not found: {sample_path}", file=sys.stderr)
        sys.exit(1)
