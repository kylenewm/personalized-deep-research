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
    # Build citation info with global IDs (set later in report_to_dict)
    citations = []
    for c in getattr(section, 'citations', []):
        citations.append({
            "marker": c.marker,
            "fact_index": c.fact_index,
            "global_id": getattr(c, 'global_id', 0),
        })

    return {
        "theme": section.theme,
        "prose": getattr(section, 'prose', ''),
        "citations": citations,
        "facts": [fact_to_dict(f) for f in section.facts],
        # Legacy fields for backward compat
        "intro": getattr(section, 'intro', ''),
        "transitions": getattr(section, 'transitions', []),
    }


def report_to_dict(report: HybridReport) -> dict:
    """Convert HybridReport object to template-friendly dict.

    Assigns global footnote IDs across all sections.
    """
    sections = []
    all_footnotes = []  # All facts with global IDs
    global_id = 1

    for section in report.sections:
        section_dict = section_to_dict(section)

        # Track which local fact indices are cited
        cited_indices = set()
        for c in section_dict.get("citations", []):
            cited_indices.add(c["fact_index"])

        # Assign global IDs to facts (cited first, then uncited)
        fact_id_map = {}  # local_index -> global_id

        # First pass: assign IDs to cited facts
        for i, fact in enumerate(section_dict["facts"]):
            if i in cited_indices:
                fact["global_id"] = global_id
                fact["cited"] = True
                fact["theme"] = section_dict["theme"]
                fact_id_map[i] = global_id
                all_footnotes.append(fact)
                global_id += 1

        # Second pass: assign IDs to uncited facts
        for i, fact in enumerate(section_dict["facts"]):
            if i not in cited_indices:
                fact["global_id"] = global_id
                fact["cited"] = False
                fact["theme"] = section_dict["theme"]
                fact_id_map[i] = global_id
                all_footnotes.append(fact)
                global_id += 1

        # Update citations with global IDs
        for c in section_dict.get("citations", []):
            c["global_id"] = fact_id_map.get(c["fact_index"], 0)

        sections.append(section_dict)

    return {
        "title": report.title,
        "executive_summary": report.executive_summary,
        "sections": sections,
        "footnotes": all_footnotes,  # All facts as footnotes
        "analysis": report.analysis,
        "conclusion": report.conclusion,
        "stats": {
            "total_extracted": report.total_extracted,
            "total_verified": report.total_verified,
            "total_used": len([f for f in all_footnotes if f.get("cited")]),
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

    # Use source_title if available, otherwise fall back to domain
    link_text = fact.get("source_title") or fact.get("source_domain") or "Source"
    return f'''<div class="fact">
            <p class="fact-text">{fact["extracted_text"]}</p>
            <div class="fact-source">
                <a href="{fact["source_url"]}" target="_blank" rel="noopener">{link_text}</a>
                <span class="source-sep">/</span>
                <span>{fact.get("source_domain", "")}</span>{confidence_html}
            </div>
        </div>'''


def render_prose_with_citations(prose: str, citations: list) -> str:
    """Convert [ID] markers in prose to superscript citation links.

    Sentences without citations get marked with [u] to indicate unverified.
    """
    if not prose:
        return ""

    # Build a map from marker to global_id
    marker_to_global = {}
    for c in citations:
        marker_to_global[c["marker"]] = c["global_id"]

    # Replace [N] with superscript links
    def replace_citation(match):
        marker = match.group(0)
        global_id = marker_to_global.get(marker, 0)
        if global_id:
            return f'<sup><a href="#fn{global_id}" class="citation">[{global_id}]</a></sup>'
        return marker

    result = re.sub(r'\[\d+\]', replace_citation, prose)

    # Mark sentences without citations as unverified
    # Split by sentence endings, check each for citation links
    sentences = re.split(r'(?<=[.!?])\s+', result)
    marked_sentences = []
    for sent in sentences:
        if sent.strip():
            # If sentence has no citation link, mark as unverified
            if 'class="citation"' not in sent:
                marked_sentences.append(f'<span class="unverified">{sent}</span>')
            else:
                marked_sentences.append(sent)

    return ' '.join(marked_sentences)


def render_section_html(section: dict) -> str:
    """Render a section with prose and citations to HTML."""
    parts = [f'<h3>{section["theme"]}</h3>']

    # New style: prose with inline citations
    if section.get("prose"):
        prose_html = render_prose_with_citations(
            section["prose"],
            section.get("citations", [])
        )
        # Split into paragraphs
        for para in prose_html.split('\n\n'):
            if para.strip():
                parts.append(f'<p class="prose-text">{para.strip()}</p>')
    # Legacy fallback: intro + fact list
    elif section.get("intro"):
        parts.append(f'<p class="synthesis">{section["intro"]}</p>')
        for fact in section.get("facts", []):
            parts.append(render_fact_html(fact))

    return "\n".join(parts)


def render_footnote_html(fact: dict) -> str:
    """Render a single fact as a footnote with cleaner structure."""
    global_id = fact.get("global_id", 0)

    # Use source_title if available, otherwise fall back to domain
    link_text = fact.get("source_title") or fact.get("source_domain") or "Source"
    domain = fact.get("source_domain") or ""
    extracted = fact.get("extracted_text", "")
    url = fact.get("source_url", "")

    # Truncate long extractions for cleaner display
    if len(extracted) > 200:
        extracted = extracted[:200].rsplit(' ', 1)[0] + "..."

    # Build meta line: source title / domain
    meta_html = f'<a href="{url}" target="_blank" rel="noopener" class="fn-source">{link_text}</a>'
    if domain and domain not in link_text:
        meta_html += f' <span class="fn-sep">/</span> <span class="fn-domain">{domain}</span>'

    return f'''<div class="footnote" id="fn{global_id}">
<span class="fn-number">[{global_id}]</span>
<span class="fn-text">{extracted}</span>
<div class="fn-meta">{meta_html}</div>
</div>'''


def render_footnotes_section(footnotes: list) -> str:
    """Render all footnotes grouped by theme. CSS handles two-column layout."""
    if not footnotes:
        return ""

    parts = ['<div class="footnotes-section">']
    parts.append('<h2>Sources & Evidence</h2>')

    # Group by theme
    by_theme = {}
    for fn in footnotes:
        theme = fn.get("theme", "Other")
        if theme not in by_theme:
            by_theme[theme] = []
        by_theme[theme].append(fn)

    for theme, facts in by_theme.items():
        parts.append(f'<h4 class="fn-theme">{theme}</h4>')
        # Sort by global_id for consistent ordering
        sorted_facts = sorted(facts, key=lambda f: f.get("global_id", 0))
        for fact in sorted_facts:
            parts.append(render_footnote_html(fact))

    parts.append('</div>')
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

    # Additional CSS for citations and footnotes
    citation_css = '''
        /* Citation links - true superscript */
        .citation {
            vertical-align: super;
            font-size: 0.7em;
            font-weight: 500;
            color: #3b82f6;
            text-decoration: none;
            margin: 0 1px;
        }

        /* Prose text in sections */
        .prose-text {
            font-size: 1rem;
            line-height: 1.7;
            color: var(--ink);
            margin-bottom: 1rem;
        }

        /* Footnotes section - full width, 3 columns */
        .footnotes-section {
            margin-top: 3rem;
            margin-left: calc(50% - 50vw);
            margin-right: calc(50% - 50vw);
            width: 100vw;
            padding: 2rem max(2rem, calc(50vw - 700px));
            background: var(--paper);
            border-top: 1px solid var(--rule);
            column-count: 3;
            column-gap: 2rem;
        }
        .footnotes-section h2 {
            column-span: all;
            margin-bottom: 1.25rem;
            font-size: 1.125rem;
            font-family: var(--serif);
            font-weight: 600;
            color: var(--ink);
            text-transform: none;
            letter-spacing: normal;
            border-bottom: none;
            padding-bottom: 0;
        }
        .fn-theme {
            font-family: var(--serif);
            font-size: 0.9375rem;
            font-weight: 600;
            color: var(--ink);
            text-transform: none;
            letter-spacing: normal;
            margin: 1.5rem 0 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--rule);
            break-after: avoid;
        }
        .fn-theme:first-of-type {
            margin-top: 0;
        }
        .footnote {
            padding: 0.625rem 0;
            border-bottom: 1px solid var(--rule-light);
            font-size: 0.8125rem;
            line-height: 1.5;
            break-inside: avoid;
        }
        .fn-number {
            font-family: var(--sans);
            font-weight: 600;
            color: var(--verified);
            font-size: 0.75rem;
            margin-right: 0.25rem;
        }
        .fn-text {
            color: var(--ink-light);
            display: inline;
        }
        .fn-meta {
            font-family: var(--sans);
            font-size: 0.6875rem;
            margin-top: 0.375rem;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.125rem 0.375rem;
        }
        .fn-source {
            font-size: 0.6875rem;
            color: var(--verified);
            text-decoration: none;
            font-weight: 500;
        }
        .fn-source:hover {
            text-decoration: underline;
        }
        .fn-domain {
            color: var(--ink-faint);
        }
        .fn-sep {
            color: var(--rule);
        }

        /* Unverified prose - blend in */
        .unverified {
            /* No special styling */
        }

        /* Fix column layout */
        .columns {
            display: grid !important;
            grid-template-columns: calc(50% - 1.25rem) calc(50% - 1.25rem) !important;
            gap: 2.5rem !important;
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid var(--rule);
        }

        .column {
            overflow-wrap: break-word;
            word-wrap: break-word;
            word-break: break-word;
        }

        .prose-text {
            overflow-wrap: break-word !important;
            word-break: break-word !important;
        }

        @media (max-width: 1200px) {
            .footnotes-section {
                column-count: 2;
            }
        }
        @media (max-width: 800px) {
            .footnotes-section {
                column-count: 1;
                padding: 2rem 1.5rem;
            }
        }
    '''

    # Split sections for two columns - balance by content length, not count
    sections = data.get("sections", [])

    # Estimate content length for each section
    def section_length(s):
        prose_len = len(s.get("prose", "") or "")
        facts_len = sum(len(f.get("extracted_text", "")) for f in s.get("facts", []))
        return prose_len + facts_len

    # Greedy balance: add sections to shorter column
    left_sections = []
    right_sections = []
    left_len = 0
    right_len = 0

    for section in sections:
        s_len = section_length(section)
        if left_len <= right_len:
            left_sections.append(section)
            left_len += s_len
        else:
            right_sections.append(section)
            right_len += s_len

    # Render columns
    left_html = ['<h2>Key Findings</h2>']
    for section in left_sections:
        left_html.append(render_section_html(section))

    right_html = ['<h2 style="visibility:hidden">Key Findings</h2>']  # Match height, invisible
    for section in right_sections:
        right_html.append(render_section_html(section))

    # Render analysis paragraphs
    analysis_paras = []
    for para in data.get("analysis", "").split('\n\n'):
        if para.strip():
            analysis_paras.append(f'<p>{para.strip()}</p>')

    # Render footnotes
    footnotes_html = render_footnotes_section(data.get("footnotes", []))

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
    <style>{css}{citation_css}</style>
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

    <!-- Analysis and Conclusion hidden for now - AI content without citations
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
    -->

    {footnotes_html}

</div>

</body>
</html>'''


def render_high_trust(data: dict, template: Optional[str] = None) -> str:
    """Render report in high trust mode - verified facts only, no AI prose.

    Visual design: Cards per theme, bullet points, generous whitespace.
    Uses <details>/<summary> for progressive disclosure.
    """
    if template is None:
        template = load_template()

    # Extract CSS from template
    css_match = re.search(r'<style>(.*?)</style>', template, re.DOTALL)
    css = css_match.group(1) if css_match else ""

    # High trust specific CSS
    high_trust_css = '''
        /* High trust mode - facts only */
        .theme-card {
            background: var(--paper);
            border: 1px solid var(--rule);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .theme-header {
            font-family: var(--sans);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--ink-muted);
            margin: 0 0 1rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--rule-light);
        }
        .fact-item {
            display: flex;
            gap: 0.75rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--rule-light);
        }
        .fact-item:last-child {
            border-bottom: none;
        }
        .fact-bullet {
            color: var(--verified);
            font-size: 0.875rem;
            margin-top: 0.125rem;
        }
        .fact-content {
            flex: 1;
        }
        .fact-content .fact-text {
            font-size: 0.9375rem;
            line-height: 1.6;
            color: var(--ink);
            margin: 0 0 0.375rem 0;
        }
        .fact-content .fact-source {
            font-family: var(--sans);
            font-size: 0.75rem;
            color: var(--verified);
            text-decoration: none;
        }
        .fact-content .fact-source:hover {
            text-decoration: underline;
        }
        details.more-facts {
            margin-top: 0.5rem;
        }
        details.more-facts summary {
            font-family: var(--sans);
            font-size: 0.8rem;
            color: var(--accent);
            cursor: pointer;
            padding: 0.5rem 0;
        }
        details.more-facts summary:hover {
            text-decoration: underline;
        }
        .high-trust-notice {
            font-family: var(--sans);
            font-size: 0.75rem;
            color: var(--verified);
            background: var(--verified-bg);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin-bottom: 1.5rem;
        }
    '''

    # Build theme cards
    sections = data.get("sections", [])
    theme_cards = []

    for section in sections:
        facts = section.get("facts", [])
        facts_html = []
        visible_count = 5  # Show first 5

        for i, fact in enumerate(facts):
            link_text = fact.get("source_title") or fact.get("source_domain") or "Source"
            fact_html = f'''<div class="fact-item">
                <span class="fact-bullet">▸</span>
                <div class="fact-content">
                    <p class="fact-text">{fact.get("extracted_text", "")}</p>
                    <a href="{fact.get("source_url", "")}" target="_blank" rel="noopener" class="fact-source">{link_text}</a>
                </div>
            </div>'''

            if i < visible_count:
                facts_html.append(fact_html)
            elif i == visible_count:
                # Start the details block
                facts_html.append(f'<details class="more-facts"><summary>Show {len(facts) - visible_count} more findings</summary>')
                facts_html.append(fact_html)
            else:
                facts_html.append(fact_html)

        # Close details if we opened it
        if len(facts) > visible_count:
            facts_html.append('</details>')

        card = f'''<div class="theme-card">
            <h3 class="theme-header">{section.get("theme", "Findings")}</h3>
            <div class="facts-list">
                {chr(10).join(facts_html)}
            </div>
        </div>'''
        theme_cards.append(card)

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
    <style>{css}{high_trust_css}</style>
</head>
<body>

<header>
    <p class="overline">Research Report</p>
    <h1>{data.get("title", "")}</h1>
</header>

<div class="container" style="max-width: 800px; margin: 0 auto; padding: 0 1rem;">

    <div class="high-trust-notice">
        ✓ High Trust Mode: All content below is verified from sources. No AI-generated prose.
    </div>

    {chr(10).join(theme_cards)}

</div>

</body>
</html>'''


def render_report(report: HybridReport, template_name: str = DEFAULT_TEMPLATE, trust_level: str = "med") -> str:
    """Render a HybridReport object to HTML.

    This is the main entry point for rendering reports from the pipeline.

    Args:
        report: HybridReport object from pipeline_v2
        template_name: Template file to use (default: report.html)
        trust_level: "high" (facts only) or "med" (prose with citations)

    Returns:
        Rendered HTML string
    """
    template = load_template(template_name)
    data = report_to_dict(report)

    if trust_level == "high":
        return render_high_trust(data, template)
    else:
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
