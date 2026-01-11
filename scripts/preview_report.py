#!/usr/bin/env python3
"""Preview report template with sample data.

Usage:
    python scripts/preview_report.py              # Render once and open
    python scripts/preview_report.py --watch      # Watch for changes and auto-reload
    python scripts/preview_report.py --no-open    # Render without opening browser
"""

import argparse
import json
import os
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).parent.parent
TEMPLATE_PATH = ROOT / "templates" / "report.html"
DATA_PATH = ROOT / "templates" / "sample_report.json"
OUTPUT_PATH = ROOT / "report_preview.html"


def load_data() -> dict:
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def load_template() -> str:
    with open(TEMPLATE_PATH, "r") as f:
        return f.read()


def render_fact(fact: dict) -> str:
    """Render a single fact card."""
    confidence_html = ""
    if fact.get("match_score"):
        confidence_html = f'''
                <span class="source-sep">/</span>
                <span class="confidence">{int(fact["match_score"] * 100)}%</span>'''

    return f'''<div class="fact">
            <p class="fact-text">{fact["extracted_text"]}</p>
            <div class="fact-source">
                <a href="{fact["source_url"]}" target="_blank" rel="noopener">{fact["source_title"]}</a>
                <span class="source-sep">/</span>
                <span>{fact.get("source_domain", "")}</span>{confidence_html}
            </div>
        </div>'''


def render_section(section: dict) -> str:
    """Render a section with its facts."""
    parts = [f'<h3>{section["theme"]}</h3>']

    if section.get("intro"):
        parts.append(f'<p class="synthesis">{section["intro"]}</p>')

    for fact in section["facts"]:
        parts.append(render_fact(fact))

    return "\n".join(parts)


def render_template(template: str, data: dict) -> str:
    """Render template with data."""
    try:
        from jinja2 import Environment, BaseLoader
        env = Environment(loader=BaseLoader())
        env.globals['enumerate'] = enumerate
        env.globals['len'] = len
        tmpl = env.from_string(template)
        return tmpl.render(**data)
    except ImportError:
        return render_manual(template, data)


def render_manual(template: str, data: dict) -> str:
    """Manual template rendering for two-column layout."""
    import re

    # Extract CSS
    css_match = re.search(r'<style>(.*?)</style>', template, re.DOTALL)
    css = css_match.group(1) if css_match else ""

    # Split sections for two columns
    sections = data.get("sections", [])
    mid = (len(sections) + 1) // 2
    left_sections = sections[:mid]
    right_sections = sections[mid:]

    # Render left column
    left_html = ['<h2>Verified Findings</h2>']
    for section in left_sections:
        left_html.append(render_section(section))

    # Render right column
    right_html = ['<h2>&nbsp;</h2>']
    for section in right_sections:
        right_html.append(render_section(section))

    # Render analysis
    analysis_paras = []
    for para in data.get("analysis", "").split('\n\n'):
        if para.strip():
            analysis_paras.append(f'<p>{para.strip()}</p>')

    # Stats
    stats = data.get("stats", {})

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{data["title"]}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>{css}</style>
</head>
<body>

<header>
    <p class="overline">Research Report</p>
    <h1>{data["title"]}</h1>
    <p class="summary">{data["executive_summary"]}</p>
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
            <p>{data["conclusion"]}</p>
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


def render_and_save() -> None:
    data = load_data()
    template = load_template()
    html = render_template(template, data)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"Rendered: {OUTPUT_PATH}")


def get_file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0


def watch_and_render(interval: float = 0.5) -> None:
    print(f"Watching for changes (Ctrl+C to stop)...")
    print(f"  Template: {TEMPLATE_PATH}")
    print(f"  Data: {DATA_PATH}")
    print()

    last_template_mtime = get_file_mtime(TEMPLATE_PATH)
    last_data_mtime = get_file_mtime(DATA_PATH)

    render_and_save()

    try:
        while True:
            time.sleep(interval)
            template_mtime = get_file_mtime(TEMPLATE_PATH)
            data_mtime = get_file_mtime(DATA_PATH)

            if template_mtime != last_template_mtime or data_mtime != last_data_mtime:
                print(f"[{time.strftime('%H:%M:%S')}] Re-rendering...")
                try:
                    render_and_save()
                except Exception as e:
                    print(f"  Error: {e}")
                last_template_mtime = template_mtime
                last_data_mtime = data_mtime
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Preview report template")
    parser.add_argument("--watch", "-w", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.watch:
        if not args.no_open:
            webbrowser.open(f"file://{OUTPUT_PATH.absolute()}")
        watch_and_render()
    else:
        render_and_save()
        if not args.no_open:
            webbrowser.open(f"file://{OUTPUT_PATH.absolute()}")


if __name__ == "__main__":
    main()
