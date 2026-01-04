#!/usr/bin/env python3
"""Test Tavily Extract API output quality across different source types.

This script validates:
1. Content cleanliness (no HTML artifacts)
2. Completeness (full article vs truncated)
3. Structure preservation (paragraphs, lists, headers)
4. Edge case handling (PDFs, GitHub, docs sites)

Run with:
    source venv/bin/activate
    python scripts/test_tavily_extract.py
"""

import os
import sys
from typing import Optional

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tavily import TavilyClient


# Test URLs representing different content types
TEST_URLS = [
    # Wikipedia - well-structured, clean HTML
    "https://en.wikipedia.org/wiki/Large_language_model",
    # News site - dynamic content, ads
    "https://www.reuters.com/technology/",
    # Academic - PDF abstract page
    "https://arxiv.org/abs/2303.08774",
    # GitHub README - markdown rendered
    "https://github.com/langchain-ai/langgraph",
    # Documentation - nested structure
    "https://docs.python.org/3/library/asyncio.html",
]


def analyze_content(content: str) -> dict:
    """Analyze content quality metrics."""
    lines = content.split('\n')
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    # Check for HTML artifacts
    html_indicators = ['<', '>', '&nbsp;', '&amp;', '&lt;', '&gt;', 'class=', 'style=']
    html_artifacts = sum(1 for indicator in html_indicators if indicator in content)

    # Check for structure preservation
    has_headers = any(line.startswith('#') or line.isupper() and len(line) < 100 for line in lines[:20])
    has_lists = any(line.strip().startswith(('-', '*', '•', '1.', '2.')) for line in lines)

    # Word/char counts
    words = content.split()

    return {
        'char_count': len(content),
        'word_count': len(words),
        'line_count': len(lines),
        'paragraph_count': len(paragraphs),
        'html_artifacts': html_artifacts,
        'has_structure': has_headers or has_lists,
        'avg_paragraph_words': sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1),
    }


def test_extract(api_key: Optional[str] = None):
    """Test Tavily Extract API on various URL types."""

    api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("ERROR: TAVILY_API_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    client = TavilyClient(api_key=api_key)

    print("=" * 80)
    print("TAVILY EXTRACT API TEST")
    print("=" * 80)

    results_summary = []

    for url in TEST_URLS:
        print(f"\n{'─' * 80}")
        print(f"URL: {url}")
        print('─' * 80)

        try:
            # Call Extract API
            result = client.extract(urls=[url])

            if not result.get("results"):
                print("  ⚠️  No results returned")
                results_summary.append({
                    'url': url,
                    'status': 'NO_RESULTS',
                    'error': None
                })
                continue

            for item in result["results"]:
                # Get content - check both possible field names
                content = item.get("raw_content") or item.get("content") or ""
                title = item.get("title", "N/A")

                # Analyze content quality
                metrics = analyze_content(content)

                print(f"  Title: {title}")
                print(f"  Content length: {metrics['char_count']:,} chars, {metrics['word_count']:,} words")
                print(f"  Structure: {metrics['paragraph_count']} paragraphs, {metrics['line_count']} lines")
                print(f"  Avg paragraph: {metrics['avg_paragraph_words']:.1f} words")
                print(f"  HTML artifacts: {metrics['html_artifacts']} {'⚠️' if metrics['html_artifacts'] > 0 else '✓'}")
                print(f"  Has structure: {'✓' if metrics['has_structure'] else '✗'}")

                # Show sample content
                print(f"\n  FIRST 400 CHARS:")
                print(f"  {'-' * 40}")
                preview = content[:400].replace('\n', '\n  ')
                print(f"  {preview}...")

                print(f"\n  LAST 200 CHARS:")
                print(f"  {'-' * 40}")
                preview = content[-200:].replace('\n', '\n  ')
                print(f"  ...{preview}")

                results_summary.append({
                    'url': url,
                    'status': 'SUCCESS',
                    'metrics': metrics,
                    'error': None
                })

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results_summary.append({
                'url': url,
                'status': 'ERROR',
                'error': str(e)
            })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
    print(f"\nSuccess rate: {success_count}/{len(results_summary)}")

    if success_count > 0:
        avg_chars = sum(r['metrics']['char_count'] for r in results_summary if r['status'] == 'SUCCESS') / success_count
        avg_artifacts = sum(r['metrics']['html_artifacts'] for r in results_summary if r['status'] == 'SUCCESS') / success_count
        print(f"Avg content length: {avg_chars:,.0f} chars")
        print(f"Avg HTML artifacts: {avg_artifacts:.1f}")

    print("\nPer-URL status:")
    for r in results_summary:
        status_emoji = '✓' if r['status'] == 'SUCCESS' else '✗'
        url_short = r['url'][:60] + '...' if len(r['url']) > 60 else r['url']
        if r['status'] == 'SUCCESS':
            print(f"  {status_emoji} {url_short} ({r['metrics']['word_count']:,} words)")
        else:
            print(f"  {status_emoji} {url_short} ({r['status']}: {r.get('error', 'N/A')[:50]})")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Determine if Extract API is good enough
    clean_count = sum(1 for r in results_summary
                      if r['status'] == 'SUCCESS' and r['metrics']['html_artifacts'] == 0)

    if clean_count == success_count and success_count > 0:
        print("✓ Content is CLEAN - can skip HTML sanitization for Extract API content")
    elif clean_count > success_count / 2:
        print("⚠️ Content is MOSTLY clean - consider light sanitization")
    else:
        print("✗ Content has artifacts - need full sanitization pipeline")

    structured_count = sum(1 for r in results_summary
                          if r['status'] == 'SUCCESS' and r['metrics']['has_structure'])
    if structured_count > success_count / 2:
        print("✓ Structure preserved - spacy sentence chunking should work well")
    else:
        print("⚠️ Structure lost - may need fallback to paragraph splitting")


if __name__ == "__main__":
    test_extract()
