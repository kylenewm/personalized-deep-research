#!/usr/bin/env python3
"""Sandbox for testing source quality guidance implementation.

Tests:
1. Per-domain limit in tavily_search
2. Quality guidance injection for trust_level=high
3. No quality guidance for trust_level=med
"""

import sys
sys.path.insert(0, 'src')

from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    research_system_prompt,
    lead_researcher_prompt,
    SOURCE_QUALITY_GUIDANCE_RESEARCHER,
    SOURCE_QUALITY_GUIDANCE_SUPERVISOR,
)
from open_deep_research.pipeline_v2 import (
    ARRANGER_PROMPT,
    ARRANGER_QUALITY_GUIDANCE,
    THEME_SYNTHESIS_PROMPT,
    SYNTHESIS_QUALITY_GUIDANCE,
)
from open_deep_research.utils import extract_domain


def test_per_domain_limit():
    """Test that per-domain limit config exists and has correct default."""
    print("\n=== Testing Per-Domain Limit ===")

    config = Configuration()

    # Check config field exists
    assert hasattr(config, 'max_sources_per_domain'), "Missing max_sources_per_domain config"
    assert config.max_sources_per_domain == 3, f"Expected default 3, got {config.max_sources_per_domain}"

    print(f"  max_sources_per_domain = {config.max_sources_per_domain}")
    print("  PASS: Per-domain limit config exists with correct default")


def test_extract_domain():
    """Test domain extraction utility."""
    print("\n=== Testing Domain Extraction ===")

    test_cases = [
        ("https://www.example.com/page", "example.com"),
        ("https://docs.github.com/en/actions", "docs.github.com"),
        ("http://www.google.com", "google.com"),
        ("https://arxiv.org/abs/1234", "arxiv.org"),
    ]

    for url, expected in test_cases:
        result = extract_domain(url)
        assert result == expected, f"Expected '{expected}', got '{result}' for URL: {url}"
        print(f"  {url} -> {result}")

    print("  PASS: Domain extraction works correctly")


def test_trust_level_config():
    """Test that trust_level config exists."""
    print("\n=== Testing Trust Level Config ===")

    config = Configuration()

    assert hasattr(config, 'trust_level'), "Missing trust_level config"
    assert config.trust_level == "med", f"Expected default 'med', got '{config.trust_level}'"

    print(f"  trust_level = {config.trust_level}")
    print("  PASS: Trust level config exists with correct default")


def test_researcher_prompt_placeholder():
    """Test that research_system_prompt has quality guidance placeholder."""
    print("\n=== Testing Researcher Prompt Placeholder ===")

    assert "{source_quality_guidance}" in research_system_prompt, \
        "Missing {source_quality_guidance} placeholder in research_system_prompt"

    # Test formatting with high trust
    formatted_high = research_system_prompt.format(
        mcp_prompt="",
        date="2026-01-11",
        source_quality_guidance=SOURCE_QUALITY_GUIDANCE_RESEARCHER
    )
    assert "Prefer authoritative sources" in formatted_high, \
        "Quality guidance not injected for high trust"

    # Test formatting with med trust (empty)
    formatted_med = research_system_prompt.format(
        mcp_prompt="",
        date="2026-01-11",
        source_quality_guidance=""
    )
    assert "Prefer authoritative sources" not in formatted_med, \
        "Quality guidance should not appear for med trust"

    print("  PASS: Researcher prompt placeholder works correctly")


def test_supervisor_prompt_placeholder():
    """Test that lead_researcher_prompt has quality guidance placeholder."""
    print("\n=== Testing Supervisor Prompt Placeholder ===")

    assert "{source_quality_guidance}" in lead_researcher_prompt, \
        "Missing {source_quality_guidance} placeholder in lead_researcher_prompt"

    # Test formatting with high trust
    formatted_high = lead_researcher_prompt.format(
        date="2026-01-11",
        max_concurrent_research_units=5,
        max_researcher_iterations=6,
        source_quality_guidance=SOURCE_QUALITY_GUIDANCE_SUPERVISOR
    )
    assert "Source Quality Check" in formatted_high, \
        "Quality guidance not injected for high trust"

    # Test formatting with med trust (empty)
    formatted_med = lead_researcher_prompt.format(
        date="2026-01-11",
        max_concurrent_research_units=5,
        max_researcher_iterations=6,
        source_quality_guidance=""
    )
    assert "Source Quality Check" not in formatted_med, \
        "Quality guidance should not appear for med trust"

    print("  PASS: Supervisor prompt placeholder works correctly")


def test_arranger_prompt_placeholder():
    """Test that ARRANGER_PROMPT has quality guidance placeholder."""
    print("\n=== Testing Arranger Prompt Placeholder ===")

    assert "{source_quality_guidance}" in ARRANGER_PROMPT, \
        "Missing {source_quality_guidance} placeholder in ARRANGER_PROMPT"

    # Test formatting with high trust
    formatted_high = ARRANGER_PROMPT.format(
        topic="test topic",
        num_facts=10,
        facts="test facts",
        source_quality_guidance=ARRANGER_QUALITY_GUIDANCE
    )
    assert "SOURCE QUALITY PREFERENCE" in formatted_high, \
        "Quality guidance not injected for high trust"

    # Test formatting with med trust (empty)
    formatted_med = ARRANGER_PROMPT.format(
        topic="test topic",
        num_facts=10,
        facts="test facts",
        source_quality_guidance=""
    )
    assert "SOURCE QUALITY PREFERENCE" not in formatted_med, \
        "Quality guidance should not appear for med trust"

    print("  PASS: Arranger prompt placeholder works correctly")


def test_synthesis_prompt_placeholder():
    """Test that THEME_SYNTHESIS_PROMPT has quality guidance placeholder."""
    print("\n=== Testing Synthesis Prompt Placeholder ===")

    assert "{source_quality_guidance}" in THEME_SYNTHESIS_PROMPT, \
        "Missing {source_quality_guidance} placeholder in THEME_SYNTHESIS_PROMPT"

    # Test formatting with high trust
    formatted_high = THEME_SYNTHESIS_PROMPT.format(
        theme="test theme",
        topic="test topic",
        facts="test facts",
        source_quality_guidance=SYNTHESIS_QUALITY_GUIDANCE
    )
    assert "authoritative source first" in formatted_high, \
        "Quality guidance not injected for high trust"

    # Test formatting with med trust (empty)
    formatted_med = THEME_SYNTHESIS_PROMPT.format(
        theme="test theme",
        topic="test topic",
        facts="test facts",
        source_quality_guidance=""
    )
    assert "authoritative source first" not in formatted_med, \
        "Quality guidance should not appear for med trust"

    print("  PASS: Synthesis prompt placeholder works correctly")


def main():
    """Run all quality sandbox tests."""
    print("=" * 60)
    print("Source Quality Guidance Sandbox Tests")
    print("=" * 60)

    try:
        test_per_domain_limit()
        test_extract_domain()
        test_trust_level_config()
        test_researcher_prompt_placeholder()
        test_supervisor_prompt_placeholder()
        test_arranger_prompt_placeholder()
        test_synthesis_prompt_placeholder()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n  FAIL: {e}")
        return 1
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
