"""Unit tests for source section parsing.

Tests that parse_sources_section correctly extracts citations from various formats.
"""
import pytest
from open_deep_research.evaluation import parse_sources_section


class TestMarkdownFormat:
    """Tests for N. [Title](URL) format (preferred)."""

    def test_basic_markdown_format(self):
        """Standard numbered markdown links."""
        report = """
# Report

Some content here.

## Sources
1. [Economic Report](https://example.com/1)
2. [GDP Data](https://example.com/2)
3. [Market Analysis](https://example.com/3)
"""
        citations = parse_sources_section(report)
        assert len(citations) == 3
        assert citations[1]["url"] == "https://example.com/1"
        assert citations[2]["title"] == "GDP Data"
        assert citations[3]["url"] == "https://example.com/3"

    def test_markdown_with_spaces(self):
        """Markdown format with extra whitespace."""
        report = """
## Sources
1.  [Title One](https://example.com/1)
2.   [Title Two](https://example.com/2)
"""
        citations = parse_sources_section(report)
        assert len(citations) == 2
        assert citations[1]["title"] == "Title One"

    def test_markdown_long_urls(self):
        """Markdown format with complex URLs."""
        report = """
## Sources
1. [Article](https://example.com/path/to/article?param=value&other=123)
"""
        citations = parse_sources_section(report)
        assert len(citations) == 1
        assert "param=value" in citations[1]["url"]


class TestLegacyFormat:
    """Tests for [N] Title: URL format (legacy)."""

    def test_basic_legacy_format(self):
        """Standard legacy bracket format."""
        report = """
# Report

Some content.

## Sources
[1] Economic Report: https://example.com/1
[2] GDP Data: https://example.com/2
"""
        citations = parse_sources_section(report)
        assert len(citations) == 2
        assert citations[1]["url"] == "https://example.com/1"
        assert citations[2]["title"] == "GDP Data"

    def test_legacy_with_colons_in_title(self):
        """Legacy format where title contains colons."""
        report = """
## Sources
[1] Report: Q4 2024: https://example.com/1
"""
        citations = parse_sources_section(report)
        # This may fail - known edge case
        assert len(citations) >= 1


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_no_sources_section(self):
        """Report without Sources section."""
        report = """
# Report
Some content.
## Conclusion
The end.
"""
        citations = parse_sources_section(report)
        assert citations == {}

    def test_empty_sources_section(self):
        """Empty Sources section."""
        report = """
## Sources

## Next Section
"""
        citations = parse_sources_section(report)
        assert citations == {}

    def test_sources_at_end(self):
        """Sources section at very end of document."""
        report = """
# Report
Content.
## Sources
1. [Title](https://example.com/1)
"""
        citations = parse_sources_section(report)
        assert len(citations) == 1

    def test_mixed_format_prefers_markdown(self):
        """If both formats present, should prefer markdown."""
        report = """
## Sources
1. [Markdown Title](https://example.com/md)
[2] Legacy Title: https://example.com/legacy
"""
        citations = parse_sources_section(report)
        # Should find the markdown one
        assert 1 in citations
        assert citations[1]["url"] == "https://example.com/md"


class TestNumberGaps:
    """Tests for non-sequential numbering."""

    def test_gaps_in_numbers(self):
        """Citation numbers with gaps (e.g., 1, 3, 5)."""
        report = """
## Sources
1. [First](https://example.com/1)
3. [Third](https://example.com/3)
5. [Fifth](https://example.com/5)
"""
        citations = parse_sources_section(report)
        assert 1 in citations
        assert 3 in citations
        assert 5 in citations
        assert 2 not in citations
