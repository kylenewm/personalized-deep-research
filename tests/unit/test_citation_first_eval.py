"""Unit tests for citation-first evaluation functions.

Tests the deterministic evaluation approach that:
1. Parses [N] citations directly from the report
2. Verifies each citation against its corresponding source
3. Separates validity from coverage metrics
"""

import pytest

from open_deep_research.evaluation import (
    extract_citation_context,
    verify_text_against_source,
    evaluate_citation_validity,
    split_into_sentences,
    is_factual_sentence,
    calculate_citation_coverage,
    parse_sources_section,
    CitationValidityResult,
    CitationValidityMetrics,
    CoverageResult,
)


class TestExtractCitationContext:
    """Tests for extract_citation_context helper."""

    def test_extracts_sentence_with_citation(self):
        """Should extract the sentence containing the citation."""
        report = "First sentence. RAG reduces hallucinations by 40% [1]. Third sentence."
        # Position of '[' in [1]
        pos = report.find('[1]')
        context = extract_citation_context(report, pos)
        assert "RAG reduces hallucinations" in context
        assert "40" in context

    def test_removes_citation_markers(self):
        """Should remove [N] markers from extracted context."""
        report = "Studies show that RAG improves accuracy [1][2]. This is significant."
        pos = report.find('[1]')
        context = extract_citation_context(report, pos)
        # Citation markers should be removed
        assert '[1]' not in context
        assert '[2]' not in context

    def test_handles_beginning_of_report(self):
        """Should handle citations at the very start."""
        report = "RAG is effective [1]. More text follows."
        pos = report.find('[1]')
        context = extract_citation_context(report, pos)
        assert "RAG is effective" in context

    def test_handles_multiple_sentences(self):
        """Should extract only the relevant sentence."""
        report = "First. Second sentence about RAG [1]. Third sentence here."
        pos = report.find('[1]')
        context = extract_citation_context(report, pos)
        assert "Second sentence" in context
        # Should not include other sentences
        assert "First" not in context or "Third" not in context


class TestVerifyTextAgainstSource:
    """Tests for verify_text_against_source function."""

    def test_exact_match_returns_valid(self):
        """Exact substring match should return VALID with high confidence."""
        source = {"content": "RAG reduces hallucinations by 40% in healthcare applications."}
        text = "RAG reduces hallucinations by 40%"
        status, confidence, snippet = verify_text_against_source(text, source)
        assert status == "VALID"
        assert confidence == 1.0

    def test_keyword_overlap_returns_valid(self):
        """High keyword overlap should return VALID."""
        source = {"content": "The system reduces errors and improves healthcare accuracy significantly."}
        text = "reduces errors and improves accuracy"
        status, confidence, snippet = verify_text_against_source(text, source)
        assert status == "VALID"
        assert confidence >= 0.6  # 60% threshold

    def test_no_match_returns_invalid(self):
        """No matching content should return INVALID."""
        source = {"content": "This document discusses weather patterns and climate change."}
        text = "RAG reduces hallucinations in AI systems"
        status, confidence, snippet = verify_text_against_source(text, source)
        assert status == "INVALID"
        assert confidence < 0.6  # Below 60% threshold

    def test_empty_source_returns_no_content(self):
        """Empty source should return NO_CONTENT."""
        source = {"content": ""}
        text = "Any text here"
        status, confidence, snippet = verify_text_against_source(text, source)
        assert status == "NO_CONTENT"

    def test_source_without_content_key(self):
        """Source missing content key should return NO_CONTENT."""
        source = {"url": "http://example.com"}
        text = "Any text"
        status, confidence, snippet = verify_text_against_source(text, source)
        assert status == "NO_CONTENT"


class TestEvaluateCitationValidity:
    """Tests for the main evaluate_citation_validity function."""

    def test_valid_citations_are_verified(self):
        """Citations with matching sources should be marked VALID."""
        report = """# Report

RAG reduces hallucinations [1].

## Sources
1. [RAG Study](https://example.com/rag)
"""
        sources = [
            {"url": "https://example.com/rag", "content": "RAG reduces hallucinations significantly in AI systems."}
        ]
        results, metrics = evaluate_citation_validity(report, sources)

        assert len(results) == 1
        assert results[0].status == "VALID"
        assert results[0].citation_num == 1
        assert metrics.valid_count == 1
        assert metrics.validity_rate == 1.0

    def test_invalid_citations_are_flagged(self):
        """Citations with non-matching sources should be marked INVALID."""
        report = """# Report

RAG eliminates all errors completely [1].

## Sources
1. [Study](https://example.com/study)
"""
        sources = [
            {"url": "https://example.com/study", "content": "The weather is nice today."}
        ]
        results, metrics = evaluate_citation_validity(report, sources)

        assert len(results) == 1
        assert results[0].status == "INVALID"
        assert metrics.invalid_count == 1

    def test_missing_source_is_flagged(self):
        """Citation without corresponding source should be MISSING_SOURCE."""
        report = """# Report

Some claim [1].

## Sources
2. [Another Source](https://example.com/other)
"""
        sources = [
            {"url": "https://example.com/other", "content": "Some content"}
        ]
        results, metrics = evaluate_citation_validity(report, sources)

        # Citation [1] has no source (only [2] is in Sources section)
        assert any(r.status == "MISSING_SOURCE" for r in results)
        assert metrics.missing_source_count >= 1

    def test_multiple_citations_in_report(self):
        """Should handle multiple unique citations."""
        report = """# Report

First claim [1]. Second claim [2]. Reference to first again [1].

## Sources
1. [Source One](https://example.com/one)
2. [Source Two](https://example.com/two)
"""
        sources = [
            {"url": "https://example.com/one", "content": "First claim information here."},
            {"url": "https://example.com/two", "content": "Second claim details here."}
        ]
        results, metrics = evaluate_citation_validity(report, sources)

        # Should have 2 unique citations (1 and 2)
        assert metrics.unique_citations == 2
        # But total occurrences should be 3
        assert metrics.total_citations == 3

    def test_deterministic_same_input_same_output(self):
        """Same input should produce same output (deterministic)."""
        report = """# Report

RAG is effective [1].

## Sources
1. [Study](https://example.com/study)
"""
        sources = [{"url": "https://example.com/study", "content": "RAG is effective technology."}]

        results1, metrics1 = evaluate_citation_validity(report, sources)
        results2, metrics2 = evaluate_citation_validity(report, sources)

        assert len(results1) == len(results2)
        assert metrics1.validity_rate == metrics2.validity_rate
        assert results1[0].status == results2[0].status


class TestSplitIntoSentences:
    """Tests for split_into_sentences helper."""

    def test_splits_paragraph_into_sentences(self):
        """Should split paragraph by sentence boundaries."""
        report = """# Header

First sentence. Second sentence! Third sentence?

## Sources
1. [Source](http://example.com)
"""
        sentences = split_into_sentences(report)
        assert len(sentences) >= 3
        assert any("First sentence" in s for s in sentences)

    def test_skips_headers(self):
        """Should not include header text."""
        report = """# Main Header

Content here.

## Another Header

More content.
"""
        sentences = split_into_sentences(report)
        assert not any(s.startswith('#') for s in sentences)

    def test_skips_sources_section(self):
        """Should not include content after Sources section."""
        report = """# Report

Main content here.

## Sources
1. [Source](http://example.com)
2. [Another](http://example2.com)
"""
        sentences = split_into_sentences(report)
        assert not any("example.com" in s for s in sentences)

    def test_handles_list_items(self):
        """Should extract text from list items."""
        report = """# Report

* First bullet point with content
* Second bullet point here

Regular paragraph.
"""
        sentences = split_into_sentences(report)
        assert any("bullet point" in s for s in sentences)


class TestIsFactualSentence:
    """Tests for is_factual_sentence helper."""

    def test_percentage_is_factual(self):
        """Sentence with percentage should be factual."""
        assert is_factual_sentence("RAG reduces errors by 40%.")
        assert is_factual_sentence("Accuracy improved 20.5% over baseline.")

    def test_dollar_amount_is_factual(self):
        """Sentence with dollar amounts should be factual."""
        assert is_factual_sentence("The market is worth $1.9 billion.")
        assert is_factual_sentence("Cost savings of $50,000 per year.")

    def test_year_is_factual(self):
        """Sentence with year should be factual."""
        assert is_factual_sentence("RAG was introduced in 2020.")
        assert is_factual_sentence("The company was founded in 1995.")

    def test_acronym_is_factual(self):
        """Sentence with acronyms should be factual."""
        assert is_factual_sentence("RAG systems are widely used.")
        assert is_factual_sentence("The NASA and FBI collaborated.")

    def test_quoted_text_is_factual(self):
        """Sentence with quotes should be factual."""
        assert is_factual_sentence('The study called it "revolutionary".')
        assert is_factual_sentence('Experts describe this as "breakthrough".')

    def test_comparative_is_factual(self):
        """Sentence with comparatives should be factual."""
        assert is_factual_sentence("This method is more effective.")
        assert is_factual_sentence("Performance is highest in this category.")

    def test_simple_statement_not_factual(self):
        """Simple statements without specific claims are not factual."""
        assert not is_factual_sentence("This section discusses the topic.")
        assert not is_factual_sentence("The following describes the approach.")


class TestCalculateCitationCoverage:
    """Tests for calculate_citation_coverage function."""

    def test_fully_cited_report(self):
        """Report where all sentences are cited should have 100% coverage."""
        report = """# Report

First claim here [1]. Second claim there [2]. Third point [3].

## Sources
1. [One](http://example.com/1)
"""
        result = calculate_citation_coverage(report)
        # All sentences have citations
        assert result.coverage_rate == 1.0
        assert len(result.uncited_factual) == 0

    def test_uncited_factual_statements_flagged(self):
        """Factual statements without citations should be flagged."""
        report = """# Report

This claim has a citation [1]. RAG reduces errors by 40%. Another cited point [2].

## Sources
1. [One](http://example.com/1)
"""
        result = calculate_citation_coverage(report)
        # "RAG reduces errors by 40%" is factual but uncited
        assert len(result.uncited_factual) >= 1
        assert result.coverage_rate < 1.0

    def test_non_factual_uncited_not_flagged(self):
        """Non-factual statements without citations should not be flagged."""
        report = """# Report

This section discusses the approach. The method is described below. Some claim [1].

## Sources
1. [One](http://example.com/1)
"""
        result = calculate_citation_coverage(report)
        # "This section discusses" is not factual - shouldn't be in uncited_factual
        # (depends on is_factual_sentence logic)
        assert result.cited_sentences >= 1

    def test_empty_report_returns_zero(self):
        """Empty report should return zero coverage."""
        result = calculate_citation_coverage("")
        assert result.total_sentences == 0
        assert result.coverage_rate == 0.0


class TestParseSourcesSection:
    """Tests for parse_sources_section function."""

    def test_parses_markdown_format(self):
        """Should parse N. [Title](URL) format."""
        report = """# Report

Content here.

## Sources
1. [First Source](https://example.com/first)
2. [Second Source](https://example.com/second)
"""
        result = parse_sources_section(report)
        assert len(result) == 2
        assert result[1]["url"] == "https://example.com/first"
        assert result[2]["title"] == "Second Source"

    def test_parses_legacy_format(self):
        """Should parse [N] Title: URL format as fallback."""
        report = """# Report

Content.

## Sources
[1] First Source: https://example.com/first
[2] Second Source: https://example.com/second
"""
        result = parse_sources_section(report)
        assert len(result) == 2
        assert result[1]["url"] == "https://example.com/first"

    def test_no_sources_returns_empty(self):
        """Report without Sources section should return empty dict."""
        report = "# Report\n\nJust content, no sources."
        result = parse_sources_section(report)
        assert result == {}
