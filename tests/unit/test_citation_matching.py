"""Unit tests for claim-to-citation matching.

Tests that extract_citations_from_claim correctly finds citations for claims,
including edge cases with short words, acronyms, and numbers.
"""
import pytest
from open_deep_research.evaluation import extract_citations_from_claim


# Sample report with various citation patterns
SAMPLE_REPORT = """
# Research Report

## Introduction

The US economy grew by 2.5% in 2024 [1]. This growth was driven by AI investments
and consumer spending. The GDP reached $25 trillion [2].

## Key Findings

According to NIST SP 800-53, security controls are essential [3]. The framework
recommends implementing MFA for all users [3][4].

IBM reported that data breaches cost an average of $4.45M per incident [5].

## Analysis

The Federal Reserve raised rates to 5.25% [1][6]. This was the highest rate
since 2007. Many economists predict a soft landing [6].

Short claims are important too. AI is key [7].

## Sources
1. [Economic Report](https://example.com/1)
2. [GDP Data](https://example.com/2)
"""


class TestShortWordHandling:
    """Tests for claims with short but important words."""

    def test_acronyms_not_filtered(self):
        """Acronyms like US, AI, GDP should not be filtered out."""
        # This claim has mostly short words
        claim = "The US economy grew by 2.5% in 2024"
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        assert 1 in citations, "Should find citation [1] for US economy claim"

    def test_numbers_not_filtered(self):
        """Numbers like $25T, 2.5% should not be filtered out."""
        claim = "The GDP reached $25 trillion"
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        assert 2 in citations, "Should find citation [2] for GDP claim"

    def test_short_claim_matches(self):
        """Very short claims should still find citations."""
        claim = "AI is key"
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        assert 7 in citations, "Should find citation [7] for short claim"

    def test_acronym_with_numbers(self):
        """Codes like NIST SP 800-53 should match."""
        claim = "NIST SP 800-53 security controls are essential"
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        assert 3 in citations, "Should find citation [3] for NIST claim"


class TestMultipleCitations:
    """Tests for claims with multiple citations."""

    def test_finds_multiple_citations(self):
        """Should find all citations in the matching paragraph."""
        claim = "The framework recommends implementing MFA for all users"
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        assert 3 in citations and 4 in citations, "Should find both [3] and [4]"

    def test_multiple_citations_same_paragraph(self):
        """Claim matching paragraph with multiple citations."""
        claim = "Federal Reserve raised rates to 5.25%"
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        assert 1 in citations or 6 in citations, "Should find at least one citation"


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_no_match_returns_empty(self):
        """Claim not in report should return empty list."""
        claim = "This claim is not in the report at all xyz123"
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        assert citations == [], "Should return empty list for non-matching claim"

    def test_empty_claim(self):
        """Empty claim should return empty list, not crash."""
        citations = extract_citations_from_claim("", SAMPLE_REPORT)
        assert citations == [], "Empty claim should return empty list"

    def test_claim_in_sources_section_ignored(self):
        """Claims matching text in Sources section should not return citations."""
        # The word "Economic Report" appears in Sources section
        claim = "Economic Report from example.com"
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        # Should not match anything since it's only in Sources section
        assert citations == [] or 1 not in citations


class TestThresholdBehavior:
    """Tests for matching threshold behavior."""

    def test_minimum_match_required(self):
        """Should not match with zero words in common."""
        # Claim with only stopwords or very short words
        claim = "a the is of to"  # All filtered out by len > 4
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        # Should NOT match random paragraph just because threshold is 0
        assert citations == [], "Should not match with no meaningful words"

    def test_partial_match_finds_best_paragraph(self):
        """Partial word matches should find the right paragraph."""
        claim = "data breaches cost average incident"
        citations = extract_citations_from_claim(claim, SAMPLE_REPORT)
        assert 5 in citations, "Should find citation [5] for data breach claim"
