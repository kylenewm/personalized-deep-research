"""Unit tests for S04: Evidence verification (verify_evidence).

Tests the deterministic verification logic that checks if quotes
actually exist in source content.
"""

import pytest

from open_deep_research.nodes.verify import (
    verify_quote_in_source,
    jaccard_similarity,
    tokenize,
)


class TestTokenize:
    """Tests for the tokenize helper function."""

    def test_tokenize_basic(self):
        """Should split text into lowercase word tokens."""
        result = tokenize("Hello World")
        assert result == {"hello", "world"}

    def test_tokenize_with_punctuation(self):
        """Should handle punctuation correctly."""
        result = tokenize("Hello, World! How are you?")
        assert result == {"hello", "world", "how", "are", "you"}

    def test_tokenize_empty(self):
        """Should return empty set for empty string."""
        result = tokenize("")
        assert result == set()

    def test_tokenize_numbers(self):
        """Should include numbers as tokens."""
        result = tokenize("RAG reduces errors by 40%")
        assert "40" in result
        assert "rag" in result


class TestJaccardSimilarity:
    """Tests for Jaccard similarity calculation."""

    def test_identical_texts(self):
        """Identical texts should have similarity of 1.0."""
        result = jaccard_similarity("hello world", "hello world")
        assert result == 1.0

    def test_completely_different(self):
        """Completely different texts should have similarity of 0.0."""
        result = jaccard_similarity("hello world", "foo bar baz")
        assert result == 0.0

    def test_partial_overlap(self):
        """Partially overlapping texts should have intermediate similarity."""
        result = jaccard_similarity("hello world foo", "hello world bar")
        # Intersection: {hello, world}, Union: {hello, world, foo, bar}
        # Expected: 2/4 = 0.5
        assert result == 0.5

    def test_empty_texts(self):
        """Empty texts should return 0.0."""
        assert jaccard_similarity("", "") == 0.0
        assert jaccard_similarity("hello", "") == 0.0
        assert jaccard_similarity("", "hello") == 0.0


class TestVerifyQuoteInSource:
    """Tests for the main verification function."""

    def test_exact_match_passes(self):
        """Quote that exists verbatim in source should PASS."""
        quote = "RAG reduces hallucinations"
        source = "Studies show RAG reduces hallucinations significantly."
        result = verify_quote_in_source(quote, source)
        assert result == "PASS"

    def test_missing_quote_fails(self):
        """Quote not in source should FAIL."""
        quote = "RAG eliminates all errors completely"
        source = "RAG helps reduce some errors in certain cases."
        result = verify_quote_in_source(quote, source)
        assert result == "FAIL"

    def test_fuzzy_match_passes(self):
        """Quote with minor differences should PASS via fuzzy matching."""
        quote = "reduces hallucinations by forty percent"
        source = "RAG reduces hallucinations by 40 percent in healthcare."
        # Should pass via Jaccard similarity
        result = verify_quote_in_source(quote, source, fuzzy_threshold=0.6)
        assert result == "PASS"

    def test_case_insensitive(self):
        """Matching should be case-insensitive after sanitization."""
        quote = "RAG REDUCES HALLUCINATIONS"
        source = "rag reduces hallucinations in many applications"
        result = verify_quote_in_source(quote, source)
        assert result == "PASS"

    def test_empty_quote_fails(self):
        """Empty quote should FAIL."""
        result = verify_quote_in_source("", "Some source content")
        assert result == "FAIL"

    def test_empty_source_fails(self):
        """Empty source should FAIL."""
        result = verify_quote_in_source("Some quote", "")
        assert result == "FAIL"

    def test_none_inputs_fail(self):
        """None inputs should FAIL."""
        assert verify_quote_in_source(None, "content") == "FAIL"
        assert verify_quote_in_source("quote", None) == "FAIL"

    def test_partial_quote_passes(self):
        """Partial substring match should PASS."""
        quote = "by 40%"
        source = "RAG reduces hallucinations by 40% in healthcare applications."
        result = verify_quote_in_source(quote, source)
        assert result == "PASS"

    def test_reordered_words_uses_fuzzy(self):
        """Words in different order should use fuzzy matching."""
        quote = "hallucinations reduces RAG"
        source = "RAG reduces hallucinations effectively."
        # Same words, different order - Jaccard should be high
        result = verify_quote_in_source(quote, source, fuzzy_threshold=0.7)
        assert result == "PASS"


class TestVerifyQuoteEdgeCases:
    """Edge case tests for verification."""

    def test_unicode_content(self):
        """Should handle unicode characters."""
        quote = "émoji and café"
        source = "This contains émoji and café references."
        result = verify_quote_in_source(quote, source)
        assert result == "PASS"

    def test_very_long_quote(self):
        """Should handle very long quotes."""
        quote = "This is a very long quote " * 50
        source = "This is a very long quote " * 50 + " with extra content."
        result = verify_quote_in_source(quote, source)
        assert result == "PASS"

    def test_special_characters(self):
        """Should handle special characters in quotes."""
        quote = "accuracy of 99.5% (±0.3)"
        source = "The model achieved accuracy of 99.5% (±0.3) on the test set."
        result = verify_quote_in_source(quote, source)
        assert result == "PASS"

    def test_newlines_in_source(self):
        """Should handle newlines in source content."""
        quote = "spans multiple lines"
        source = "This content\nspans multiple\nlines in the document."
        result = verify_quote_in_source(quote, source)
        # After sanitization, newlines become spaces
        assert result == "PASS"
