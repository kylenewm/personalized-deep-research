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

    def test_tokenize_preserves_hyphenated_codes(self):
        """Should preserve hyphenated codes as single tokens."""
        result = tokenize("NIST-SP-800-53 and AEF-1 are standards")
        assert "nist-sp-800-53" in result
        assert "aef-1" in result
        # Should also have individual tokens for backwards compatibility
        assert "nist" in result
        assert "standards" in result

    def test_tokenize_preserves_acronyms_with_periods(self):
        """Should preserve acronyms with periods as single tokens."""
        result = tokenize("The U.S.A. and N.A.S.A. agencies")
        assert "u.s.a." in result
        assert "n.a.s.a." in result
        # Should also have split tokens
        assert "agencies" in result

    def test_tokenize_preserves_dollar_amounts(self):
        """Should preserve dollar amounts as single tokens."""
        result = tokenize("The budget is $1.2B and revenue is $25T")
        assert "$1.2b" in result
        assert "$25t" in result
        assert "budget" in result

    def test_tokenize_different_references_dont_match(self):
        """NIST-SP-800-53 should not have same tokens as NIST SP 800 53."""
        result1 = tokenize("NIST-SP-800-53")
        result2 = tokenize("NIST SP 800 53")

        # result1 should have the hyphenated version
        assert "nist-sp-800-53" in result1

        # result2 should NOT have the hyphenated version
        assert "nist-sp-800-53" not in result2

        # This means Jaccard similarity will be < 1.0 for these
        # (the hyphenated token creates a difference)


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

    def test_quote_longer_than_source_exact_match(self):
        """Quote longer than source should still PASS if source is a substring."""
        # Source is 5 words, quote is 10 words but contains source
        source = "short source content here only"
        quote = "this has short source content here only and more words"
        result = verify_quote_in_source(quote, source)
        # The source appears in the quote, so substring check should find it
        # But we're checking if QUOTE is in SOURCE, not vice versa
        # So this should FAIL (quote not in source)
        assert result == "FAIL"

    def test_quote_longer_than_source_high_overlap(self):
        """Quote longer than source with high word overlap should use fuzzy match."""
        source = "RAG reduces hallucinations effectively"  # 4 words
        quote = "RAG reduces hallucinations effectively in healthcare applications"  # 7 words
        # 4/7 words overlap, Jaccard = 4/(4+3) = 4/7 ≈ 0.57
        result = verify_quote_in_source(quote, source, fuzzy_threshold=0.5)
        assert result == "PASS"

    def test_quote_longer_than_source_low_overlap(self):
        """Quote much longer than source with low overlap should FAIL."""
        source = "short text"  # 2 words
        quote = "this is a completely different long quote with many words"  # 10 words
        result = verify_quote_in_source(quote, source)
        assert result == "FAIL"

    def test_source_contains_quote_words_scrambled(self):
        """Source with same words but scrambled order should use Jaccard."""
        quote = "one two three four five"
        source = "five four three two one"
        # Same words, Jaccard = 1.0
        result = verify_quote_in_source(quote, source, fuzzy_threshold=0.8)
        assert result == "PASS"


class TestVerifyEvidenceDisabled:
    """Tests for verify_evidence when verification is disabled."""

    @pytest.mark.asyncio
    async def test_disabled_marks_snippets_as_skip(self):
        """When verification is disabled, snippets should be marked SKIP not left PENDING."""
        from open_deep_research.nodes.verify import verify_evidence

        state = {
            "verified_disabled": True,
            "evidence_snippets": [
                {"snippet_id": "s1", "quote": "test quote", "status": "PENDING"},
                {"snippet_id": "s2", "quote": "another quote", "status": "PENDING"},
            ]
        }
        config = {}

        result = await verify_evidence(state, config)

        # Should return override with SKIP status
        assert "evidence_snippets" in result
        assert result["evidence_snippets"]["type"] == "override"
        snippets = result["evidence_snippets"]["value"]
        assert len(snippets) == 2
        assert all(s["status"] == "SKIP" for s in snippets)

    @pytest.mark.asyncio
    async def test_disabled_empty_snippets_returns_empty(self):
        """When verification is disabled with no snippets, should return empty dict."""
        from open_deep_research.nodes.verify import verify_evidence

        state = {
            "verified_disabled": True,
            "evidence_snippets": []
        }
        config = {}

        result = await verify_evidence(state, config)
        assert result == {}
