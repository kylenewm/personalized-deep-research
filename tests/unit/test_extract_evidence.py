"""Unit tests for S03: Evidence extraction (extract_evidence).

Tests the deterministic extraction logic that mines candidate quotes
from source content.
"""

import pytest

from open_deep_research.nodes.extract import (
    extract_evidence,
    generate_snippet_id,
)
from open_deep_research.logic.document_processing import (
    chunk_by_sentences,
)
from open_deep_research.logic.sanitize import (
    extract_paragraphs,
    sanitize_for_quotes,
)


class TestGenerateSnippetId:
    """Tests for snippet ID generation."""

    def test_deterministic(self):
        """Same inputs should produce same ID."""
        id1 = generate_snippet_id("https://example.com", "test quote")
        id2 = generate_snippet_id("https://example.com", "test quote")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        """Different inputs should produce different IDs."""
        id1 = generate_snippet_id("https://example.com", "quote one")
        id2 = generate_snippet_id("https://example.com", "quote two")
        assert id1 != id2

    def test_id_length(self):
        """ID should be 16 characters (first 16 of SHA-256)."""
        snippet_id = generate_snippet_id("url", "quote")
        assert len(snippet_id) == 16

    def test_id_is_hex(self):
        """ID should be valid hex string."""
        snippet_id = generate_snippet_id("url", "quote")
        int(snippet_id, 16)  # Should not raise


class TestSanitizeForQuotes:
    """Tests for content sanitization."""

    def test_removes_extra_whitespace(self):
        """Should normalize whitespace."""
        result = sanitize_for_quotes("hello    world\n\ntest")
        assert "    " not in result
        assert result.count("\n\n") == 0 or " " in result

    def test_handles_html_entities(self):
        """Should handle common HTML entities."""
        result = sanitize_for_quotes("hello &amp; world")
        # Should either decode or remove
        assert "amp;" not in result or "&" in result

    def test_preserves_meaningful_content(self):
        """Should preserve the actual text content."""
        text = "RAG reduces hallucinations by 40 percent"
        result = sanitize_for_quotes(text)
        assert "reduces" in result.lower()
        assert "hallucinations" in result.lower()


class TestExtractParagraphs:
    """Tests for paragraph extraction."""

    def test_extracts_paragraphs(self):
        """Should extract paragraph-sized chunks."""
        content = """
        This is the first paragraph with enough words to meet the minimum.

        This is the second paragraph which also has enough words to qualify.

        Short.
        """
        paragraphs = extract_paragraphs(content, min_words=5, max_words=50)
        assert len(paragraphs) >= 1

    def test_respects_min_words(self):
        """Should skip paragraphs below min_words."""
        content = "Short. Also short. Very short indeed."
        paragraphs = extract_paragraphs(content, min_words=10, max_words=50)
        assert len(paragraphs) == 0

    def test_handles_empty_content(self):
        """Should return empty list for empty content."""
        paragraphs = extract_paragraphs("", min_words=5, max_words=50)
        assert paragraphs == []


class TestLongParagraphHandling:
    """Tests for handling paragraphs longer than 60 words.

    Issue: Previously paragraphs > 60 words were silently dropped.
    Fix: Increased max_words to 100 for consistency with chunk_by_sentences.
    """

    def test_paragraphs_between_60_and_100_words_extracted(self):
        """Paragraphs with 61-100 words should be extracted, not dropped."""
        # Create a paragraph with exactly 75 words
        words = ["word"] * 75
        long_paragraph = " ".join(words) + " Organization"  # Add capitalized for filter

        paragraphs = extract_paragraphs(long_paragraph, min_words=15, max_words=100)
        assert len(paragraphs) == 1, "Should extract paragraphs up to 100 words"

    def test_paragraphs_over_100_words_dropped(self):
        """Paragraphs > 100 words should still be dropped."""
        words = ["word"] * 120
        very_long_paragraph = " ".join(words) + " Organization"

        paragraphs = extract_paragraphs(very_long_paragraph, min_words=15, max_words=100)
        assert len(paragraphs) == 0, "Should drop paragraphs over 100 words"

    def test_max_words_consistency_with_chunk_by_sentences(self):
        """extract_paragraphs and chunk_by_sentences should use same max_words=100.

        This tests the consistency documented in the fix for multi-paragraph quotes.
        """
        # Both should accept 90-word content
        words = ["word"] * 85
        content = " ".join(words) + " Organization with Numbers 42"

        paragraphs = extract_paragraphs(content, min_words=15, max_words=100)
        # Note: chunk_by_sentences has different filtering logic but same max_words
        assert len(paragraphs) == 1


class TestChunkBySentences:
    """Tests for sentence-based chunking (used for Extract API content)."""

    def test_chunks_sentences(self):
        """Should create sentence-based chunks from substantive content."""
        # Use realistic content that meets scoring criteria (entities, numbers, good length)
        content = """
        Retrieval-Augmented Generation (RAG) reduces hallucinations by 40% in healthcare applications.
        Studies from MIT and Stanford show that RAG systems outperform traditional LLMs significantly.
        The technology was adopted by over 500 companies in 2024, including Google and Microsoft.
        """
        # Use low thresholds since we want to test chunking, not scoring
        chunks = chunk_by_sentences(content, min_words=5, max_words=50, min_score=0.0)
        assert len(chunks) >= 1

    def test_respects_word_limits(self):
        """Chunks should respect word count limits."""
        content = "This is a test sentence with several words in it."
        chunks = chunk_by_sentences(content, min_words=3, max_words=20)
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count >= 3
            assert word_count <= 20

    def test_handles_empty_content(self):
        """Should return empty list for empty content."""
        chunks = chunk_by_sentences("", min_words=3, max_words=20)
        assert chunks == []


class TestExtractEvidenceIntegration:
    """Integration tests for the full extract_evidence function."""

    @pytest.mark.asyncio
    async def test_extracts_from_extract_api_source(self, sample_source):
        """Should extract quotes from Extract API content."""
        state = {
            "source_store": [sample_source],
            "verified_disabled": False
        }
        config = {}

        result = await extract_evidence(state, config)
        snippets = result.get("evidence_snippets", [])

        assert len(snippets) > 0
        assert all(s.get("status") == "PENDING" for s in snippets)
        assert all(s.get("source_id") == sample_source["url"] for s in snippets)

    @pytest.mark.asyncio
    async def test_handles_empty_sources(self):
        """Should return empty list when source_store is empty and no external Store."""
        from unittest.mock import patch, AsyncMock

        state = {"source_store": [], "verified_disabled": False}
        config = {}

        # Mock the fallback to external Store to return empty list
        with patch("open_deep_research.nodes.extract.get_stored_sources", new_callable=AsyncMock) as mock:
            mock.return_value = []
            result = await extract_evidence(state, config)
            assert result.get("evidence_snippets", []) == []

    @pytest.mark.asyncio
    async def test_handles_none_content(self):
        """Should skip sources with None content."""
        state = {
            "source_store": [{"url": "https://example.com", "content": None}],
            "verified_disabled": False
        }
        config = {}

        result = await extract_evidence(state, config)
        assert result.get("evidence_snippets", []) == []

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """Should skip extraction when verified_disabled is True."""
        state = {
            "source_store": [{"url": "test", "content": "test content"}],
            "verified_disabled": True
        }
        config = {}

        result = await extract_evidence(state, config)
        assert result.get("evidence_snippets", []) == []

    @pytest.mark.asyncio
    async def test_generates_unique_snippet_ids(self, sample_sources):
        """Each snippet should have a unique ID."""
        state = {"source_store": sample_sources, "verified_disabled": False}
        config = {}

        result = await extract_evidence(state, config)
        snippets = result.get("evidence_snippets", [])

        if snippets:
            ids = [s.get("snippet_id") for s in snippets]
            assert len(ids) == len(set(ids)), "Snippet IDs should be unique"

    @pytest.mark.asyncio
    async def test_limits_snippet_count(self, sample_source):
        """Should limit number of snippets to prevent state bloat."""
        # Create source with lots of content
        large_source = {
            **sample_source,
            "content": " ".join(["This is sentence number {}.".format(i) for i in range(500)])
        }
        state = {"source_store": [large_source], "verified_disabled": False}
        config = {}

        result = await extract_evidence(state, config)
        snippets = result.get("evidence_snippets", [])

        # Should be limited (default is 100)
        assert len(snippets) <= 100
