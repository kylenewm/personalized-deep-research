"""Unit tests for pointer extraction module.

Uses mock data only - no API calls.
"""

import pytest
from open_deep_research.pointer_extract import (
    Pointer,
    Extraction,
    find_best_match,
    extract_from_pointer,
    parse_pointer_response,
    format_sources_for_prompt,
    format_extraction_markdown,
)


class TestFindBestMatch:
    """Tests for the fuzzy matching function."""

    def test_exact_keywords_found(self):
        """All keywords present should return high score."""
        content = "The RAND Corporation released a security report in October 2025."
        keywords = ["RAND", "security", "October 2025"]

        text, score = find_best_match(keywords, content)

        assert text is not None
        assert score >= 0.9
        assert "RAND" in text

    def test_partial_keywords_found(self):
        """Some keywords missing should return lower score."""
        content = "The RAND Corporation released a report."
        keywords = ["RAND", "security", "October 2025"]

        text, score = find_best_match(keywords, content, min_score=0.3)

        assert score < 1.0
        assert score > 0.0  # At least RAND found

    def test_no_keywords_found(self):
        """No matching keywords should return None."""
        content = "This is about something completely different."
        keywords = ["RAND", "security", "October 2025"]

        text, score = find_best_match(keywords, content)

        assert text is None
        assert score == 0.0

    def test_empty_keywords(self):
        """Empty keywords list should return None."""
        content = "Some content here."
        keywords = []

        text, score = find_best_match(keywords, content)

        assert text is None
        assert score == 0.0

    def test_empty_content(self):
        """Empty content should return None."""
        content = ""
        keywords = ["RAND", "security"]

        text, score = find_best_match(keywords, content)

        assert text is None
        assert score == 0.0

    def test_case_insensitive(self):
        """Matching should be case insensitive."""
        content = "The RAND corporation released a SECURITY report."
        keywords = ["rand", "Security"]

        text, score = find_best_match(keywords, content)

        assert text is not None
        assert score >= 0.6

    def test_multi_sentence_passage(self):
        """Should find keywords across sentence boundaries if needed."""
        content = "RAND released a report. It covers security topics. Published October 2025."
        keywords = ["RAND", "security", "October 2025"]

        text, score = find_best_match(keywords, content, min_score=0.5)

        # Should find at least partial match
        assert score > 0


class TestExtractFromPointer:
    """Tests for extraction from pointer."""

    @pytest.fixture
    def sample_sources(self):
        return {
            "src_001": {
                "content": "The RAND Corporation released a comprehensive security report in October 2025 recommending multi-layered approaches.",
                "url": "https://rand.org/report",
                "title": "RAND Report"
            },
            "src_002": {
                "content": "OpenAI announced new safety measures including defense-in-depth strategies.",
                "url": "https://openai.com/safety",
                "title": "OpenAI Safety"
            }
        }

    def test_verified_extraction(self, sample_sources):
        """Valid pointer should extract verified text."""
        pointer = Pointer(
            source_id="src_001",
            keywords=["RAND", "security", "October 2025"],
            context="RAND recommendations"
        )

        result = extract_from_pointer(pointer, sample_sources)

        assert result.status == "verified"
        assert result.extracted_text is not None
        assert "RAND" in result.extracted_text
        assert result.match_score >= 0.6

    def test_not_found_wrong_keywords(self, sample_sources):
        """Wrong keywords should return not_found."""
        pointer = Pointer(
            source_id="src_001",
            keywords=["hallucination", "fake", "wrong"],
            context="Should fail"
        )

        result = extract_from_pointer(pointer, sample_sources)

        assert result.status == "not_found"
        assert result.extracted_text is None
        assert result.match_score == 0.0

    def test_not_found_missing_source(self, sample_sources):
        """Missing source_id should return not_found."""
        pointer = Pointer(
            source_id="src_999",
            keywords=["anything"],
            context="Missing source"
        )

        result = extract_from_pointer(pointer, sample_sources)

        assert result.status == "not_found"

    def test_source_url_preserved(self, sample_sources):
        """Extraction should preserve source URL."""
        pointer = Pointer(
            source_id="src_001",
            keywords=["RAND", "security"],
            context="Test"
        )

        result = extract_from_pointer(pointer, sample_sources)

        assert result.source_url == "https://rand.org/report"


class TestParsePointerResponse:
    """Tests for parsing LLM response."""

    def test_valid_json_array(self):
        """Should parse valid JSON array."""
        response = '''[
            {"source_id": "src_001", "keywords": ["RAND", "security"], "context": "Test 1"},
            {"source_id": "src_002", "keywords": ["OpenAI"], "context": "Test 2"}
        ]'''

        pointers = parse_pointer_response(response)

        assert len(pointers) == 2
        assert pointers[0].source_id == "src_001"
        assert pointers[0].keywords == ["RAND", "security"]
        assert pointers[1].context == "Test 2"

    def test_json_with_surrounding_text(self):
        """Should extract JSON from text with preamble."""
        response = '''Here are the pointers:
        [{"source_id": "src_001", "keywords": ["test"], "context": "Found it"}]
        That's all.'''

        pointers = parse_pointer_response(response)

        assert len(pointers) == 1
        assert pointers[0].source_id == "src_001"

    def test_invalid_json(self):
        """Should return empty list for invalid JSON."""
        response = "This is not JSON at all"

        pointers = parse_pointer_response(response)

        assert pointers == []

    def test_empty_array(self):
        """Should handle empty array."""
        response = "[]"

        pointers = parse_pointer_response(response)

        assert pointers == []


class TestFormatSourcesForPrompt:
    """Tests for source formatting."""

    def test_basic_formatting(self):
        sources = {
            "src_001": {
                "content": "Some content here",
                "title": "Test Title"
            }
        }

        formatted = format_sources_for_prompt(sources)

        assert "[src_001]" in formatted
        assert "Test Title" in formatted
        assert "Some content" in formatted

    def test_truncation(self):
        """Long content should be truncated."""
        sources = {
            "src_001": {
                "content": "x" * 5000,
                "title": "Long Content"
            }
        }

        formatted = format_sources_for_prompt(sources, max_chars=100)

        assert len(formatted) < 5000
        assert "..." in formatted


class TestFormatExtractionMarkdown:
    """Tests for markdown output formatting."""

    def test_verified_extraction_output(self):
        extractions = [
            Extraction(
                pointer=Pointer("src_001", ["test"], "Test context"),
                status="verified",
                extracted_text="This is the extracted text.",
                match_score=0.9,
                source_url="https://example.com"
            )
        ]

        output = format_extraction_markdown(extractions, use_color=False)

        assert "This is the extracted text" in output
        assert "https://example.com" in output

    def test_not_found_shows_context(self):
        extractions = [
            Extraction(
                pointer=Pointer("src_001", ["missing"], "Missing content"),
                status="not_found",
                match_score=0.0
            )
        ]

        output = format_extraction_markdown(extractions, use_color=False)

        assert "NOT FOUND" in output
        assert "Missing content" in output

    def test_color_styling(self):
        extractions = [
            Extraction(
                pointer=Pointer("src_001", ["test"], "Test"),
                status="verified",
                extracted_text="Text here",
                match_score=1.0
            )
        ]

        output = format_extraction_markdown(extractions, use_color=True)

        assert "style=" in output  # Has inline styles
        assert "#dcfce7" in output or "green" in output.lower()  # Green for verified
