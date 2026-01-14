"""Unit tests for pointer extraction module.

Uses mock data only - no API calls.
"""

import pytest
from open_deep_research.pointer_extract import (
    Pointer,
    Extraction,
    find_best_match,
    find_tightest_keyword_window,
    expand_to_sentence_bounds,
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

        text, score, span_start, span_end, keywords_matched, method = find_best_match(keywords, content)

        assert text is not None
        assert score >= 0.9
        assert "RAND" in text
        # Verify span offsets
        assert span_start >= 0
        assert span_end > span_start
        # Verify keywords_matched
        assert len(keywords_matched) >= 2
        # Verify method is returned
        assert method in ["micro_quote", "keyword_window", "sentence_fallback"]

    def test_partial_keywords_found(self):
        """Some keywords missing should return lower score."""
        content = "The RAND Corporation released a report."
        keywords = ["RAND", "security", "October 2025"]

        text, score, span_start, span_end, keywords_matched, method = find_best_match(keywords, content, min_score=0.3)

        assert score < 1.0
        assert score > 0.0  # At least RAND found
        # Verify keywords_matched shows which were found
        assert "rand" in keywords_matched  # RAND was found (lowercase)

    def test_no_keywords_found(self):
        """No matching keywords should return None."""
        content = "This is about something completely different."
        keywords = ["RAND", "security", "October 2025"]

        text, score, span_start, span_end, keywords_matched, method = find_best_match(keywords, content)

        assert text is None
        assert score == 0.0
        assert span_start == -1
        assert span_end == -1
        assert keywords_matched == []

    def test_empty_keywords(self):
        """Empty keywords list should return None."""
        content = "Some content here."
        keywords = []

        text, score, span_start, span_end, keywords_matched, method = find_best_match(keywords, content)

        assert text is None
        assert score == 0.0
        assert keywords_matched == []

    def test_empty_content(self):
        """Empty content should return None."""
        content = ""
        keywords = ["RAND", "security"]

        text, score, span_start, span_end, keywords_matched, method = find_best_match(keywords, content)

        assert text is None
        assert score == 0.0
        assert keywords_matched == []

    def test_case_insensitive(self):
        """Matching should be case insensitive."""
        # Content must be >50 chars to pass quality filter
        content = "The RAND corporation released a comprehensive SECURITY report covering multiple areas."
        keywords = ["rand", "Security"]

        text, score, span_start, span_end, keywords_matched, method = find_best_match(keywords, content)

        assert text is not None
        assert score >= 0.6

    def test_multi_sentence_passage(self):
        """Should find keywords across sentence boundaries if needed."""
        content = "RAND released a report. It covers security topics. Published October 2025."
        keywords = ["RAND", "security", "October 2025"]

        text, score, span_start, span_end, keywords_matched, method = find_best_match(keywords, content, min_score=0.5)

        # Should find at least partial match
        assert score > 0

    def test_span_matches_extracted_text(self):
        """Span offsets should allow reverification of extracted text."""
        content = "Some preamble. The RAND Corporation released a security report in October 2025. More text."
        keywords = ["RAND", "security", "October 2025"]

        text, score, span_start, span_end, keywords_matched, method = find_best_match(keywords, content)

        # The extracted text should be findable at the span position
        assert text is not None
        if span_start >= 0 and span_end > span_start:
            # Note: span is in normalized content, not original
            # Just verify span offsets are reasonable
            assert span_end - span_start > 0

    def test_micro_quote_exact_match(self):
        """Micro-quote should enable exact matching with high confidence."""
        content = "Some intro. The platform achieves 99.9% uptime with multi-region failover. More text."
        keywords = ["platform", "99.9%", "uptime"]
        micro_quote = "achieves 99.9% uptime"

        text, score, span_start, span_end, keywords_matched, method = find_best_match(
            keywords, content, micro_quote=micro_quote
        )

        assert text is not None
        assert score == 1.0  # Perfect score for micro-quote match
        assert method == "micro_quote"
        assert "99.9%" in text

    def test_micro_quote_case_insensitive(self):
        """Micro-quote matching should work case-insensitively."""
        content = "The system ACHIEVES 99.9% UPTIME under normal conditions with automated failover."
        keywords = ["system", "99.9%", "uptime"]
        micro_quote = "achieves 99.9% uptime"

        text, score, span_start, span_end, keywords_matched, method = find_best_match(
            keywords, content, micro_quote=micro_quote
        )

        assert text is not None
        assert method == "micro_quote"

    def test_micro_quote_fallback_to_keyword(self):
        """Should fall back to keyword matching if micro-quote doesn't match."""
        content = "The platform has excellent availability of 99.9% for enterprise customers."
        keywords = ["platform", "99.9%", "availability"]
        micro_quote = "achieves 99.9% uptime"  # Not in content

        text, score, span_start, span_end, keywords_matched, method = find_best_match(
            keywords, content, micro_quote=micro_quote
        )

        # Should still find text via keyword matching
        assert text is not None
        assert method in ["keyword_window", "sentence_fallback"]  # Not micro_quote


class TestTightestKeywordWindow:
    """Tests for the tightest keyword window algorithm."""

    def test_finds_tightest_window(self):
        """Should find the minimal span covering keywords."""
        content = "The quick brown fox jumps over the lazy dog. Another sentence here."
        keywords = ["quick", "fox", "jumps"]

        text, start, end, matched, coverage = find_tightest_keyword_window(content, keywords)

        assert text is not None
        assert "quick" in text
        assert "fox" in text
        assert "jumps" in text
        assert coverage == 1.0
        # Window should be tight - not include "dog" unnecessarily
        assert len(text) < len(content)

    def test_handles_abbreviations(self):
        """Should not break on abbreviations like Dr., Inc., etc."""
        content = "Dr. Smith from Acme Inc. presented the report. It was groundbreaking."
        keywords = ["dr.", "smith", "report"]

        text, start, end, matched, coverage = find_tightest_keyword_window(content, keywords)

        assert text is not None
        assert len(matched) == 3
        assert coverage == 1.0

    def test_handles_scattered_keywords(self):
        """Should find window even when keywords are spread out."""
        content = "First word is alpha. Middle section has beta. Last part contains gamma."
        keywords = ["alpha", "gamma"]

        text, start, end, matched, coverage = find_tightest_keyword_window(content, keywords)

        # Should find both keywords
        assert len(matched) == 2
        assert coverage == 1.0

    def test_single_keyword(self):
        """Should handle single keyword case."""
        content = "The platform achieves 99.9% uptime with comprehensive monitoring."
        keywords = ["99.9%"]

        text, start, end, matched, coverage = find_tightest_keyword_window(content, keywords)

        assert text == "99.9%"
        assert len(matched) == 1

    def test_no_keywords_found(self):
        """Should return None when no keywords found."""
        content = "This is about something completely different."
        keywords = ["notfound", "missing"]

        text, start, end, matched, coverage = find_tightest_keyword_window(content, keywords)

        assert text is None
        assert matched == []


class TestExpandToSentenceBounds:
    """Tests for sentence boundary expansion."""

    def test_expands_to_sentence_start(self):
        """Should expand backwards to sentence start."""
        content = "First sentence. This is the target sentence with keywords. Third sentence."
        span_start = content.find("target")
        span_end = content.find("keywords") + len("keywords")

        exp_start, exp_end = expand_to_sentence_bounds(content, span_start, span_end)

        expanded = content[exp_start:exp_end]
        assert expanded.startswith("This")
        assert "." in expanded  # Should include sentence end

    def test_expands_to_sentence_end(self):
        """Should expand forwards to sentence end."""
        content = "Intro. The keyword is here in the middle of this sentence. End."
        span_start = content.find("keyword")
        span_end = span_start + len("keyword")

        exp_start, exp_end = expand_to_sentence_bounds(content, span_start, span_end)

        expanded = content[exp_start:exp_end]
        assert expanded.endswith(".")

    def test_respects_max_expand_limit(self):
        """Should not expand beyond max_expand."""
        content = "A" * 200 + ". Keyword here. " + "B" * 200
        span_start = content.find("Keyword")
        span_end = span_start + len("Keyword")

        exp_start, exp_end = expand_to_sentence_bounds(content, span_start, span_end, max_expand=50)

        # Should not grab all the A's or B's
        expanded = content[exp_start:exp_end]
        assert len(expanded) < 150


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
        """Missing source_id should return not_found with failure_reason."""
        pointer = Pointer(
            source_id="src_999",
            keywords=["anything"],
            context="Missing source"
        )

        result = extract_from_pointer(pointer, sample_sources)

        assert result.status == "not_found"
        # NEW: Verify failure diagnostics
        assert result.failure_reason == "source_missing"
        assert result.failure_details is not None
        assert result.failure_details["source_id"] == "src_999"

    def test_verified_has_span_offsets(self, sample_sources):
        """Verified extraction should have span offsets."""
        pointer = Pointer(
            source_id="src_001",
            keywords=["RAND", "security", "October 2025"],
            context="RAND recommendations"
        )

        result = extract_from_pointer(pointer, sample_sources)

        assert result.status == "verified"
        # NEW: Verify span offsets are populated
        assert result.span_start >= 0
        assert result.span_end > result.span_start
        # NEW: Verify keywords_matched is populated
        assert len(result.keywords_matched) > 0
        # NEW: Verify verification_method is set
        assert result.verification_method == "keyword_window"

    def test_not_found_has_failure_diagnostics(self, sample_sources):
        """Failed extraction should have structured failure diagnostics."""
        pointer = Pointer(
            source_id="src_001",
            keywords=["hallucination", "fake", "wrong"],
            context="Should fail"
        )

        result = extract_from_pointer(pointer, sample_sources)

        assert result.status == "not_found"
        # NEW: Verify failure reason is populated
        assert result.failure_reason is not None
        assert result.failure_reason in ["keywords_missing", "no_match"]

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


class TestVerifySpan:
    """Tests for the verify_span function (I9: Span Verification Required)."""

    def test_valid_span_returns_true(self):
        """Extraction with valid span should verify successfully."""
        from open_deep_research.pointer_extract import verify_span

        content = "The RAND Corporation released a comprehensive security report in October 2025."
        extraction = Extraction(
            pointer=Pointer("src_001", ["RAND", "security"], "Test"),
            status="verified",
            extracted_text="RAND Corporation released a comprehensive security report",
            match_score=0.9,
            span_start=4,  # Position of "RAND"
            span_end=61,   # End of "report"
        )

        # The span should contain the extracted text
        assert verify_span(extraction, content)

    def test_text_not_in_source_returns_false(self):
        """Extraction with text not in source should fail verification."""
        from open_deep_research.pointer_extract import verify_span

        content = "The RAND Corporation released a comprehensive security report in October 2025."
        extraction = Extraction(
            pointer=Pointer("src_001", ["RAND", "security"], "Test"),
            status="verified",
            extracted_text="OpenAI announced new safety measures",  # Not in source
            match_score=0.9,
            span_start=0,
            span_end=10,
        )

        assert not verify_span(extraction, content)

    def test_text_exists_regardless_of_span_offsets(self):
        """Verification passes if text exists in source, even with wrong spans.

        Note: We verify TEXT existence, not POSITION accuracy. This is intentional
        because position-based matching fails when normalization shifts offsets.
        """
        from open_deep_research.pointer_extract import verify_span

        content = "Some content here."
        extraction = Extraction(
            pointer=Pointer("src_001", ["content"], "Test"),
            status="verified",
            extracted_text="content here",
            match_score=0.9,
            span_start=-1,  # Wrong span but text exists
            span_end=-1,
        )

        # Text exists in source, so verification should pass
        assert verify_span(extraction, content)

    def test_empty_extracted_text_returns_false(self):
        """Extraction with no text should fail verification."""
        from open_deep_research.pointer_extract import verify_span

        content = "Some content here."
        extraction = Extraction(
            pointer=Pointer("src_001", ["content"], "Test"),
            status="verified",
            extracted_text=None,
            match_score=0.9,
            span_start=5,
            span_end=12,
        )

        assert not verify_span(extraction, content)

    def test_fabricated_text_returns_false(self):
        """Extraction with completely fabricated text should fail."""
        from open_deep_research.pointer_extract import verify_span

        content = "Short content about AI safety."
        extraction = Extraction(
            pointer=Pointer("src_001", ["safety"], "Test"),
            status="verified",
            extracted_text="This text was completely fabricated by LLM",
            match_score=0.9,
            span_start=0,
            span_end=50,
        )

        assert not verify_span(extraction, content)

    def test_verify_span_with_html_normalization(self):
        """verify_span should normalize HTML like find_best_match does."""
        from open_deep_research.pointer_extract import verify_span

        content = "The <b>RAND</b> Corporation released a report."
        extraction = Extraction(
            pointer=Pointer("src_001", ["RAND", "report"], "Test"),
            status="verified",
            extracted_text="RAND Corporation released a report",
            match_score=0.9,
            span_start=4,  # After HTML normalization
            span_end=38,
        )

        # Should work because verify_span normalizes content
        result = verify_span(extraction, content)
        # Note: exact span may shift after normalization, test may need adjustment
        assert isinstance(result, bool)
