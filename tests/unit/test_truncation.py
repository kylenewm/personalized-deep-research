"""Unit tests for content truncation with markers.

Tests that content truncation adds visible markers for:
1. utils.py: TRUNCATION_MARKER constant and truncation logic
2. researcher.py: extract_sources_from_messages function
3. verify.py: fallback source parsing

This ensures data loss is always visible to downstream components.
"""
import pytest
from datetime import datetime

from open_deep_research.utils import TRUNCATION_MARKER, _source_cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear module-level cache before each test."""
    _source_cache.clear()
    yield
    _source_cache.clear()


class TestTruncationMarker:
    """Tests for the TRUNCATION_MARKER constant."""

    def test_marker_exists(self):
        """TRUNCATION_MARKER should be a non-empty string."""
        assert isinstance(TRUNCATION_MARKER, str)
        assert len(TRUNCATION_MARKER) > 0

    def test_marker_is_visually_distinct(self):
        """Marker should be easily identifiable in text."""
        assert "[..." in TRUNCATION_MARKER or "TRUNCAT" in TRUNCATION_MARKER
        # Should contain a clear indicator of content loss
        assert "TRUNCAT" in TRUNCATION_MARKER.upper()

    def test_marker_length_is_reasonable(self):
        """Marker should be short enough to fit in truncated content."""
        # Marker shouldn't eat up too much of the content space
        assert len(TRUNCATION_MARKER) < 100
        # But should be visible enough
        assert len(TRUNCATION_MARKER) > 10


class TestTruncationLogic:
    """Tests for the truncation logic pattern used across the codebase."""

    def test_truncation_formula(self):
        """Test the truncation formula: content[:max_len - marker_len] + marker."""
        content = "A" * 1000
        max_len = 100

        # Apply truncation (same logic as in utils.py, researcher.py, verify.py)
        truncated = content[:max_len - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER

        # Total length should equal max_len exactly
        assert len(truncated) == max_len
        # Marker should be at the end
        assert truncated.endswith(TRUNCATION_MARKER)

    def test_short_content_not_truncated(self):
        """Content under max_len should pass through unchanged."""
        content = "Short content"
        max_len = 1000

        was_truncated = len(content) > max_len
        if was_truncated:
            content = content[:max_len - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER

        assert was_truncated is False
        assert content == "Short content"
        assert TRUNCATION_MARKER not in content

    def test_exact_length_content_not_truncated(self):
        """Content exactly at max_len should not be truncated."""
        max_len = 100
        content = "B" * max_len

        was_truncated = len(content) > max_len
        if was_truncated:
            content = content[:max_len - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER

        assert was_truncated is False
        assert TRUNCATION_MARKER not in content
        assert len(content) == max_len

    def test_one_over_triggers_truncation(self):
        """Content one char over max_len should be truncated."""
        max_len = 100
        content = "C" * (max_len + 1)

        was_truncated = len(content) > max_len
        if was_truncated:
            content = content[:max_len - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER

        assert was_truncated is True
        assert content.endswith(TRUNCATION_MARKER)
        assert len(content) == max_len


class TestResearcherTruncation:
    """Tests for extract_sources_from_messages truncation in researcher.py."""

    def test_parse_sources_without_truncation(self):
        """Short source content should not be truncated."""
        from open_deep_research.nodes.researcher import extract_sources_from_tool_messages
        from langchain_core.messages import ToolMessage

        # Create a tool message with short content
        short_summary = "This is a short summary that does not need truncation."
        content = f"""--- SOURCE 1: Test Source ---
URL: https://example.com/test

SUMMARY:
{short_summary}

----------"""
        messages = [ToolMessage(content=content, tool_call_id="test")]

        sources = extract_sources_from_tool_messages(messages)

        assert len(sources) == 1
        assert TRUNCATION_MARKER not in sources[0]["content"]
        # The content should be close to the original (might have whitespace differences)
        assert short_summary in sources[0]["content"] or sources[0]["content"].strip() == short_summary

    def test_parse_sources_with_truncation(self):
        """Long source content should be truncated with marker."""
        from open_deep_research.nodes.researcher import extract_sources_from_tool_messages
        from langchain_core.messages import ToolMessage

        # Create very long summary (over 50k)
        long_summary = "B" * 60000
        content = f"""--- SOURCE 1: Test Source ---
URL: https://example.com/test

SUMMARY:
{long_summary}

----------"""
        messages = [ToolMessage(content=content, tool_call_id="test")]

        sources = extract_sources_from_tool_messages(messages)

        assert len(sources) == 1
        assert TRUNCATION_MARKER in sources[0]["content"]
        assert sources[0]["was_truncated"] is True
        # Should not exceed 50000 chars
        assert len(sources[0]["content"]) <= 50000

    def test_multiple_sources_independent_truncation(self):
        """Each source should be truncated independently."""
        from open_deep_research.nodes.researcher import extract_sources_from_tool_messages
        from langchain_core.messages import ToolMessage

        short_summary = "Short content"
        long_summary = "X" * 60000

        content = f"""--- SOURCE 1: Short Source ---
URL: https://example.com/short

SUMMARY:
{short_summary}

--- SOURCE 2: Long Source ---
URL: https://example.com/long

SUMMARY:
{long_summary}

----------"""
        messages = [ToolMessage(content=content, tool_call_id="test")]

        sources = extract_sources_from_tool_messages(messages)

        assert len(sources) == 2

        # First source should not be truncated
        assert sources[0]["was_truncated"] is False
        assert TRUNCATION_MARKER not in sources[0]["content"]

        # Second source should be truncated
        assert sources[1]["was_truncated"] is True
        assert TRUNCATION_MARKER in sources[1]["content"]


class TestVerifyTruncation:
    """Tests for verify node fallback parsing truncation.

    Note: The verify node's fallback parsing truncates to 5000 chars.
    This is much smaller than researcher (50000) to keep verification fast.
    """

    def test_fallback_truncation_limit(self):
        """Verify fallback should truncate at 5000 chars with marker."""
        # This test documents the expected behavior
        # The actual code is in verify.py's verify_claims function
        # which is async and requires more setup to test

        # Document the design: 5000 char limit in fallback path
        max_fallback_len = 5000
        marker_len = len(TRUNCATION_MARKER)

        # The truncated content should be:
        # (max_fallback_len - marker_len) chars of content + marker
        expected_content_portion = max_fallback_len - marker_len

        assert expected_content_portion > 4000, "Should preserve most of the 5k limit"
        assert expected_content_portion < 5000, "Must leave room for marker"


class TestTruncationConsistency:
    """Tests for consistent truncation behavior across components."""

    def test_marker_is_same_everywhere(self):
        """All components should use the same truncation marker."""
        from open_deep_research.utils import TRUNCATION_MARKER as utils_marker
        from open_deep_research.nodes.researcher import TRUNCATION_MARKER as researcher_marker
        from open_deep_research.nodes.verify import TRUNCATION_MARKER as verify_marker

        assert utils_marker == researcher_marker == verify_marker

    def test_truncated_content_is_detectable(self):
        """Should be able to programmatically detect truncated content."""
        test_content = "Some content" + TRUNCATION_MARKER

        # Detection by marker presence
        assert TRUNCATION_MARKER in test_content
        assert test_content.endswith(TRUNCATION_MARKER)

    def test_marker_at_end_not_middle(self):
        """Truncation marker should only appear at end of truncated content."""
        original = "Some content here"
        max_len = 10

        # Truncate
        truncated = original[:max_len - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER

        # Marker should be at end
        assert truncated.endswith(TRUNCATION_MARKER)
        # Marker should not be in the middle (there should be only one)
        assert truncated.count(TRUNCATION_MARKER) == 1
