"""Unit tests for source storage and flow.

Tests that sources flow correctly from researcher to verification,
and documents the current behavior of dual storage systems.
"""
import pytest
from open_deep_research.utils import (
    cache_sources,
    get_cached_sources,
    get_source_store_key,
    _source_cache,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear module-level cache before each test."""
    _source_cache.clear()
    yield
    _source_cache.clear()


class TestSourceCache:
    """Tests for module-level source cache."""

    def test_cache_and_retrieve(self):
        """Basic cache and retrieve."""
        config = {"configurable": {"thread_id": "test-thread"}}
        sources = [
            {"url": "https://example.com/1", "content": "Content 1"},
            {"url": "https://example.com/2", "content": "Content 2"},
        ]

        cache_sources(sources, config)
        retrieved = get_cached_sources(config)

        assert len(retrieved) == 2
        assert retrieved[0]["url"] == "https://example.com/1"

    def test_cache_dedupes_by_url(self):
        """Cache should deduplicate by URL."""
        config = {"configurable": {"thread_id": "test-thread"}}

        cache_sources([{"url": "https://example.com/1", "content": "First"}], config)
        cache_sources([{"url": "https://example.com/1", "content": "Second"}], config)

        retrieved = get_cached_sources(config)

        assert len(retrieved) == 1
        # First one should be kept (not replaced)
        assert retrieved[0]["content"] == "First"

    def test_separate_threads(self):
        """Different threads should have separate caches."""
        config1 = {"configurable": {"thread_id": "thread-1"}}
        config2 = {"configurable": {"thread_id": "thread-2"}}

        cache_sources([{"url": "https://1.com", "content": "One"}], config1)
        cache_sources([{"url": "https://2.com", "content": "Two"}], config2)

        assert len(get_cached_sources(config1)) == 1
        assert len(get_cached_sources(config2)) == 1
        assert get_cached_sources(config1)[0]["url"] == "https://1.com"
        assert get_cached_sources(config2)[0]["url"] == "https://2.com"


class TestQualityFiltering:
    """Document quality filtering behavior in supervisor.

    Note: This documents the current behavior where supervisor filters
    sources <500 chars. This means verification might not have all
    sources the researcher cited.
    """

    def test_documented_behavior(self):
        """Document that short sources get filtered by supervisor.

        This is not necessarily a bug - short sources are likely
        paywalled/JS-heavy and couldn't be cited meaningfully.

        Current flow:
        1. Researcher sees all cached sources
        2. Supervisor filters sources < min_content_len (default 500)
        3. Verification only sees filtered sources

        Impact: If researcher cites a short source, verification will fail
        to find it. This is acceptable because:
        - Short sources likely don't have useful content
        - Verification would fail anyway with insufficient content
        """
        # This test documents behavior, doesn't test code
        pass


class TestDualStorageSync:
    """Document dual storage architecture.

    Current architecture:
    - _source_cache: Module-level dict, always populated
    - LangGraph Store: External store, async "fire and forget"
    - state.source_store: Flows through graph, primary for verification

    The cache and Store can technically diverge if Store write fails,
    but this doesn't affect verification because:
    1. Cache is always populated synchronously
    2. state.source_store comes from cache via researcher
    3. External Store is only a backup, rarely read
    """

    def test_documented_architecture(self):
        """Document that dual storage exists but primary path uses state."""
        # This test documents architecture, doesn't test code
        # See: utils.py lines 44-78 for cache
        # See: utils.py lines 80-130 for Store
        # See: researcher.py lines 287-301 for source_store flow
        pass
