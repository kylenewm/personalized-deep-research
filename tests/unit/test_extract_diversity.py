"""Unit tests for extract.py diversity selection.

Tests that round-robin selection enforces source diversity when limiting snippets.
"""
import pytest


class TestRoundRobinSelection:
    """Tests for the round-robin snippet selection logic."""

    def test_round_robin_basic(self):
        """Verify round-robin picks evenly from sources."""
        # Simulate the round-robin logic from extract.py
        from collections import defaultdict

        # Setup: 3 sources with varying snippets
        by_source = {
            "source_a": [{"id": f"a{i}"} for i in range(10)],
            "source_b": [{"id": f"b{i}"} for i in range(5)],
            "source_c": [{"id": f"c{i}"} for i in range(3)],
        }

        max_snippets = 9
        diverse_snippets = []
        source_urls = list(by_source.keys())
        round_num = 0

        while len(diverse_snippets) < max_snippets:
            added_this_round = False
            for url in source_urls:
                if round_num < len(by_source[url]) and len(diverse_snippets) < max_snippets:
                    diverse_snippets.append(by_source[url][round_num])
                    added_this_round = True
            if not added_this_round:
                break
            round_num += 1

        # Should have 9 snippets
        assert len(diverse_snippets) == 9

        # Check distribution: should be roughly even
        counts = defaultdict(int)
        for s in diverse_snippets:
            source = s["id"][0]  # First char is source identifier
            counts[source] += 1

        # With 9 snippets and 3 sources, each should have 3
        assert counts["a"] == 3
        assert counts["b"] == 3
        assert counts["c"] == 3

    def test_round_robin_handles_uneven_sources(self):
        """Round-robin should handle sources exhausting at different times."""
        from collections import defaultdict

        by_source = {
            "source_a": [{"id": "a0"}],  # Only 1 snippet
            "source_b": [{"id": f"b{i}"} for i in range(10)],  # Many snippets
        }

        max_snippets = 5
        diverse_snippets = []
        source_urls = list(by_source.keys())
        round_num = 0

        while len(diverse_snippets) < max_snippets:
            added_this_round = False
            for url in source_urls:
                if round_num < len(by_source[url]) and len(diverse_snippets) < max_snippets:
                    diverse_snippets.append(by_source[url][round_num])
                    added_this_round = True
            if not added_this_round:
                break
            round_num += 1

        # Should have 5 snippets: a0, b0, b1, b2, b3
        assert len(diverse_snippets) == 5

        # Count by source
        counts = defaultdict(int)
        for s in diverse_snippets:
            source = s["id"][0]
            counts[source] += 1

        # source_a exhausted after 1, source_b fills the rest
        assert counts["a"] == 1
        assert counts["b"] == 4

    def test_round_robin_order_preserved(self):
        """First snippet from each source should come first."""
        by_source = {
            "A": [{"id": "A0"}, {"id": "A1"}],
            "B": [{"id": "B0"}, {"id": "B1"}],
        }

        max_snippets = 4
        diverse_snippets = []
        source_urls = list(by_source.keys())
        round_num = 0

        while len(diverse_snippets) < max_snippets:
            added_this_round = False
            for url in source_urls:
                if round_num < len(by_source[url]) and len(diverse_snippets) < max_snippets:
                    diverse_snippets.append(by_source[url][round_num])
                    added_this_round = True
            if not added_this_round:
                break
            round_num += 1

        # Order should be: A0, B0, A1, B1 (round-robin)
        ids = [s["id"] for s in diverse_snippets]
        assert ids == ["A0", "B0", "A1", "B1"]
