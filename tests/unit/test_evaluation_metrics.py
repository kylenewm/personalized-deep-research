"""Unit tests for evaluation metrics calculation.

Tests the hallucination rate fix: should include FALSE + UNCITED claims.
"""
import pytest
from open_deep_research.evaluation import (
    ClaimResult,
    ClaimMetrics,
    calculate_claim_metrics,
)


def make_claim(
    status: str = "TRUE",
    is_uncited: bool = False,
    claim_id: str = "c001"
) -> ClaimResult:
    """Helper to create mock ClaimResult."""
    return ClaimResult(
        claim_id=claim_id,
        claim_text="Test claim",
        citations=[1] if not is_uncited else [],
        is_uncited=is_uncited,
        status=status,
        confidence=0.9,
        sources_checked=["http://example.com"],
        evidence_snippet="Evidence"
    )


class TestHallucinationRate:
    """Tests for hallucination rate calculation."""

    def test_all_cited_true_zero_hallucination(self):
        """All claims cited and TRUE → 0% hallucination."""
        results = [
            make_claim(status="TRUE", is_uncited=False),
            make_claim(status="TRUE", is_uncited=False),
            make_claim(status="TRUE", is_uncited=False),
        ]
        metrics = calculate_claim_metrics(results)
        assert metrics.hallucination_rate == 0.0
        assert metrics.grounding_rate == 1.0

    def test_false_claims_count_as_hallucination(self):
        """FALSE claims should be counted in hallucination rate."""
        results = [
            make_claim(status="TRUE", is_uncited=False),
            make_claim(status="FALSE", is_uncited=False),  # This is a hallucination
            make_claim(status="TRUE", is_uncited=False),
        ]
        metrics = calculate_claim_metrics(results)
        assert metrics.hallucination_rate == pytest.approx(1/3)
        assert metrics.false_count == 1

    def test_uncited_claims_count_as_hallucination(self):
        """UNCITED claims should be counted in hallucination rate (trust risk)."""
        results = [
            make_claim(status="TRUE", is_uncited=False),
            make_claim(status="TRUE", is_uncited=True),  # Uncited = trust risk
            make_claim(status="TRUE", is_uncited=False),
        ]
        metrics = calculate_claim_metrics(results)
        # Even though all TRUE, the uncited one is a trust risk
        assert metrics.hallucination_rate == pytest.approx(1/3)
        assert metrics.uncited_count == 1

    def test_no_double_counting_uncited_false(self):
        """An UNCITED FALSE claim should only be counted once."""
        results = [
            make_claim(status="TRUE", is_uncited=False),
            make_claim(status="FALSE", is_uncited=True),  # Both false AND uncited
            make_claim(status="TRUE", is_uncited=False),
        ]
        metrics = calculate_claim_metrics(results)
        # Should be 1/3, not 2/3 (no double counting)
        assert metrics.hallucination_rate == pytest.approx(1/3)
        assert metrics.false_count == 1
        assert metrics.uncited_count == 1

    def test_mixed_scenario(self):
        """Mix of FALSE, UNCITED, and good claims."""
        results = [
            make_claim(status="TRUE", is_uncited=False),   # Good
            make_claim(status="FALSE", is_uncited=False),  # Hallucination (false)
            make_claim(status="TRUE", is_uncited=True),    # Hallucination (uncited)
            make_claim(status="FALSE", is_uncited=True),   # Hallucination (both, count once)
            make_claim(status="TRUE", is_uncited=False),   # Good
        ]
        metrics = calculate_claim_metrics(results)
        # Risky: FALSE(cited) + TRUE(uncited) + FALSE(uncited) = 3 unique
        assert metrics.hallucination_rate == pytest.approx(3/5)
        assert metrics.grounding_rate == pytest.approx(3/5)  # 3 TRUE out of 5
        assert metrics.false_count == 2
        assert metrics.uncited_count == 2

    def test_empty_results(self):
        """Empty results should return zero metrics."""
        metrics = calculate_claim_metrics([])
        assert metrics.total == 0
        assert metrics.hallucination_rate == 0.0

    def test_all_unverifiable(self):
        """All UNVERIFIABLE claims."""
        results = [
            make_claim(status="UNVERIFIABLE", is_uncited=False),
            make_claim(status="UNVERIFIABLE", is_uncited=False),
        ]
        metrics = calculate_claim_metrics(results)
        assert metrics.unverifiable_count == 2
        assert metrics.hallucination_rate == 0.0  # Not FALSE, not UNCITED
        assert metrics.grounding_rate == 0.0  # Not TRUE


class TestOldVsNewBehavior:
    """Regression tests comparing old vs new hallucination rate definition."""

    def test_old_behavior_would_miss_uncited_true(self):
        """
        OLD: hallucination_rate = false_count / total
        NEW: hallucination_rate = (false OR uncited) / total

        This test verifies the NEW behavior catches uncited TRUE claims.
        """
        results = [
            make_claim(status="TRUE", is_uncited=False),
            make_claim(status="TRUE", is_uncited=True),  # OLD would miss this
            make_claim(status="TRUE", is_uncited=False),
        ]
        metrics = calculate_claim_metrics(results)

        # OLD behavior would have: 0/3 = 0%
        old_rate = metrics.false_count / metrics.total

        # NEW behavior has: 1/3 = 33% (includes uncited)
        new_rate = metrics.hallucination_rate

        assert old_rate == 0.0
        assert new_rate == pytest.approx(1/3)
        assert new_rate > old_rate  # New is more conservative (catches more risk)
