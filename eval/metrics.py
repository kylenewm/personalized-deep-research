"""Metrics calculation and thresholds - no pipeline dependencies."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResult:
    """Result of an evaluation run."""
    dataset: str
    mode: str  # mini, medium, full

    # Brief metrics (query → brief transformation)
    brief_preservation: Optional[float] = None  # Did brief keep query specifics?
    brief_dilution: Optional[float] = None  # Did brief generalize too much?
    brief_assumptions: Optional[float] = None  # Did brief add unwarranted assumptions?
    brief_recommendation: Optional[str] = None  # GOOD/WARN/FAIL

    # Upstream metrics
    avg_fact_quality: float = 0
    avg_theme_coverage: float = 0
    duplicate_rate: float = 0
    low_quality_rate: float = 0
    match_score_avg: Optional[float] = None  # From saved data, may not exist

    # Downstream metrics
    avg_citation_accuracy: Optional[float] = None
    avg_synthesis_quality: Optional[float] = None
    uncited_rate: Optional[float] = None

    # Meta
    total_facts: int = 0
    total_themes: int = 0
    cost_estimate: float = 0

    def brief_status(self) -> str:
        """Return PASS/WARN/FAIL for brief transformation."""
        if self.brief_recommendation is None:
            return "SKIP (no brief)"
        return self.brief_recommendation

    def upstream_status(self) -> str:
        """Return PASS/WARN/FAIL for upstream metrics."""
        # Hard fails
        if self.avg_fact_quality < 2.0:
            return "FAIL"
        if self.duplicate_rate > 0.20:
            return "FAIL"

        # Warnings
        warnings = []
        if self.avg_fact_quality < 3.5:
            warnings.append("fact_quality")
        if self.avg_theme_coverage < 3.5:
            warnings.append("theme_coverage")
        if self.duplicate_rate > 0.15:
            warnings.append("duplicates")
        if self.low_quality_rate > 0.10:
            warnings.append("low_quality")

        if warnings:
            return f"WARN ({', '.join(warnings)})"
        return "PASS"

    def downstream_status(self) -> str:
        """Return PASS/WARN/FAIL for downstream metrics."""
        if self.avg_citation_accuracy is None:
            return "SKIP (no report)"

        # Hard fails
        if self.uncited_rate is not None and self.uncited_rate > 0.30:
            return "FAIL"

        # Warnings
        warnings = []
        if self.avg_citation_accuracy < 4.0:
            warnings.append("citation_accuracy")
        if self.avg_synthesis_quality is not None and self.avg_synthesis_quality < 3.5:
            warnings.append("synthesis_quality")
        if self.uncited_rate is not None and self.uncited_rate > 0.05:
            warnings.append("uncited_rate")

        if warnings:
            return f"WARN ({', '.join(warnings)})"
        return "PASS"

    def overall_status(self) -> str:
        """Return overall PASS/WARN/FAIL."""
        brief = self.brief_status()
        up = self.upstream_status()
        down = self.downstream_status()

        if "FAIL" in brief or "FAIL" in up or "FAIL" in down:
            return "FAIL"
        if "WARN" in brief or "WARN" in up or "WARN" in down:
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict:
        """Convert to dict for JSON output."""
        return {
            "dataset": self.dataset,
            "mode": self.mode,
            "brief": {
                "preservation": self.brief_preservation,
                "dilution": self.brief_dilution,
                "assumptions": self.brief_assumptions,
                "status": self.brief_status()
            },
            "upstream": {
                "avg_fact_quality": self.avg_fact_quality,
                "avg_theme_coverage": self.avg_theme_coverage,
                "duplicate_rate": self.duplicate_rate,
                "low_quality_rate": self.low_quality_rate,
                "match_score_avg": self.match_score_avg,
                "status": self.upstream_status()
            },
            "downstream": {
                "avg_citation_accuracy": self.avg_citation_accuracy,
                "avg_synthesis_quality": self.avg_synthesis_quality,
                "uncited_rate": self.uncited_rate,
                "status": self.downstream_status()
            },
            "summary": {
                "total_facts": self.total_facts,
                "total_themes": self.total_themes,
                "cost_estimate": self.cost_estimate,
                "overall_status": self.overall_status()
            }
        }


# Thresholds (for reference)
THRESHOLDS = {
    "upstream": {
        "avg_fact_quality": {"target": 3.5, "hard_fail": 2.0},
        "avg_theme_coverage": {"target": 3.5, "hard_fail": None},
        "duplicate_rate": {"target": 0.15, "hard_fail": 0.20},
        "low_quality_rate": {"target": 0.10, "hard_fail": None},
        "match_score_avg": {"target": 0.80, "hard_fail": None},
    },
    "downstream": {
        "avg_citation_accuracy": {"target": 4.0, "hard_fail": None},
        "avg_synthesis_quality": {"target": 3.5, "hard_fail": None},
        "uncited_rate": {"target": 0.05, "hard_fail": 0.30},
    }
}
