"""Metrics collection and logging for Deep Research runs.

This module provides:
- RunMetrics: Dataclass for tracking run statistics
- MetricsCollector: Context manager for timing stages
- save_metrics_json: Save metrics to JSON file per run
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, success: bool = True, error: str = None):
        """Mark stage as complete."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.success = success
        self.error = error


@dataclass
class RunMetrics:
    """Complete metrics for a research run."""

    # Run identification
    run_id: str = ""
    query: str = ""
    started_at: str = ""
    finished_at: str = ""

    # Timing
    total_duration_seconds: float = 0.0

    # Stage metrics
    stages: List[StageMetrics] = field(default_factory=list)

    # Research metrics
    brief_length_chars: int = 0
    sources_found: int = 0
    sources_stored: int = 0
    notes_generated: int = 0
    raw_notes_chars: int = 0

    # Evidence metrics (S03/S04)
    snippets_extracted: int = 0
    snippets_verified_pass: int = 0
    snippets_verified_fail: int = 0

    # Report metrics
    report_length_chars: int = 0

    # Verification metrics
    claims_verified: int = 0
    claims_supported: int = 0
    verification_confidence: float = 0.0

    # Config snapshot
    config: Dict[str, Any] = field(default_factory=dict)

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MetricsCollector:
    """Collects metrics throughout a research run.

    Usage:
        collector = MetricsCollector(query="What are AI agents?")

        with collector.stage("brief_generation"):
            # ... do brief generation ...
            collector.metrics.brief_length_chars = len(brief)

        with collector.stage("research"):
            # ... do research ...
            collector.metrics.sources_found = 25

        collector.finish()
        collector.save("./metrics/")
    """

    def __init__(self, query: str = "", run_id: str = None, config: dict = None):
        """Initialize metrics collector.

        Args:
            query: The research query being executed
            run_id: Optional run ID (auto-generated if not provided)
            config: Optional config dict to snapshot
        """
        self.metrics = RunMetrics(
            run_id=run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            query=query,
            started_at=datetime.now().isoformat(),
            config=config or {}
        )
        self._start_time = time.time()
        self._current_stage: Optional[StageMetrics] = None

    def stage(self, name: str, metadata: dict = None) -> "StageContext":
        """Create a context manager for timing a stage.

        Args:
            name: Name of the stage (e.g., "brief_generation", "research")
            metadata: Optional metadata to attach to stage

        Returns:
            StageContext for use in with statement
        """
        return StageContext(self, name, metadata)

    def start_stage(self, name: str, metadata: dict = None) -> StageMetrics:
        """Manually start a stage (prefer using stage() context manager)."""
        stage = StageMetrics(
            name=name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self._current_stage = stage
        self.metrics.stages.append(stage)
        return stage

    def end_stage(self, success: bool = True, error: str = None):
        """Manually end the current stage."""
        if self._current_stage:
            self._current_stage.finish(success, error)
            self._current_stage = None

    def log_error(self, error: str):
        """Log an error that occurred during the run."""
        self.metrics.errors.append(f"[{datetime.now().isoformat()}] {error}")

    def log_warning(self, warning: str):
        """Log a warning that occurred during the run."""
        self.metrics.warnings.append(f"[{datetime.now().isoformat()}] {warning}")

    def finish(self):
        """Mark the run as complete and calculate totals."""
        self.metrics.finished_at = datetime.now().isoformat()
        self.metrics.total_duration_seconds = time.time() - self._start_time

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        data = asdict(self.metrics)
        # Convert StageMetrics to dicts
        data["stages"] = [asdict(s) for s in self.metrics.stages]
        return data

    def save(self, output_dir: str = "./metrics", filename: str = None) -> Path:
        """Save metrics to JSON file.

        Args:
            output_dir: Directory to save metrics to
            filename: Optional filename (default: metrics_{run_id}.json)

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"metrics_{self.metrics.run_id}.json"

        file_path = output_path / filename

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return file_path

    def print_summary(self):
        """Print a human-readable summary of the metrics."""
        m = self.metrics

        print("\n" + "=" * 60)
        print("RUN METRICS SUMMARY")
        print("=" * 60)

        print(f"\nRun ID: {m.run_id}")
        print(f"Query: {m.query[:50]}..." if len(m.query) > 50 else f"Query: {m.query}")
        print(f"Total Duration: {m.total_duration_seconds:.1f}s")

        print("\n--- Stage Timing ---")
        for stage in m.stages:
            status = "OK" if stage.success else f"FAIL: {stage.error}"
            print(f"  {stage.name}: {stage.duration_seconds:.1f}s [{status}]")

        print("\n--- Research Metrics ---")
        print(f"  Brief: {m.brief_length_chars} chars")
        print(f"  Sources: {m.sources_stored} stored ({m.sources_found} found)")
        print(f"  Notes: {m.notes_generated} entries ({m.raw_notes_chars} chars)")

        if m.snippets_extracted > 0:
            print("\n--- Evidence Metrics ---")
            print(f"  Extracted: {m.snippets_extracted} snippets")
            print(f"  Verified: {m.snippets_verified_pass} PASS, {m.snippets_verified_fail} FAIL")

        if m.report_length_chars > 0:
            print("\n--- Report Metrics ---")
            print(f"  Report: {m.report_length_chars} chars")

        if m.claims_verified > 0:
            print("\n--- Verification Metrics ---")
            print(f"  Claims: {m.claims_supported}/{m.claims_verified} supported")
            print(f"  Confidence: {m.verification_confidence:.0%}")

        if m.errors:
            print("\n--- Errors ---")
            for error in m.errors[:5]:
                print(f"  {error}")

        if m.warnings:
            print("\n--- Warnings ---")
            for warning in m.warnings[:5]:
                print(f"  {warning}")

        print("\n" + "=" * 60)


class StageContext:
    """Context manager for timing a pipeline stage."""

    def __init__(self, collector: MetricsCollector, name: str, metadata: dict = None):
        self.collector = collector
        self.name = name
        self.metadata = metadata
        self.stage: Optional[StageMetrics] = None

    def __enter__(self) -> StageMetrics:
        self.stage = self.collector.start_stage(self.name, self.metadata)
        return self.stage

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.collector.end_stage(success=False, error=str(exc_val))
            self.collector.log_error(f"{self.name}: {exc_val}")
        else:
            self.collector.end_stage(success=True)
        return False  # Don't suppress exceptions


def create_run_summary(state: dict, duration: float = 0.0) -> dict:
    """Create a quick summary from final state for logging.

    Args:
        state: The final agent state
        duration: Total run duration in seconds

    Returns:
        Summary dict suitable for logging
    """
    return {
        "duration_seconds": duration,
        "brief_length": len(state.get("research_brief", "")),
        "sources_count": len(state.get("source_store", [])),
        "notes_count": len(state.get("notes", [])),
        "raw_notes_chars": sum(len(n) for n in state.get("raw_notes", [])),
        "report_length": len(state.get("final_report", "")),
        "verified_disabled": state.get("verified_disabled", False),
        "snippets_extracted": len(state.get("evidence_snippets", [])),
        "snippets_pass": sum(1 for s in state.get("evidence_snippets", []) if s.get("status") == "PASS"),
        "verification_result": state.get("verification_result") is not None,
    }


# Convenience function for quick metrics
def quick_metrics(query: str, config: dict = None) -> MetricsCollector:
    """Create a MetricsCollector with sensible defaults.

    Args:
        query: The research query
        config: Optional config dict

    Returns:
        MetricsCollector ready to use
    """
    return MetricsCollector(
        query=query,
        config=config
    )
