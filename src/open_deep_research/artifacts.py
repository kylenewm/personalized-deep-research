"""Run artifacts for reproducibility and debugging.

Stores immutable records of pipeline runs including:
- Inputs (query, config, prompt versions)
- Sources (URL, content hash, title)
- Pipeline stages (pointers, extractions, dedup decisions, arrangements, synthesis)
- Final output hash

This allows:
- Replaying runs with different prompts
- Diffing runs to understand changes
- Attributing regressions to specific prompt changes
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class SourceArtifact:
    """Immutable record of a source document."""
    url: str
    title: str
    content_hash: str  # SHA256 of content
    content_length: int


@dataclass
class PointerArtifact:
    """Record of LLM pointer output."""
    source_id: str
    keywords: List[str]
    micro_quote: Optional[str]
    context: str


@dataclass
class ExtractionArtifact:
    """Record of extraction result."""
    pointer_source_id: str
    status: str
    extracted_text: Optional[str]
    match_score: float
    span_start: int
    span_end: int
    keywords_matched: List[str]
    verification_method: str
    failure_reason: Optional[str]


@dataclass
class DedupDecision:
    """Record of what was removed in deduplication."""
    kept_index: int
    removed_index: int
    similarity: float
    reason: str  # "jaccard_duplicate", "same_source", etc.


@dataclass
class RunArtifacts:
    """Complete record of a pipeline run."""
    run_id: str
    timestamp: str

    # Inputs
    query: str
    config_hash: str
    prompt_versions: Dict[str, str]  # prompt_name -> sha256[:8]

    # Sources (immutable)
    sources: List[SourceArtifact] = field(default_factory=list)

    # Pipeline stages
    pointers: List[PointerArtifact] = field(default_factory=list)
    extractions: List[ExtractionArtifact] = field(default_factory=list)
    dedup_decisions: List[DedupDecision] = field(default_factory=list)
    arrangement: Dict[str, Any] = field(default_factory=dict)
    synthesis_themes: List[str] = field(default_factory=list)

    # Validation results
    synthesis_violations: List[str] = field(default_factory=list)

    # Final output
    report_hash: str = ""
    verified_count: int = 0
    total_extracted: int = 0


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_prompt_version(prompt: str) -> str:
    """Compute short hash for prompt versioning."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def compute_prompt_versions() -> Dict[str, str]:
    """Compute version hashes for all prompts in the pipeline.

    Import prompts dynamically to avoid circular imports.
    """
    versions = {}

    try:
        from .pointer_extract import POINTER_PROMPT, CLEANUP_PROMPT
        versions["POINTER_PROMPT"] = compute_prompt_version(POINTER_PROMPT)
        versions["CLEANUP_PROMPT"] = compute_prompt_version(CLEANUP_PROMPT)
    except ImportError:
        pass

    try:
        from .pipeline_v2 import (
            ARRANGER_PROMPT,
            THEME_SYNTHESIS_PROMPT,
            EXECUTIVE_SUMMARY_PROMPT,
            ANALYSIS_PROMPT,
            CONCLUSION_PROMPT,
        )
        versions["ARRANGER_PROMPT"] = compute_prompt_version(ARRANGER_PROMPT)
        versions["THEME_SYNTHESIS_PROMPT"] = compute_prompt_version(THEME_SYNTHESIS_PROMPT)
        versions["EXECUTIVE_SUMMARY_PROMPT"] = compute_prompt_version(EXECUTIVE_SUMMARY_PROMPT)
        versions["ANALYSIS_PROMPT"] = compute_prompt_version(ANALYSIS_PROMPT)
        versions["CONCLUSION_PROMPT"] = compute_prompt_version(CONCLUSION_PROMPT)
    except ImportError:
        pass

    return versions


def create_run_artifacts(
    query: str,
    config: Optional[dict] = None,
    run_id: Optional[str] = None
) -> RunArtifacts:
    """Create a new run artifacts record.

    Args:
        query: Research query
        config: Pipeline configuration dict
        run_id: Optional run ID (auto-generated if not provided)

    Returns:
        Initialized RunArtifacts
    """
    import uuid

    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    config_str = json.dumps(config or {}, sort_keys=True)
    config_hash = compute_content_hash(config_str)

    return RunArtifacts(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        query=query,
        config_hash=config_hash,
        prompt_versions=compute_prompt_versions()
    )


def save_run_artifacts(artifacts: RunArtifacts, output_dir: Path) -> Path:
    """Save run artifacts to JSON file.

    Args:
        artifacts: The artifacts to save
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"run_{artifacts.run_id}_{artifacts.timestamp[:10]}.json"
    filepath = output_dir / filename

    # Convert to dict for JSON serialization
    data = asdict(artifacts)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return filepath


def load_run_artifacts(filepath: Path) -> RunArtifacts:
    """Load run artifacts from JSON file.

    Args:
        filepath: Path to artifacts file

    Returns:
        Loaded RunArtifacts
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Convert nested dicts back to dataclasses
    sources = [SourceArtifact(**s) for s in data.pop('sources', [])]
    pointers = [PointerArtifact(**p) for p in data.pop('pointers', [])]
    extractions = [ExtractionArtifact(**e) for e in data.pop('extractions', [])]
    dedup_decisions = [DedupDecision(**d) for d in data.pop('dedup_decisions', [])]

    return RunArtifacts(
        sources=sources,
        pointers=pointers,
        extractions=extractions,
        dedup_decisions=dedup_decisions,
        **data
    )


def diff_prompt_versions(
    old_artifacts: RunArtifacts,
    new_artifacts: RunArtifacts
) -> Dict[str, tuple]:
    """Compare prompt versions between two runs.

    Args:
        old_artifacts: Previous run
        new_artifacts: Current run

    Returns:
        Dict of changed prompts: {prompt_name: (old_hash, new_hash)}
    """
    changes = {}

    all_prompts = set(old_artifacts.prompt_versions.keys()) | set(new_artifacts.prompt_versions.keys())

    for prompt_name in all_prompts:
        old_hash = old_artifacts.prompt_versions.get(prompt_name, "missing")
        new_hash = new_artifacts.prompt_versions.get(prompt_name, "missing")

        if old_hash != new_hash:
            changes[prompt_name] = (old_hash, new_hash)

    return changes
