"""Node modules for the Deep Research agent.

This package contains modular node implementations for the research workflow.
"""

from open_deep_research.nodes.brief import validate_brief, write_research_brief
from open_deep_research.nodes.clarify import clarify_with_user
from open_deep_research.nodes.extract import extract_evidence
from open_deep_research.nodes.findings import validate_findings
from open_deep_research.nodes.report import final_report_generation
from open_deep_research.nodes.store import check_store
from open_deep_research.nodes.researcher import (
    compress_research,
    researcher,
    researcher_subgraph,
    researcher_tools,
)
from open_deep_research.nodes.supervisor import (
    supervisor,
    supervisor_subgraph,
    supervisor_tools,
)
from open_deep_research.nodes.verify import verify_claims, verify_evidence

__all__ = [
    # Store Gating (S02)
    "check_store",
    # Evidence Extraction (S03)
    "extract_evidence",
    # Clarification
    "clarify_with_user",
    # Brief
    "write_research_brief",
    "validate_brief",
    # Supervisor
    "supervisor",
    "supervisor_tools",
    "supervisor_subgraph",
    # Researcher
    "researcher",
    "researcher_tools",
    "compress_research",
    "researcher_subgraph",
    # Findings
    "validate_findings",
    # Report
    "final_report_generation",
    # Verification
    "verify_claims",
    "verify_evidence",
]
