"""Logic modules for the Deep Research agent.

This package contains deterministic processing logic (no LLM calls).
"""

from open_deep_research.logic.sanitize import extract_paragraphs, sanitize_for_quotes

__all__ = [
    "sanitize_for_quotes",
    "extract_paragraphs",
]
