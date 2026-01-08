"""Evidence extraction node for the Deep Research agent.

This module implements deterministic evidence extraction (S03) that mines
candidate quotes from stored raw content per TRUST_ARCH.md.

Supports two extraction modes:
- extract_api: Uses spacy-based sentence chunking for clean Tavily Extract content
- search_raw/fallback: Uses regex-based sanitization for raw HTML content
"""

import hashlib
import logging
from typing import List

from langchain_core.runnables import RunnableConfig

from open_deep_research.logic.document_processing import chunk_by_sentences
from open_deep_research.logic.sanitize import extract_paragraphs, sanitize_for_quotes
from open_deep_research.state import AgentState, EvidenceSnippet
from open_deep_research.utils import get_stored_sources


def generate_snippet_id(source_id: str, quote: str) -> str:
    """Generate a unique ID for an evidence snippet.

    Args:
        source_id: The source URL or ID
        quote: The quote text

    Returns:
        SHA-256 hash of source_id + quote (first 16 chars)
    """
    content = f"{source_id}:{quote}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def extract_evidence(state: AgentState, config: RunnableConfig) -> dict:
    """Extract candidate evidence snippets from stored source content.

    Per TRUST_ARCH.md Section B (Evidence Extraction):
    - Input: state["source_store"] or sources from LangGraph Store
    - Process: Sanitize HTML, split paragraphs, filter by word count
    - Output: state["evidence_snippets"] (list of EvidenceSnippet dicts)

    This is a DETERMINISTIC node - no LLM calls, purely string processing.
    Per Invariant I5, this only READS from Store, never writes.

    Args:
        state: Current agent state with source_store
        config: Runtime configuration

    Returns:
        Dictionary with evidence_snippets to add to state
    """
    # Step 1: Check if verification is disabled (fail-fast from S02)
    if state.get("verified_disabled", False):
        logging.info("[EXTRACT] Verification disabled, skipping evidence extraction.")
        print("[EXTRACT] ⚠️ Skipping extraction (verification disabled)")
        return {"evidence_snippets": []}

    # Step 2: Get sources - try state first, then LangGraph Store
    sources = state.get("source_store", [])
    sources_from_fallback = False

    if not sources:
        # Fallback to LangGraph Store
        sources = await get_stored_sources(config)
        sources_from_fallback = bool(sources)

    if not sources:
        logging.warning("[EXTRACT] No sources available for extraction.")
        print("[EXTRACT] ⚠️ No sources found for extraction")
        return {"evidence_snippets": []}

    print(f"[EXTRACT] Processing {len(sources)} sources for evidence...")

    # Step 3: Extract evidence from each source
    all_snippets: List[EvidenceSnippet] = []

    for source in sources:
        source_url = source.get("url", "")
        source_title = source.get("title", "Unknown")
        content = source.get("content", "")
        extraction_method = source.get("extraction_method", "search_raw")

        if not content:
            logging.debug(f"[EXTRACT] Skipping source with no content: {source_url}")
            continue

        # Step 3a: Extract passages based on content type
        if extraction_method == "extract_api":
            # Clean content from Tavily Extract - use spacy-based chunking
            passages = chunk_by_sentences(
                content,
                min_words=10,
                max_words=100,
                min_score=0.3
            )
        else:
            # Raw HTML content - use regex-based sanitization + paragraph extraction
            clean_text = sanitize_for_quotes(content)
            if not clean_text:
                continue
            passages = extract_paragraphs(clean_text, min_words=15, max_words=100)

        # Step 3b: Create EvidenceSnippet for each passage
        for passage in passages:
            snippet: EvidenceSnippet = {
                "snippet_id": generate_snippet_id(source_url, passage),
                "source_id": source_url,  # Keying ID (currently URL, may become hash)
                "url": source_url,        # Display URL for report links
                "source_title": source_title,
                "quote": passage,
                "status": "PENDING"  # Will be verified in S04
            }
            all_snippets.append(snippet)

    # Step 4: Log results
    logging.info(f"[EXTRACT] Extracted {len(all_snippets)} candidate snippets from {len(sources)} sources")
    print(f"[EXTRACT] ✓ Extracted {len(all_snippets)} candidate quotes")

    # Limit to reasonable number to prevent state bloat
    # Use round-robin selection to ensure diversity across sources
    max_snippets = 100
    if len(all_snippets) > max_snippets:
        logging.warning(f"[EXTRACT] Limiting to {max_snippets} snippets (had {len(all_snippets)})")

        # Group by source for diversity
        from collections import defaultdict
        by_source = defaultdict(list)
        for snippet in all_snippets:
            source_url = snippet.get("source_id", snippet.get("url", "unknown"))
            by_source[source_url].append(snippet)

        # Round-robin selection across sources
        diverse_snippets = []
        source_urls = list(by_source.keys())
        round_num = 0
        max_per_source = max(5, max_snippets // len(source_urls) + 1) if source_urls else 5

        while len(diverse_snippets) < max_snippets:
            added_this_round = False
            for url in source_urls:
                if round_num < len(by_source[url]) and len(diverse_snippets) < max_snippets:
                    diverse_snippets.append(by_source[url][round_num])
                    added_this_round = True
            if not added_this_round:
                break
            round_num += 1

        all_snippets = diverse_snippets
        unique_sources = len(set(s.get("source_id", s.get("url", "")) for s in all_snippets))
        logging.info(f"[EXTRACT] Selected {len(all_snippets)} snippets from {unique_sources} sources")

    # If sources were retrieved from fallback, propagate them to state for downstream nodes
    result = {"evidence_snippets": all_snippets}
    if sources_from_fallback:
        result["source_store"] = sources
        print(f"[EXTRACT] Propagating {len(sources)} sources to state (from Store fallback)")

    return result
