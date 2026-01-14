"""Claim verification node (Trust Engine) for the Deep Research agent.

This module implements:
- verify_evidence (S04): Deterministic substring/Jaccard verification of extracted snippets
- verify_claims: Post-report claim verification using embeddings
"""

import logging
import re
from typing import Dict, List, Set

from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.logic.sanitize import sanitize_for_quotes
from open_deep_research.state import AgentState, EvidenceSnippet
from open_deep_research.utils import TRUNCATION_MARKER, get_stored_sources


##########################
# S04: Evidence Verification (Deterministic)
##########################

def tokenize(text: str) -> Set[str]:
    """Tokenize text into a set of lowercase words, preserving acronyms and codes.

    Handles special patterns that should stay together:
    - Acronyms with periods: U.S.A., N.A.S.A.
    - Hyphenated codes: NIST-SP-800-53, AEF-1, M-25-22
    - Dollar amounts: $1.2B, $25T

    Args:
        text: Input text to tokenize

    Returns:
        Set of lowercase word tokens
    """
    # First, extract and preserve special patterns that should stay together
    preserved = []

    # Hyphenated codes (e.g., NIST-SP-800-53, AEF-1)
    preserved.extend(re.findall(r'\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b', text))

    # Acronyms with periods (e.g., U.S.A., N.A.S.A.)
    preserved.extend(re.findall(r'\b(?:[A-Za-z]\.){2,}[A-Za-z]?', text))

    # Dollar amounts (e.g., $1.2B, $25T, $1,000)
    preserved.extend(re.findall(r'\$[\d,]+(?:\.\d+)?(?:[BMKT])?', text, re.IGNORECASE))

    # Standard word tokenization for everything else
    words = re.findall(r'\b\w+\b', text.lower())

    # Combine: preserved tokens (lowercased) + regular words
    all_tokens = set(words)
    for token in preserved:
        all_tokens.add(token.lower())

    return all_tokens


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts.

    Jaccard similarity = |intersection| / |union|

    Per TRUST_ARCH.md, threshold is 0.8 for fuzzy matching.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity score between 0.0 and 1.0
    """
    set1 = tokenize(text1)
    set2 = tokenize(text2)

    if not set1 or not set2:
        return 0.0

    intersection = set1 & set2
    union = set1 | set2

    return len(intersection) / len(union)


def verify_quote_in_source(quote: str, source_content: str, fuzzy_threshold: float = 0.8) -> str:
    """Verify if a quote exists in source content.

    Per TRUST_ARCH.md Section C (Verification):
    1. Check 1 (Strict): if quote in clean_text -> PASS
    2. Check 2 (Fuzzy): Jaccard similarity > 0.8 -> PASS
    3. Else -> FAIL

    This is DETERMINISTIC - no LLM calls (Invariant I3).

    Args:
        quote: The quote to verify
        source_content: Raw or sanitized source content
        fuzzy_threshold: Jaccard threshold for fuzzy match (default 0.8)

    Returns:
        "PASS" or "FAIL"
    """
    if not quote or not source_content:
        return "FAIL"

    # Sanitize source content for comparison
    clean_source = sanitize_for_quotes(source_content)

    # Check 1: Strict substring match
    if quote in clean_source:
        return "PASS"

    # Check 2: Fuzzy matching with Jaccard similarity
    # We need to find if any substring of similar length has high similarity
    quote_words = quote.split()
    quote_len = len(quote_words)

    # Slide a window across the source to find best match
    source_words = clean_source.split()

    best_similarity = 0.0
    window_size = quote_len

    for i in range(max(1, len(source_words) - window_size + 1)):
        window = ' '.join(source_words[i:i + window_size])
        similarity = jaccard_similarity(quote, window)
        best_similarity = max(best_similarity, similarity)

        # Early exit if we found a good match
        if best_similarity >= fuzzy_threshold:
            return "PASS"

    # Also try with slightly larger/smaller windows for flexibility
    for size_delta in [-2, -1, 1, 2]:
        adjusted_size = quote_len + size_delta
        if adjusted_size < 5:
            continue

        for i in range(max(1, len(source_words) - adjusted_size + 1)):
            window = ' '.join(source_words[i:i + adjusted_size])
            similarity = jaccard_similarity(quote, window)
            if similarity >= fuzzy_threshold:
                return "PASS"

    return "FAIL"


async def verify_evidence(state: AgentState, config: RunnableConfig) -> dict:
    """Verify extracted evidence snippets against source content.

    Per TRUST_ARCH.md Section C (Verification):
    - Input: state["evidence_snippets"]
    - Process: For each snippet, check strict then fuzzy match
    - Output: Updated evidence_snippets with status PASS/FAIL

    This is DETERMINISTIC - no LLM calls (Invariant I3).

    Args:
        state: Current agent state with evidence_snippets
        config: Runtime configuration

    Returns:
        Dictionary with updated evidence_snippets (status field set)
    """
    # Step 1: Check if verification is disabled
    if state.get("verified_disabled", False):
        logging.info("[VERIFY_EVIDENCE] Verification disabled, marking snippets as SKIP.")
        print("[VERIFY_EVIDENCE] ⚠️ Skipping (verification disabled)")
        # Mark all snippets as SKIP instead of leaving them PENDING
        snippets = state.get("evidence_snippets", [])
        if snippets:
            skipped_snippets = [{**s, "status": "SKIP"} for s in snippets]
            return {
                "evidence_snippets": {
                    "type": "override",
                    "value": skipped_snippets
                }
            }
        return {}

    # Step 2: Get evidence snippets
    snippets = state.get("evidence_snippets", [])
    if not snippets:
        logging.info("[VERIFY_EVIDENCE] No snippets to verify.")
        print("[VERIFY_EVIDENCE] ⚠️ No snippets to verify")
        return {}

    # Step 3: Build source content lookup
    sources = state.get("source_store", [])
    if not sources:
        sources = await get_stored_sources(config)

    # Create URL -> content mapping
    source_content_map: Dict[str, str] = {
        source.get("url", ""): source.get("content", "")
        for source in sources
    }

    print(f"[VERIFY_EVIDENCE] Verifying {len(snippets)} snippets against {len(sources)} sources...")

    # Step 4: Verify each snippet
    verified_snippets: List[EvidenceSnippet] = []
    pass_count = 0
    fail_count = 0

    for snippet in snippets:
        source_id = snippet.get("source_id", "")
        quote = snippet.get("quote", "")

        # Get source content
        source_content = source_content_map.get(source_id, "")

        # Verify quote against source
        status = verify_quote_in_source(quote, source_content)

        # Create updated snippet with status
        verified_snippet: EvidenceSnippet = {
            **snippet,
            "status": status
        }
        verified_snippets.append(verified_snippet)

        if status == "PASS":
            pass_count += 1
        else:
            fail_count += 1

    # Step 5: Log results
    total = len(verified_snippets)
    logging.info(f"[VERIFY_EVIDENCE] Verified {total}: {pass_count} PASS, {fail_count} FAIL")
    print(f"[VERIFY] ✓ {pass_count}/{total} snippets verified (PASS)")
    if fail_count > 0:
        print(f"[VERIFY] ⚠️ {fail_count}/{total} snippets failed verification")

    # Log source diversity for PASS snippets
    pass_sources = set(s.get("source_id", s.get("url", "")) for s in verified_snippets if s.get("status") == "PASS")
    print(f"[VERIFY] Source diversity: {len(pass_sources)} sources with PASS quotes")

    # Return with override to replace existing snippets
    return {
        "evidence_snippets": {
            "type": "override",
            "value": verified_snippets
        }
    }


