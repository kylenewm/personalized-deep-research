"""Layer 3: Claim soft-gate for the Deep Research pipeline.

Extracts key factual claims from research notes and verifies they
can be traced to source content. Logs warnings for unverifiable claims
but does NOT block report generation (warn-only approach).
"""

import re
from typing import List, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.models import configurable_model
from open_deep_research.state import AgentState, SourceRecord
from open_deep_research.utils import get_api_key_for_model


class ExtractedClaims(BaseModel):
    """Structured output for claim extraction."""
    claims: List[str] = Field(
        description="List of key factual claims from the research notes. "
                    "Focus on specific facts: names, numbers, dates, acronyms, "
                    "product names, organizations, and statistics."
    )


CLAIM_EXTRACTION_PROMPT = """Extract the key FACTUAL claims from these research notes.

Focus on claims that contain:
- Specific names (people, products, organizations, acts, bills)
- Numbers and statistics (percentages, dollar amounts, dates)
- Acronyms and codes (e.g., AEF-1, M-25-22)
- Specific events or actions with dates

Do NOT include:
- Vague statements ("many experts believe", "research shows")
- Opinions or interpretations
- General context or background

Return 10-20 of the most important factual claims.

Research Notes:
{notes}
"""


def extract_key_terms(claim: str) -> List[str]:
    """Extract key verifiable terms from a claim.

    Finds named entities, acronyms, and specific terms that should
    be searchable in source content.
    """
    key_terms = []

    # Acronyms and codes (e.g., AEF-1, M-25-22, LITHOS)
    key_terms.extend(re.findall(r'\b[A-Z][A-Z0-9-]+(?:-\d+)?\b', claim))

    # Proper nouns (2+ capitalized words like "Digital Trust Centre")
    key_terms.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', claim))

    # Quoted strings
    key_terms.extend(re.findall(r'"([^"]+)"', claim))

    # Percentages and dollar amounts
    key_terms.extend(re.findall(r'\d+(?:\.\d+)?%', claim))
    key_terms.extend(re.findall(r'\$[\d,]+(?:\s*(?:billion|million))?', claim))

    # Deduplicate and filter short terms
    return list(set(t for t in key_terms if len(t) >= 3))


def verify_claim_in_sources(
    claim: str,
    key_terms: List[str],
    source_store: List[SourceRecord]
) -> Tuple[bool, str]:
    """Check if a claim's key terms appear in any source.

    Returns:
        (is_verified, reason)
    """
    if not key_terms:
        return True, "no_key_terms"  # Can't verify, assume ok

    # Build combined content from all sources
    all_content = " ".join(
        s.get("content", "").lower() for s in source_store
    )

    # Check each key term
    found_terms = []
    missing_terms = []

    for term in key_terms:
        if term.lower() in all_content:
            found_terms.append(term)
        else:
            missing_terms.append(term)

    # If more than half of key terms are missing, flag it
    if missing_terms and len(missing_terms) > len(key_terms) * 0.5:
        return False, f"missing: {', '.join(missing_terms[:3])}"

    return True, f"found: {len(found_terms)}/{len(key_terms)} terms"


async def claim_pre_check(state: AgentState, config: RunnableConfig) -> dict:
    """Layer 3: Extract and verify key claims from research notes.

    This node:
    1. Extracts key factual claims via 1 LLM call (~$0.01)
    2. Verifies each claim's key terms exist in sources (string matching)
    3. Logs warnings for unverifiable claims (warn-only)

    Returns:
        Dictionary with claim_warnings added to state
    """
    configurable = Configuration.from_runnable_config(config)

    # Check if claim pre-check is enabled
    if not getattr(configurable, 'claim_pre_check', True):
        print("[LAYER3] Claim pre-check disabled, skipping")
        return {}

    notes = state.get("notes", [])
    source_store = state.get("source_store", [])

    if not notes:
        print("[LAYER3] No notes to check")
        return {}

    findings = "\n".join(notes)

    # Truncate if too long (save tokens)
    if len(findings) > 30000:
        findings = findings[:30000] + "\n...[truncated]"

    # Step 1: Extract claims via LLM
    print("[LAYER3] Extracting key claims from research notes...")

    # Use evaluation model (gpt-4.1-mini by default) for fast, cheap extraction
    extraction_model = getattr(configurable, 'evaluation_model', 'openai:gpt-4.1-mini')
    model_config = {
        "model": extraction_model,
        "api_key": get_api_key_for_model(extraction_model, config),
        "tags": ["langsmith:nostream", "layer3:extract"]
    }

    try:
        extraction_llm = configurable_model.with_config(model_config).with_structured_output(ExtractedClaims)
        prompt = CLAIM_EXTRACTION_PROMPT.format(notes=findings[:20000])

        result = await extraction_llm.ainvoke([HumanMessage(content=prompt)])
        claims = result.claims[:20]  # Limit to 20 claims

        print(f"[LAYER3] Extracted {len(claims)} claims")

    except Exception as e:
        print(f"[LAYER3] Claim extraction failed: {e}")
        return {}

    # Step 2: Verify each claim against sources
    warnings = []
    verified_count = 0

    for claim in claims:
        key_terms = extract_key_terms(claim)

        if not key_terms:
            continue  # Skip claims without verifiable terms

        is_verified, reason = verify_claim_in_sources(claim, key_terms, source_store)

        if is_verified:
            verified_count += 1
        else:
            warnings.append(f"{claim[:80]}... ({reason})")

    # Step 3: Log warnings
    if warnings:
        print(f"[LAYER3] Found {len(warnings)} unverifiable claims:")
        for w in warnings[:5]:
            print(f"[LAYER3]   - {w}")
        if len(warnings) > 5:
            print(f"[LAYER3]   ... and {len(warnings) - 5} more")
    else:
        print(f"[LAYER3] All {verified_count} claims verified against sources")

    return {
        "claim_warnings": warnings
    }
