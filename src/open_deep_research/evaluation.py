"""
Evaluation framework for Deep Research reports.

Provides claim-level verification, citation checking, and quality metrics.
This is a SEPARATE module from the pipeline - runs after report generation.

Key design decisions:
- Eval is separate from report generation (not embedded)
- Reuses existing verification.py for claim extraction and verification
- Produces structured JSON output for analysis
- I9: Eval-driven changes require human checkpoint
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.state import AgentState, SourceRecord, EvidenceSnippet
from open_deep_research.utils import get_api_key_for_model
from open_deep_research.verification import (
    extract_claims,
    find_relevant_passages,
    verify_single_claim,
    ExtractedClaims,
    VerificationVote,
)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    max_claims: int = 30
    model: str = "openai:gpt-4.1-mini"
    top_k_sources: int = 3
    verify_citations: bool = True
    check_verified_findings: bool = True


@dataclass
class ClaimResult:
    """Result for a single claim verification."""
    claim_id: str
    claim_text: str
    citation: Optional[str]  # e.g., "[9]"
    priority: str  # "high", "medium", "low"
    status: str  # "TRUE", "FALSE", "UNVERIFIABLE"
    confidence: float
    source_url: Optional[str]
    source_title: Optional[str]
    evidence_snippet: Optional[str]


@dataclass
class ClaimMetrics:
    """Aggregate metrics for claims."""
    total: int = 0
    true_count: int = 0
    false_count: int = 0
    unverifiable_count: int = 0
    hallucination_rate: float = 0.0
    grounding_rate: float = 0.0


@dataclass
class CitationMetrics:
    """Metrics for citation verification."""
    total: int = 0
    verified: int = 0
    accuracy: float = 0.0
    unique_sources: int = 0


@dataclass
class VerifiedFindingsMetrics:
    """Metrics for Verified Findings section."""
    quotes: int = 0
    all_pass: bool = True
    source_diversity: int = 0
    error: Optional[str] = None


@dataclass
class EvalResult:
    """Complete evaluation result."""
    run_id: str
    query: str
    eval_timestamp: str
    claims: ClaimMetrics
    citations: CitationMetrics
    verified_findings: VerifiedFindingsMetrics
    per_claim: List[ClaimResult]
    warnings: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "query": self.query,
            "eval_timestamp": self.eval_timestamp,
            "claims": asdict(self.claims),
            "citations": asdict(self.citations),
            "verified_findings": asdict(self.verified_findings),
            "per_claim": [asdict(c) for c in self.per_claim],
            "warnings": self.warnings,
        }

    def to_json(self, path: str) -> None:
        """Write evaluation result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[EVAL] Results saved to {path}")


def extract_citations(report: str) -> List[Dict]:
    """Extract all citations [N] from report with context.

    Returns list of {"citation_num": int, "context": str, "position": int}
    """
    pattern = r'\[(\d+)\]'
    citations = []

    for match in re.finditer(pattern, report):
        num = int(match.group(1))
        # Get surrounding context (50 chars before)
        start = max(0, match.start() - 50)
        context = report[start:match.start()].strip()
        citations.append({
            "citation_num": num,
            "context": context,
            "position": match.start()
        })

    return citations


def check_citations(report: str, sources: List[SourceRecord]) -> CitationMetrics:
    """Verify citations reference valid sources.

    Args:
        report: The final report text
        sources: List of source records

    Returns:
        CitationMetrics with verification results
    """
    citations = extract_citations(report)

    if not citations:
        return CitationMetrics(total=0, verified=0, accuracy=1.0, unique_sources=0)

    # Build source index (1-indexed as they appear in Sources section)
    # Note: This assumes sources are numbered in order in the report
    num_sources = len(sources)

    verified = 0
    for cit in citations:
        # Citation is valid if it's within range of available sources
        if 1 <= cit["citation_num"] <= num_sources:
            verified += 1

    unique_cited = len(set(c["citation_num"] for c in citations))

    return CitationMetrics(
        total=len(citations),
        verified=verified,
        accuracy=verified / len(citations) if citations else 1.0,
        unique_sources=unique_cited
    )


def check_verified_findings(state: dict) -> VerifiedFindingsMetrics:
    """Check Verified Findings section integrity.

    Validates that:
    - Section exists
    - All quotes are from PASS snippets
    - Source diversity is maintained
    """
    report = state.get("final_report", "")
    snippets = state.get("evidence_snippets", [])

    # Extract VF section
    vf_match = re.search(
        r'## Verified Findings\n(.*?)(?=\n## |\Z)',
        report,
        re.DOTALL
    )

    if not vf_match:
        return VerifiedFindingsMetrics(
            quotes=0,
            all_pass=False,
            source_diversity=0,
            error="Verified Findings section not found"
        )

    vf_section = vf_match.group(1)

    # Count quotes in section (look for quoted text)
    quotes = re.findall(r'"([^"]+)"', vf_section)

    # Check all are PASS status
    pass_snippets = [s for s in snippets if s.get("status") == "PASS"]
    pass_quotes = {s["quote"] for s in pass_snippets}

    # Fuzzy matching - quote might be slightly different
    all_pass = True
    for quote in quotes:
        # Check if quote (or close version) exists in pass_quotes
        found = False
        for pq in pass_quotes:
            if quote in pq or pq in quote:
                found = True
                break
        if not found:
            all_pass = False
            break

    # Check source diversity
    sources_in_vf = set()
    for snippet in pass_snippets:
        if snippet["quote"] in vf_section or any(q in snippet["quote"] for q in quotes):
            sources_in_vf.add(snippet.get("url", snippet.get("source_id", "")))

    return VerifiedFindingsMetrics(
        quotes=len(quotes),
        all_pass=all_pass,
        source_diversity=len(sources_in_vf)
    )


def prioritize_claim(claim: str, citation: Optional[str]) -> str:
    """Assign priority to a claim for triage.

    High priority:
    - Claims with numbers/dates/percentages
    - Uncited claims
    - Claims with proper nouns
    """
    # Check for numbers, percentages, dates
    has_numbers = bool(re.search(r'\d+(?:\.\d+)?%?', claim))
    has_dollar = bool(re.search(r'\$[\d,]+', claim))
    has_date = bool(re.search(r'\b(?:19|20)\d{2}\b', claim))

    # Check for proper nouns (capitalized words not at start)
    words = claim.split()
    has_proper_nouns = any(
        w[0].isupper() and i > 0
        for i, w in enumerate(words)
        if len(w) > 2
    )

    # Uncited claims are high priority
    is_uncited = citation is None

    if is_uncited or has_numbers or has_dollar or has_date:
        return "high"
    elif has_proper_nouns:
        return "medium"
    else:
        return "low"


def extract_claim_citation(claim: str, report: str) -> Optional[str]:
    """Find the citation associated with a claim in the report."""
    # Look for the claim text (or close match) and nearby citation
    claim_start = report.find(claim[:50])  # First 50 chars
    if claim_start == -1:
        return None

    # Look for citation within 20 chars after claim
    search_region = report[claim_start:claim_start + len(claim) + 20]
    citation_match = re.search(r'\[(\d+)\]', search_region)

    if citation_match:
        return f"[{citation_match.group(1)}]"
    return None


def map_verification_status(status: str) -> str:
    """Map verification status to eval status.

    Verification uses: SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, UNCERTAIN
    Eval uses: TRUE, FALSE, UNVERIFIABLE
    """
    mapping = {
        "SUPPORTED": "TRUE",
        "PARTIALLY_SUPPORTED": "TRUE",  # Count as grounded
        "UNSUPPORTED": "FALSE",
        "UNCERTAIN": "UNVERIFIABLE",
    }
    return mapping.get(status, "UNVERIFIABLE")


async def evaluate_report(
    state: dict,
    config: EvalConfig = None,
    runnable_config: RunnableConfig = None,
) -> EvalResult:
    """Run full evaluation on a completed research report.

    This is the main entry point for evaluation.

    Args:
        state: Final agent state with final_report, source_store, evidence_snippets
        config: Evaluation configuration
        runnable_config: LangGraph runnable config for model initialization

    Returns:
        EvalResult with all metrics and per-claim details
    """
    config = config or EvalConfig()
    warnings = []

    # Extract key data from state
    report = state.get("final_report", "")
    sources = state.get("source_store", [])
    query = ""

    # Try to extract query from messages
    messages = state.get("messages", [])
    if messages:
        for msg in messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                if len(msg.content) < 500:  # Likely the query
                    query = msg.content
                    break

    if not report:
        warnings.append("No report found in state")
        return EvalResult(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            query=query,
            eval_timestamp=datetime.now().isoformat(),
            claims=ClaimMetrics(),
            citations=CitationMetrics(),
            verified_findings=VerifiedFindingsMetrics(error="No report"),
            per_claim=[],
            warnings=warnings,
        )

    print(f"[EVAL] Starting evaluation...")
    print(f"[EVAL] Report length: {len(report)} chars")
    print(f"[EVAL] Sources available: {len(sources)}")

    # Initialize LLM for claim extraction and verification
    if runnable_config:
        configurable = Configuration.from_runnable_config(runnable_config)
        api_key = get_api_key_for_model(config.model, runnable_config)
    else:
        # Default initialization
        import os
        api_key = os.environ.get("OPENAI_API_KEY")

    extraction_llm = init_chat_model(
        model=config.model,
        api_key=api_key,
        tags=["langsmith:nostream", "eval:extraction"]
    ).with_structured_output(ExtractedClaims)

    verification_llm = init_chat_model(
        model=config.model,
        api_key=api_key,
        tags=["langsmith:nostream", "eval:verify"]
    ).with_structured_output(VerificationVote)

    # Step 1: Extract claims
    print(f"[EVAL] Extracting claims...")
    claims = await extract_claims(report, extraction_llm)
    claims = claims[:config.max_claims]
    print(f"[EVAL] Extracted {len(claims)} claims")

    if not claims:
        warnings.append("No claims extracted from report")

    # Step 2: Verify each claim
    print(f"[EVAL] Verifying claims...")
    per_claim_results = []

    for i, claim_text in enumerate(claims):
        claim_id = f"c{i+1:03d}"
        citation = extract_claim_citation(claim_text, report)
        priority = prioritize_claim(claim_text, citation)

        # Find relevant sources using embeddings
        try:
            sources_with_passages = await find_relevant_passages(
                claim_text,
                sources,
                top_k=config.top_k_sources
            )
        except Exception as e:
            logging.warning(f"[EVAL] Source matching failed for {claim_id}: {e}")
            sources_with_passages = []

        # Verify against best source
        if sources_with_passages:
            source, passage, score = sources_with_passages[0]
            verification = await verify_single_claim(
                claim_text, passage, source, verification_llm, claim_id
            )

            result = ClaimResult(
                claim_id=claim_id,
                claim_text=claim_text,
                citation=citation,
                priority=priority,
                status=map_verification_status(verification["status"]),
                confidence=verification["confidence"],
                source_url=verification.get("source_url"),
                source_title=verification.get("source_title"),
                evidence_snippet=verification.get("evidence_snippet"),
            )
        else:
            result = ClaimResult(
                claim_id=claim_id,
                claim_text=claim_text,
                citation=citation,
                priority=priority,
                status="UNVERIFIABLE",
                confidence=0.0,
                source_url=None,
                source_title=None,
                evidence_snippet="No matching source found",
            )

        per_claim_results.append(result)

        # Progress logging
        if (i + 1) % 10 == 0:
            print(f"[EVAL] Verified {i + 1}/{len(claims)} claims")

    # Step 3: Aggregate claim metrics
    true_count = sum(1 for r in per_claim_results if r.status == "TRUE")
    false_count = sum(1 for r in per_claim_results if r.status == "FALSE")
    unverifiable_count = sum(1 for r in per_claim_results if r.status == "UNVERIFIABLE")
    total = len(per_claim_results)

    claim_metrics = ClaimMetrics(
        total=total,
        true_count=true_count,
        false_count=false_count,
        unverifiable_count=unverifiable_count,
        hallucination_rate=false_count / total if total > 0 else 0.0,
        grounding_rate=true_count / total if total > 0 else 0.0,
    )

    # Add warnings for quality issues
    if claim_metrics.hallucination_rate > 0.02:
        warnings.append(f"Hallucination rate ({claim_metrics.hallucination_rate:.1%}) exceeds 2% target")
    if claim_metrics.grounding_rate < 0.85:
        warnings.append(f"Grounding rate ({claim_metrics.grounding_rate:.1%}) below 85% target")

    # Step 4: Check citations
    print(f"[EVAL] Checking citations...")
    citation_metrics = CitationMetrics() if not config.verify_citations else check_citations(report, sources)

    if citation_metrics.accuracy < 0.90:
        warnings.append(f"Citation accuracy ({citation_metrics.accuracy:.1%}) below 90% target")

    # Step 5: Check Verified Findings
    print(f"[EVAL] Checking Verified Findings...")
    vf_metrics = VerifiedFindingsMetrics() if not config.check_verified_findings else check_verified_findings(state)

    if vf_metrics.error:
        warnings.append(f"Verified Findings issue: {vf_metrics.error}")
    if not vf_metrics.all_pass:
        warnings.append("Verified Findings contains non-PASS quotes")

    # Build final result
    result = EvalResult(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        query=query,
        eval_timestamp=datetime.now().isoformat(),
        claims=claim_metrics,
        citations=citation_metrics,
        verified_findings=vf_metrics,
        per_claim=per_claim_results,
        warnings=warnings,
    )

    # Print summary
    print(f"\n[EVAL] ========== EVALUATION COMPLETE ==========")
    print(f"[EVAL] Claims: {true_count}/{total} TRUE ({claim_metrics.grounding_rate:.0%} grounding)")
    print(f"[EVAL] Hallucination rate: {claim_metrics.hallucination_rate:.1%} (target: <2%)")
    print(f"[EVAL] Citations: {citation_metrics.verified}/{citation_metrics.total} verified ({citation_metrics.accuracy:.0%})")
    print(f"[EVAL] Verified Findings: {vf_metrics.quotes} quotes, diversity={vf_metrics.source_diversity}")
    if warnings:
        print(f"[EVAL] Warnings: {len(warnings)}")
        for w in warnings:
            print(f"[EVAL]   - {w}")
    print(f"[EVAL] ============================================\n")

    return result
