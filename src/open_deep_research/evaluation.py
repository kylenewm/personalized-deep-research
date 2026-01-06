"""
Evaluation framework for Deep Research reports.

CITATION-FIRST APPROACH:
- For claims WITH citations [N], verify against the cited source directly (fast, accurate)
- For claims with MULTIPLE citations [1][2], verify against ALL cited sources
- For UNCITED claims, flag as high-risk and use embedding search (fallback)

This is SEPARATE from the pipeline - runs as post-hoc quality check.

Quality Targets:
- Hallucination rate: <2%
- Grounding rate: >85%
- Citation accuracy: >90%
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.state import SourceRecord
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
    verify_citations: bool = True
    check_verified_findings: bool = True
    dry_run: bool = False  # If True, only estimate cost without API calls
    parallel_batch_size: int = 5  # Claims to verify in parallel
    fallback_to_embedding: bool = True  # For uncited claims, search all sources


@dataclass
class ClaimResult:
    """Result for a single claim verification."""
    claim_id: str
    claim_text: str
    citations: List[int]  # e.g., [1, 3] for "[1][3]"
    is_uncited: bool
    status: str  # "TRUE", "FALSE", "UNVERIFIABLE"
    confidence: float
    sources_checked: List[str]  # URLs of sources checked
    evidence_snippet: Optional[str]


@dataclass
class ClaimMetrics:
    """Aggregate metrics for claims."""
    total: int = 0
    true_count: int = 0
    false_count: int = 0
    unverifiable_count: int = 0
    uncited_count: int = 0  # High-risk: factual claims without citations
    hallucination_rate: float = 0.0
    grounding_rate: float = 0.0


@dataclass
class CitationMetrics:
    """Metrics for citation verification."""
    total: int = 0
    valid: int = 0  # Citation points to existing source
    supported: int = 0  # Cited source actually supports the claim
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
class CostEstimate:
    """Cost estimate for evaluation."""
    extraction_calls: int = 1
    verification_calls: int = 0
    embedding_calls: int = 0
    estimated_cost_usd: float = 0.0

    def __str__(self):
        return (f"Extraction: {self.extraction_calls} call (~$0.01)\n"
                f"Verification: {self.verification_calls} calls (~${self.verification_calls * 0.01:.2f})\n"
                f"Embedding: {self.embedding_calls} calls (~${self.embedding_calls * 0.001:.3f})\n"
                f"TOTAL: ~${self.estimated_cost_usd:.2f}")


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
    cost_estimate: Optional[CostEstimate] = None

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
            "cost_estimate": asdict(self.cost_estimate) if self.cost_estimate else None,
        }

    def to_json(self, path: str) -> None:
        """Write evaluation result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[EVAL] Results saved to {path}")


def parse_sources_section(report: str) -> Dict[int, Dict]:
    """Parse the Sources section to build citation_num -> source info mapping.

    Expected format:
    ## Sources
    [1] Title, Author: https://example.com
    [2] Another Title: https://example2.com

    Handles titles with colons (e.g., "From Labs to Policy: IMDA's Journey")

    Returns: {1: {"title": "...", "url": "..."}, 2: {...}}
    """
    citation_map = {}

    # Find Sources section
    sources_match = re.search(r'## Sources\n(.*?)(?=\n## |\Z)', report, re.DOTALL)
    if not sources_match:
        return citation_map

    sources_text = sources_match.group(1)

    # Parse each citation line: [N] Title...: URL
    # Use greedy match for title to handle titles with colons
    # The pattern will match the LAST colon before https://
    pattern = r'\[(\d+)\]\s*(.+):\s*(https?://[^\s\n]+)'
    for match in re.finditer(pattern, sources_text):
        num = int(match.group(1))
        title = match.group(2).strip()
        url = match.group(3).strip()
        citation_map[num] = {"title": title, "url": url}

    return citation_map


def extract_citations_from_claim(claim: str, report: str) -> List[int]:
    """Extract citation numbers from a claim by finding it in the report.

    Strategy: Find the paragraph containing the claim, then extract citations
    from that paragraph (citations often appear at end of paragraphs).

    Returns list of citation numbers, e.g., [1, 3] for "...fact [1][3]..."
    """
    # Remove Sources section to avoid false matches
    sources_idx = report.find('## Sources')
    if sources_idx > 0:
        report_body = report[:sources_idx]
    else:
        report_body = report

    # Strategy 1: Find paragraph containing the claim
    # Split by double newlines (paragraphs) or bullet points
    paragraphs = re.split(r'\n\n+|\n-\s+', report_body)

    claim_lower = claim.lower()
    claim_words = set(w.lower() for w in claim.split() if len(w) > 4)

    for para in paragraphs:
        para_lower = para.lower()

        # Check if this paragraph contains enough of the claim's key words
        word_matches = sum(1 for w in claim_words if w in para_lower)

        if word_matches >= min(5, len(claim_words) * 0.5):
            # Found a matching paragraph - extract citations from it
            citations = [int(m) for m in re.findall(r'\[(\d+)\]', para)]
            if citations:
                # Limit to reasonable number (paragraph shouldn't have >5 citations)
                if len(citations) <= 5:
                    return list(set(citations))

    # Strategy 2: Exact substring match
    if claim[:40] in report_body:
        idx = report_body.find(claim[:40])
        # Look for citations within 200 chars after the claim starts
        region = report_body[idx:idx + 300]
        citations = [int(m) for m in re.findall(r'\[(\d+)\]', region)]
        if citations and len(citations) <= 5:
            return list(set(citations))

    return []


def find_source_by_url(url: str, sources: List[SourceRecord]) -> Optional[SourceRecord]:
    """Find a source in source_store by URL (fuzzy match)."""
    url_clean = url.rstrip('/').lower()
    for source in sources:
        source_url = source.get("url", "").rstrip('/').lower()
        if source_url == url_clean or url_clean in source_url or source_url in url_clean:
            return source
    return None


def estimate_cost(num_claims: int, num_uncited: int, config: EvalConfig) -> CostEstimate:
    """Estimate cost before running evaluation."""
    # 1 extraction call
    extraction_calls = 1

    # Verification: 1 call per claim (cited claims check cited source, uncited use embedding)
    verification_calls = num_claims

    # Embedding calls only for uncited claims (if fallback enabled)
    embedding_calls = num_uncited if config.fallback_to_embedding else 0

    # Cost estimates (gpt-4.1-mini)
    # Extraction: ~2k input + 1k output = ~$0.01
    # Verification: ~1k input + 100 output = ~$0.005 each
    # Embedding: ~$0.0001 per call
    estimated_cost = 0.01 + (verification_calls * 0.005) + (embedding_calls * 0.001)

    return CostEstimate(
        extraction_calls=extraction_calls,
        verification_calls=verification_calls,
        embedding_calls=embedding_calls,
        estimated_cost_usd=round(estimated_cost, 3)
    )


def check_verified_findings(state: dict) -> VerifiedFindingsMetrics:
    """Check Verified Findings section integrity."""
    report = state.get("final_report", "")
    snippets = state.get("evidence_snippets", [])

    vf_match = re.search(r'## Verified Findings\n(.*?)(?=\n## |\Z)', report, re.DOTALL)
    if not vf_match:
        return VerifiedFindingsMetrics(quotes=0, all_pass=False, source_diversity=0,
                                        error="Verified Findings section not found")

    vf_section = vf_match.group(1)
    quotes = re.findall(r'"([^"]+)"', vf_section)

    pass_snippets = [s for s in snippets if s.get("status") == "PASS"]
    pass_quotes = {s["quote"] for s in pass_snippets}

    all_pass = True
    for quote in quotes:
        if not any(quote in pq or pq in quote for pq in pass_quotes):
            all_pass = False
            break

    sources_in_vf = set()
    for snippet in pass_snippets:
        if snippet["quote"] in vf_section:
            sources_in_vf.add(snippet.get("url", ""))

    return VerifiedFindingsMetrics(
        quotes=len(quotes), all_pass=all_pass, source_diversity=len(sources_in_vf)
    )


def map_verification_status(status: str) -> str:
    """Map verification status to eval status."""
    mapping = {
        "SUPPORTED": "TRUE",
        "PARTIALLY_SUPPORTED": "TRUE",
        "UNSUPPORTED": "FALSE",
        "UNCERTAIN": "UNVERIFIABLE",
    }
    return mapping.get(status, "UNVERIFIABLE")


async def verify_claim_against_source(
    claim: str,
    source: SourceRecord,
    llm,
    claim_id: str
) -> Tuple[str, float, str]:
    """Verify a claim against a specific source.

    Returns: (status, confidence, evidence_snippet)
    """
    content = source.get("content", "")
    if not content:
        return "UNVERIFIABLE", 0.0, "Source has no content"

    # Extract best paragraph from source for this claim
    claim_words = set(w.lower() for w in claim.split() if len(w) > 3)
    paragraphs = content.split('\n\n')

    best_para = ""
    best_score = 0
    for para in paragraphs:
        if len(para) < 50:
            continue
        score = sum(1 for w in claim_words if w in para.lower())
        if score > best_score:
            best_score = score
            best_para = para

    passage = best_para[:1500] if best_para else content[:1500]

    result = await verify_single_claim(claim, passage, source, llm, claim_id)
    return (
        result.get("status", "UNCERTAIN"),
        result.get("confidence", 0.0),
        result.get("evidence_snippet", "")
    )


async def verify_single_claim_task(
    claim_text: str,
    claim_id: str,
    citations: List[int],
    citation_map: Dict[int, Dict],
    sources: List[SourceRecord],
    llm,
    config: EvalConfig
) -> ClaimResult:
    """Verify a single claim - citation-first approach.

    If claim has citations: verify against cited sources
    If uncited: flag as high-risk, optionally search all sources
    """
    is_uncited = len(citations) == 0
    sources_checked = []
    best_status = "UNVERIFIABLE"
    best_confidence = 0.0
    best_evidence = ""

    if not is_uncited:
        # CITATION-FIRST: Check the cited sources directly
        for cit_num in citations:
            if cit_num not in citation_map:
                continue

            cit_info = citation_map[cit_num]
            source = find_source_by_url(cit_info["url"], sources)

            if not source:
                continue

            sources_checked.append(cit_info["url"])
            status, confidence, evidence = await verify_claim_against_source(
                claim_text, source, llm, claim_id
            )

            # Keep best result (prefer SUPPORTED > PARTIAL > others)
            if status == "SUPPORTED" and confidence > best_confidence:
                best_status = status
                best_confidence = confidence
                best_evidence = evidence
            elif best_status != "SUPPORTED" and confidence > best_confidence:
                best_status = status
                best_confidence = confidence
                best_evidence = evidence

    elif config.fallback_to_embedding:
        # UNCITED: Use embedding search as fallback
        try:
            results = await find_relevant_passages(claim_text, sources, top_k=1)
            if results:
                source, passage, score = results[0]
                sources_checked.append(source.get("url", ""))
                status, confidence, evidence = await verify_claim_against_source(
                    claim_text, source, llm, claim_id
                )
                best_status = status
                best_confidence = confidence
                best_evidence = evidence
        except Exception as e:
            logging.warning(f"[EVAL] Embedding search failed for {claim_id}: {e}")

    return ClaimResult(
        claim_id=claim_id,
        claim_text=claim_text,
        citations=citations,
        is_uncited=is_uncited,
        status=map_verification_status(best_status),
        confidence=best_confidence,
        sources_checked=sources_checked,
        evidence_snippet=best_evidence[:200] if best_evidence else None
    )


async def evaluate_report(
    state: dict,
    config: EvalConfig = None,
    runnable_config: RunnableConfig = None,
) -> EvalResult:
    """Run evaluation on a completed research report.

    CITATION-FIRST APPROACH:
    1. Extract claims from report
    2. For each claim, find its citations [N] in the report
    3. Verify against the cited source(s) directly
    4. For uncited claims, flag as high-risk
    """
    config = config or EvalConfig()
    warnings = []

    report = state.get("final_report", "")
    sources = state.get("source_store", [])

    # Extract query from messages
    query = ""
    messages = state.get("messages", [])
    if messages and isinstance(messages, list) and len(messages) > 0:
        first_msg = messages[0]
        if isinstance(first_msg, dict):
            query = first_msg.get("content", "")[:200]
        elif hasattr(first_msg, 'content'):
            query = str(first_msg.content)[:200]

    if not report:
        return EvalResult(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            query=query, eval_timestamp=datetime.now().isoformat(),
            claims=ClaimMetrics(), citations=CitationMetrics(),
            verified_findings=VerifiedFindingsMetrics(error="No report"),
            per_claim=[], warnings=["No report found in state"]
        )

    print(f"[EVAL] Starting evaluation...")
    print(f"[EVAL] Report: {len(report)} chars | Sources: {len(sources)}")

    # Parse Sources section to build citation -> URL mapping
    citation_map = parse_sources_section(report)
    print(f"[EVAL] Parsed {len(citation_map)} citations from Sources section")

    # Initialize LLM
    import os
    api_key = os.environ.get("OPENAI_API_KEY")

    extraction_llm = init_chat_model(
        model=config.model, api_key=api_key,
        tags=["langsmith:nostream", "eval:extraction"]
    ).with_structured_output(ExtractedClaims)

    verification_llm = init_chat_model(
        model=config.model, api_key=api_key,
        tags=["langsmith:nostream", "eval:verify"]
    ).with_structured_output(VerificationVote)

    # Step 1: Extract claims
    print(f"[EVAL] Extracting claims...")
    claims = await extract_claims(report, extraction_llm)
    claims = claims[:config.max_claims]
    print(f"[EVAL] Extracted {len(claims)} claims")

    # Pre-compute citations for each claim
    claims_with_citations = []
    for claim in claims:
        citations = extract_citations_from_claim(claim, report)
        claims_with_citations.append((claim, citations))

    uncited_count = sum(1 for _, cits in claims_with_citations if not cits)
    print(f"[EVAL] Claims with citations: {len(claims) - uncited_count} | Uncited (high-risk): {uncited_count}")

    # Cost estimate
    cost_est = estimate_cost(len(claims), uncited_count, config)
    print(f"[EVAL] Estimated cost: ~${cost_est.estimated_cost_usd:.2f}")

    if config.dry_run:
        print(f"[EVAL] DRY RUN - not executing verification")
        print(cost_est)
        return EvalResult(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            query=query, eval_timestamp=datetime.now().isoformat(),
            claims=ClaimMetrics(total=len(claims), uncited_count=uncited_count),
            citations=CitationMetrics(total=len(citation_map)),
            verified_findings=VerifiedFindingsMetrics(),
            per_claim=[], warnings=["DRY RUN - no verification performed"],
            cost_estimate=cost_est
        )

    # Step 2: Verify claims in parallel batches
    print(f"[EVAL] Verifying claims (batch size: {config.parallel_batch_size})...")
    per_claim_results = []

    for batch_start in range(0, len(claims_with_citations), config.parallel_batch_size):
        batch = claims_with_citations[batch_start:batch_start + config.parallel_batch_size]

        tasks = [
            verify_single_claim_task(
                claim_text=claim,
                claim_id=f"c{batch_start + i + 1:03d}",
                citations=citations,
                citation_map=citation_map,
                sources=sources,
                llm=verification_llm,
                config=config
            )
            for i, (claim, citations) in enumerate(batch)
        ]

        batch_results = await asyncio.gather(*tasks)
        per_claim_results.extend(batch_results)
        print(f"[EVAL] Verified {len(per_claim_results)}/{len(claims)} claims")

    # Step 3: Aggregate metrics
    true_count = sum(1 for r in per_claim_results if r.status == "TRUE")
    false_count = sum(1 for r in per_claim_results if r.status == "FALSE")
    unverifiable_count = sum(1 for r in per_claim_results if r.status == "UNVERIFIABLE")
    total = len(per_claim_results)

    claim_metrics = ClaimMetrics(
        total=total,
        true_count=true_count,
        false_count=false_count,
        unverifiable_count=unverifiable_count,
        uncited_count=uncited_count,
        hallucination_rate=false_count / total if total > 0 else 0.0,
        grounding_rate=true_count / total if total > 0 else 0.0,
    )

    # Citation metrics
    cited_claims = [r for r in per_claim_results if not r.is_uncited]
    supported_citations = sum(1 for r in cited_claims if r.status == "TRUE")

    citation_metrics = CitationMetrics(
        total=sum(len(r.citations) for r in per_claim_results),
        valid=len(citation_map),
        supported=supported_citations,
        accuracy=supported_citations / len(cited_claims) if cited_claims else 1.0,
        unique_sources=len(set(c for r in per_claim_results for c in r.citations))
    )

    # Warnings
    if claim_metrics.hallucination_rate > 0.02:
        warnings.append(f"Hallucination rate ({claim_metrics.hallucination_rate:.1%}) exceeds 2% target")
    if claim_metrics.grounding_rate < 0.85:
        warnings.append(f"Grounding rate ({claim_metrics.grounding_rate:.1%}) below 85% target")
    if uncited_count > 0:
        warnings.append(f"{uncited_count} uncited factual claims (high-risk)")

    # Check Verified Findings
    vf_metrics = check_verified_findings(state) if config.check_verified_findings else VerifiedFindingsMetrics()

    result = EvalResult(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        query=query,
        eval_timestamp=datetime.now().isoformat(),
        claims=claim_metrics,
        citations=citation_metrics,
        verified_findings=vf_metrics,
        per_claim=per_claim_results,
        warnings=warnings,
        cost_estimate=cost_est
    )

    # Print summary
    print(f"\n[EVAL] ========== EVALUATION COMPLETE ==========")
    print(f"[EVAL] Claims: {true_count}/{total} TRUE ({claim_metrics.grounding_rate:.0%} grounding)")
    print(f"[EVAL] Hallucination rate: {claim_metrics.hallucination_rate:.1%} (target: <2%)")
    print(f"[EVAL] Uncited claims: {uncited_count} (high-risk)")
    print(f"[EVAL] Citation accuracy: {citation_metrics.accuracy:.0%}")
    print(f"[EVAL] Actual cost: ~${cost_est.estimated_cost_usd:.2f}")
    if warnings:
        print(f"[EVAL] Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"[EVAL]   - {w}")
    print(f"[EVAL] ============================================\n")

    return result
