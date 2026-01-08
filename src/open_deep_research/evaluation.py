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
class CitationValidityResult:
    """Result for a single citation verification (deterministic approach).

    This is used by the citation-first evaluation which parses [N] citations
    directly from the report, avoiding LLM extraction paraphrasing issues.
    """
    citation_num: int
    context: str  # The sentence/clause containing the citation
    status: str  # "VALID", "INVALID", "MISSING_SOURCE", "NO_CONTENT"
    source_url: Optional[str] = None
    confidence: float = 0.0
    evidence_snippet: Optional[str] = None


@dataclass
class CitationValidityMetrics:
    """Aggregate metrics for citation validity evaluation."""
    total_citations: int = 0
    unique_citations: int = 0  # Unique [N] numbers used
    valid_count: int = 0
    invalid_count: int = 0
    missing_source_count: int = 0
    no_content_count: int = 0
    validity_rate: float = 0.0  # valid / (valid + invalid)
    error_rate: float = 0.0  # invalid / total

    def __str__(self):
        return (f"Citation Validity: {self.valid_count}/{self.valid_count + self.invalid_count} "
                f"({self.validity_rate:.1%})\n"
                f"Errors: {self.invalid_count} invalid, {self.missing_source_count} missing sources")


@dataclass
class CoverageResult:
    """Citation coverage metrics (separate from validity).

    This measures what percentage of sentences have citations,
    NOT whether those citations are valid. This is a quality metric.
    """
    total_sentences: int = 0
    cited_sentences: int = 0
    uncited_factual: List[str] = None  # Sentences without citations that look factual
    coverage_rate: float = 0.0

    def __post_init__(self):
        if self.uncited_factual is None:
            self.uncited_factual = []

    def __str__(self):
        return (f"Coverage: {self.cited_sentences}/{self.total_sentences} sentences cited "
                f"({self.coverage_rate:.1%})\n"
                f"Uncited factual: {len(self.uncited_factual)} statements")


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

    Handles TWO formats:

    Format 1 (Markdown links - current report format):
    ## Sources
    1. [Title](https://example.com)
    2. [Another Title](https://example2.com)

    Format 2 (Legacy format):
    ## Sources
    [1] Title: https://example.com
    [2] Another Title: https://example2.com

    Returns: {1: {"title": "...", "url": "..."}, 2: {...}}
    """
    citation_map = {}

    # Find Sources section
    sources_match = re.search(r'## Sources\n(.*?)(?=\n## |\Z)', report, re.DOTALL)
    if not sources_match:
        return citation_map

    sources_text = sources_match.group(1)

    # Try Format 1 first: N. [Title](URL) - markdown links
    # Pattern: number, dot, optional whitespace, [title], (url)
    md_pattern = r'(\d+)\.\s*\[([^\]]+)\]\((https?://[^)]+)\)'
    for match in re.finditer(md_pattern, sources_text):
        num = int(match.group(1))
        title = match.group(2).strip()
        url = match.group(3).strip()
        citation_map[num] = {"title": title, "url": url}

    # If Format 1 found matches, return them
    if citation_map:
        return citation_map

    # Fallback to Format 2: [N] Title: URL - legacy format
    legacy_pattern = r'\[(\d+)\]\s*(.+):\s*(https?://[^\s\n]+)'
    for match in re.finditer(legacy_pattern, sources_text):
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
    # Handle empty claim
    if not claim or not claim.strip():
        return []

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
    # English stopwords from NLTK (v3.8.1)
    # Source: https://www.nltk.org/nltk_data/ (stopwords corpus)
    # These are function words with no semantic content for paragraph matching
    # Using full NLTK list rather than arbitrary subset for reproducibility
    STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
        'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
        'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
        'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    }
    # Use regex to extract words without punctuation (split() keeps punctuation attached)
    claim_tokens = re.findall(r'\b\w+\b', claim.lower())
    claim_words = set(w for w in claim_tokens if len(w) > 1 and w not in STOPWORDS)

    # Require minimum 2 words to match (avoid matching with nothing)
    if not claim_words:
        # Fallback to substring match only
        pass
    else:
        for para in paragraphs:
            para_lower = para.lower()

            # Check if this paragraph contains enough of the claim's key words
            word_matches = sum(1 for w in claim_words if w in para_lower)

            # Require at least 2 absolute matches AND 50% of words
            min_required = max(2, int(len(claim_words) * 0.5))
            if word_matches >= min_required:
                # Found a matching paragraph - extract citations from it
                citations = [int(m) for m in re.findall(r'\[(\d+)\]', para)]
                if citations:
                    # Limit to reasonable number (paragraph shouldn't have >10 citations)
                    if len(citations) <= 10:
                        return list(set(citations))

    # Strategy 2: Exact substring match (for short claims or fallback)
    if len(claim) >= 10 and claim[:40] in report_body:
        idx = report_body.find(claim[:40])
        # Look for citations within 200 chars after the claim starts
        region = report_body[idx:idx + 300]
        citations = [int(m) for m in re.findall(r'\[(\d+)\]', region)]
        if citations and len(citations) <= 10:
            return list(set(citations))

    # Strategy 3: Fuzzy match for very short claims
    # Look for exact phrase match of the whole claim
    if claim_lower in report_body.lower():
        idx = report_body.lower().find(claim_lower)
        region = report_body[max(0, idx-50):idx + len(claim) + 100]
        citations = [int(m) for m in re.findall(r'\[(\d+)\]', region)]
        if citations and len(citations) <= 10:
            return list(set(citations))

    return []


def find_source_by_url(url: str, sources: List[SourceRecord]) -> Optional[SourceRecord]:
    """Find a source in source_store by URL (fuzzy match).

    If multiple sources match the same URL, returns the one with the longest content.
    This handles the duplicate source issue where truncated and full versions coexist.
    """
    url_clean = url.rstrip('/').lower()
    matches = []

    for source in sources:
        source_url = source.get("url", "").rstrip('/').lower()
        if source_url == url_clean or url_clean in source_url or source_url in url_clean:
            matches.append(source)

    if not matches:
        return None

    # Return the source with the longest content (prefer full over truncated)
    return max(matches, key=lambda s: len(s.get("content", "")))


# =============================================================================
# CITATION-FIRST EVALUATION (Deterministic Approach)
# =============================================================================
# These functions implement a deterministic evaluation approach that:
# 1. Parses [N] citations directly from the report (no LLM extraction)
# 2. Verifies each citation against its corresponding source
# 3. Separates validity (are citations correct?) from coverage (do we cite enough?)


def extract_citation_context(report: str, citation_pos: int, context_chars: int = 200) -> str:
    """Extract the sentence/clause containing a citation.

    Args:
        report: Full report text
        citation_pos: Position of the '[' in the citation [N]
        context_chars: How many chars before/after to consider

    Returns:
        The sentence or clause containing the citation
    """
    # Get surrounding context
    start = max(0, citation_pos - context_chars)
    end = min(len(report), citation_pos + context_chars)
    context = report[start:end]

    # Find sentence boundaries within context
    # Look for sentence-ending punctuation followed by space/newline
    sentences = re.split(r'(?<=[.!?])\s+', context)

    # Find which sentence contains our citation
    rel_pos = citation_pos - start
    char_count = 0
    for sent in sentences:
        if char_count <= rel_pos < char_count + len(sent) + 1:
            # Clean up the sentence
            sent = sent.strip()
            # Remove the citation markers for cleaner context
            sent_clean = re.sub(r'\[\d+\]', '', sent).strip()
            # Skip if this is a header line
            if sent_clean.startswith('#'):
                continue
            return sent_clean if sent_clean else sent
        char_count += len(sent) + 1  # +1 for the split character

    # Fallback: return the whole context cleaned up, without headers
    context_clean = re.sub(r'\[\d+\]', '', context).strip()
    # Remove header lines from context
    lines = context_clean.split('\n')
    non_header_lines = [l for l in lines if not l.strip().startswith('#')]
    return ' '.join(non_header_lines).strip()


def verify_text_against_source(text: str, source: SourceRecord) -> Tuple[str, float, str]:
    """Deterministically verify if source content supports the text.

    Uses substring and keyword matching (no LLM) for fast verification.

    Args:
        text: The text/claim to verify
        source: Source record with 'content' field

    Returns:
        Tuple of (status, confidence, evidence_snippet)
        status: "VALID", "INVALID", or "NO_CONTENT"
    """
    content = source.get("content", "")
    if not content:
        return "NO_CONTENT", 0.0, ""

    content_lower = content.lower()
    text_lower = text.lower()

    # Strategy 1: Direct substring match (high confidence)
    if text_lower in content_lower:
        # Find the matching snippet
        idx = content_lower.find(text_lower)
        snippet = content[max(0, idx-50):idx + len(text) + 50]
        return "VALID", 1.0, snippet.strip()

    # Strategy 2: Keyword overlap (medium confidence)
    # Extract meaningful words from text (>3 chars, not stopwords)
    STOPWORDS = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                 'were', 'they', 'this', 'that', 'with', 'from', 'will', 'what'}
    text_words = set(w.lower() for w in re.findall(r'\b\w+\b', text)
                     if len(w) > 3 and w.lower() not in STOPWORDS)

    if not text_words:
        # Very short text, can't verify
        return "INVALID", 0.0, ""

    # Count how many key words appear in source
    matches = sum(1 for w in text_words if w in content_lower)
    overlap_ratio = matches / len(text_words)

    if overlap_ratio >= 0.6:  # 60% keyword overlap threshold
        # Find a relevant snippet containing most matches
        paragraphs = content.split('\n\n')
        best_para = ""
        best_score = 0
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(1 for w in text_words if w in para_lower)
            if score > best_score:
                best_score = score
                best_para = para

        return "VALID", overlap_ratio, best_para[:300].strip()

    return "INVALID", overlap_ratio, ""


def evaluate_citation_validity(
    report: str,
    sources: List[SourceRecord],
    citation_map: Optional[Dict[int, Dict]] = None
) -> Tuple[List[CitationValidityResult], CitationValidityMetrics]:
    """Evaluate validity of all [N] citations in the report.

    DETERMINISTIC: Same input → same output every time.

    This is the core of the citation-first approach:
    1. Parse all [N] citations from report body
    2. Extract the text context around each citation
    3. Verify that source N supports the context

    Args:
        report: Full report text
        sources: List of source records from source_store
        citation_map: Optional pre-parsed {N: {"url": ..., "title": ...}} mapping

    Returns:
        Tuple of (per-citation results, aggregate metrics)
    """
    results = []

    # Parse citation map if not provided
    if citation_map is None:
        citation_map = parse_sources_section(report)

    # Find report body (before Sources section)
    sources_idx = report.find("## Sources")
    if sources_idx == -1:
        sources_idx = report.find("### Sources")
    report_body = report[:sources_idx] if sources_idx != -1 else report

    # Track unique citations and their occurrences
    citation_occurrences = {}  # {N: [(pos, context), ...]}

    # Find all [N] references in report body
    for match in re.finditer(r'\[(\d+)\]', report_body):
        citation_num = int(match.group(1))
        pos = match.start()
        context = extract_citation_context(report_body, pos)

        if citation_num not in citation_occurrences:
            citation_occurrences[citation_num] = []
        citation_occurrences[citation_num].append((pos, context))

    # Verify each unique citation
    for citation_num, occurrences in sorted(citation_occurrences.items()):
        # Use the first occurrence for verification (they should all cite the same source)
        _, context = occurrences[0]

        # Get source info from citation map
        source_info = citation_map.get(citation_num)
        if not source_info:
            results.append(CitationValidityResult(
                citation_num=citation_num,
                context=context,
                status="MISSING_SOURCE",
                source_url=None,
                confidence=0.0,
                evidence_snippet=None
            ))
            continue

        source_url = source_info.get("url", "")

        # Find the actual source content
        source = find_source_by_url(source_url, sources)
        if not source or not source.get("content"):
            results.append(CitationValidityResult(
                citation_num=citation_num,
                context=context,
                status="NO_CONTENT",
                source_url=source_url,
                confidence=0.0,
                evidence_snippet=None
            ))
            continue

        # Verify the context against the source
        status, confidence, evidence = verify_text_against_source(context, source)

        results.append(CitationValidityResult(
            citation_num=citation_num,
            context=context,
            status=status,
            source_url=source_url,
            confidence=confidence,
            evidence_snippet=evidence
        ))

    # Calculate metrics
    total = len(results)
    valid = sum(1 for r in results if r.status == "VALID")
    invalid = sum(1 for r in results if r.status == "INVALID")
    missing = sum(1 for r in results if r.status == "MISSING_SOURCE")
    no_content = sum(1 for r in results if r.status == "NO_CONTENT")

    # Validity rate excludes missing/no_content (not the report's fault)
    verifiable = valid + invalid
    validity_rate = valid / verifiable if verifiable > 0 else 0.0
    error_rate = invalid / total if total > 0 else 0.0

    metrics = CitationValidityMetrics(
        total_citations=sum(len(occs) for occs in citation_occurrences.values()),
        unique_citations=len(citation_occurrences),
        valid_count=valid,
        invalid_count=invalid,
        missing_source_count=missing,
        no_content_count=no_content,
        validity_rate=validity_rate,
        error_rate=error_rate
    )

    return results, metrics


def split_into_sentences(report: str) -> List[str]:
    """Split report into sentences, excluding headers and metadata.

    Returns only content sentences, not headers, list items, or source listings.
    """
    # Find report body (before Sources section)
    sources_idx = report.find("## Sources")
    if sources_idx == -1:
        sources_idx = report.find("### Sources")
    report_body = report[:sources_idx] if sources_idx != -1 else report

    sentences = []

    # Split into paragraphs first
    paragraphs = report_body.split('\n\n')

    for para in paragraphs:
        para = para.strip()

        # Skip headers
        if para.startswith('#'):
            continue

        # Skip empty paragraphs
        if not para:
            continue

        # Skip horizontal rules
        if para.startswith('---'):
            continue

        # For list items, extract the text after the bullet
        if para.startswith('* ') or para.startswith('- '):
            # Could be a Verified Findings bullet or regular list
            text = para[2:].strip()
            if text:
                sentences.append(text)
            continue

        # Regular paragraph - split into sentences
        # Handle abbreviations to avoid false splits
        para_processed = para.replace('e.g.', 'eg').replace('i.e.', 'ie')
        para_processed = para_processed.replace('U.S.', 'US').replace('etc.', 'etc')

        sent_list = re.split(r'(?<=[.!?])\s+', para_processed)
        for sent in sent_list:
            sent = sent.strip()
            if len(sent) > 10:  # Skip very short fragments
                sentences.append(sent)

    return sentences


def is_factual_sentence(sentence: str) -> bool:
    """Determine if a sentence contains factual content that should be cited.

    Returns True if sentence contains:
    - Numbers, percentages, dollar amounts
    - Specific dates or years
    - Acronyms or proper nouns
    - Quoted text
    - Comparative claims (more, less, best, worst)

    Returns False for:
    - Meta-commentary ("This section discusses...")
    - Vague statements ("Some experts believe...")
    - Simple definitions or explanations
    """
    # Numbers, percentages, money
    if re.search(r'\d+\.?\d*%', sentence):
        return True
    if re.search(r'\$[\d,]+', sentence):
        return True
    if re.search(r'\b\d{4}\b', sentence):  # Year
        return True
    if re.search(r'\b\d+\s*(billion|million|trillion|thousand)\b', sentence, re.I):
        return True

    # Acronyms (2+ capital letters together)
    if re.search(r'\b[A-Z]{2,}\b', sentence):
        return True

    # Quoted text suggests specific claims
    if '"' in sentence or '"' in sentence:
        return True

    # Comparative claims
    if re.search(r'\b(more|less|best|worst|highest|lowest|largest|smallest)\b', sentence, re.I):
        return True

    # Specific verbs that suggest factual claims
    if re.search(r'\b(reduces?|increases?|improves?|prevents?|causes?|enables?)\b', sentence, re.I):
        return True

    return False


def calculate_citation_coverage(report: str) -> CoverageResult:
    """Calculate what percentage of factual sentences have citations.

    This is a QUALITY metric, separate from validity.
    - High coverage = most statements are cited
    - Low coverage = many uncited statements (may need more sources)

    Args:
        report: Full report text

    Returns:
        CoverageResult with coverage statistics
    """
    sentences = split_into_sentences(report)

    if not sentences:
        return CoverageResult()

    cited_count = 0
    uncited_factual = []

    for sent in sentences:
        if re.search(r'\[\d+\]', sent):
            cited_count += 1
        elif is_factual_sentence(sent):
            # This is a factual-looking sentence without citations
            uncited_factual.append(sent[:100])  # Truncate for display

    return CoverageResult(
        total_sentences=len(sentences),
        cited_sentences=cited_count,
        uncited_factual=uncited_factual,
        coverage_rate=cited_count / len(sentences) if sentences else 0.0
    )


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


def calculate_claim_metrics(per_claim_results: List[ClaimResult]) -> ClaimMetrics:
    """Calculate claim metrics from verification results.

    Hallucination rate = (FALSE + UNCITED) / total
    - FALSE: proven incorrect claims
    - UNCITED: claims without citations (trust risk even if true)

    Uses OR logic to prevent double-counting (an uncited FALSE claim counts once).
    """
    if not per_claim_results:
        return ClaimMetrics()

    true_count = sum(1 for r in per_claim_results if r.status == "TRUE")
    false_count = sum(1 for r in per_claim_results if r.status == "FALSE")
    unverifiable_count = sum(1 for r in per_claim_results if r.status == "UNVERIFIABLE")
    uncited_count = sum(1 for r in per_claim_results if r.is_uncited)
    total = len(per_claim_results)

    # Risky = FALSE OR UNCITED (OR prevents double-counting)
    risky_claims = sum(1 for r in per_claim_results if r.status == "FALSE" or r.is_uncited)

    return ClaimMetrics(
        total=total,
        true_count=true_count,
        false_count=false_count,
        unverifiable_count=unverifiable_count,
        uncited_count=uncited_count,
        hallucination_rate=risky_claims / total,
        grounding_rate=true_count / total,
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


def extract_key_terms(claim: str) -> List[str]:
    """Extract key terms from a claim for targeted passage search.

    Finds named entities, acronyms, quoted terms, and proper nouns that
    should be searched as exact strings rather than individual words.
    """
    key_terms = []

    # 1. Quoted strings (e.g., "Project Moonshot", "defense-in-depth")
    quoted = re.findall(r'"([^"]+)"', claim)
    key_terms.extend(quoted)

    # 2. Acronyms and codes (e.g., AEF-1, M-25-22, LITHOS)
    acronyms = re.findall(r'\b[A-Z][A-Z0-9-]+(?:-\d+)?\b', claim)
    key_terms.extend(acronyms)

    # 3. Proper nouns (capitalized phrases like "Digital Trust Centre")
    # Match 2-4 consecutive capitalized words
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', claim)
    key_terms.extend(proper_nouns)

    # Deduplicate and filter short terms
    return list(set(t for t in key_terms if len(t) >= 3))


async def verify_claim_against_source(
    claim: str,
    source: SourceRecord,
    llm,
    claim_id: str
) -> Tuple[str, float, str]:
    """Verify a claim against a specific source.

    Uses a two-phase passage extraction:
    1. First, search for key terms (acronyms, quoted strings, proper nouns)
    2. Fall back to keyword-based paragraph matching

    Returns: (status, confidence, evidence_snippet)
    """
    content = source.get("content", "")
    if not content:
        return "UNVERIFIABLE", 0.0, "Source has no content"

    content_lower = content.lower()
    paragraphs = content.split('\n\n')

    # Phase 1: Search for key terms (exact substring match)
    key_terms = extract_key_terms(claim)
    term_passages = []

    for term in key_terms:
        term_lower = term.lower()
        if term_lower in content_lower:
            # Find the paragraph containing this term
            for para in paragraphs:
                if term_lower in para.lower() and len(para) >= 50:
                    term_passages.append(para)
                    break

    # Phase 2: Keyword-based paragraph scoring (fallback)
    claim_words = set(w.lower() for w in claim.split() if len(w) > 3)
    best_para = ""
    best_score = 0

    for para in paragraphs:
        if len(para) < 50:
            continue
        score = sum(1 for w in claim_words if w in para.lower())
        if score > best_score:
            best_score = score
            best_para = para

    # Combine: prioritize term-matched passages, then keyword-matched
    all_passages = term_passages + ([best_para] if best_para else [])

    if all_passages:
        # Deduplicate and take up to 2 passages
        seen = set()
        unique_passages = []
        for p in all_passages:
            p_key = p[:100]  # Use first 100 chars as key
            if p_key not in seen:
                seen.add(p_key)
                unique_passages.append(p)
        passage = "\n\n".join(unique_passages[:2])[:2000]
    else:
        passage = content[:1500]

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

    # Check for duplicate sources and log stats
    url_counts = {}
    for source in sources:
        url = source.get("url", "").rstrip('/').lower()
        if url:
            if url not in url_counts:
                url_counts[url] = []
            url_counts[url].append(len(source.get("content", "")))

    duplicates = {url: lengths for url, lengths in url_counts.items() if len(lengths) > 1}
    if duplicates:
        print(f"[EVAL] Found {len(duplicates)} URLs with duplicate sources (will use longest content):")
        for url, lengths in list(duplicates.items())[:3]:  # Show first 3
            print(f"[EVAL]   - {url[:60]}... content lengths: {sorted(lengths)}")

    # Parse Sources section to build citation -> URL mapping
    citation_map = parse_sources_section(report)
    print(f"[EVAL] Parsed {len(citation_map)} citations from Sources section")

    # ==========================================================================
    # CITATION-FIRST EVALUATION (Deterministic)
    # ==========================================================================
    # These metrics are computed WITHOUT LLM calls - fast and deterministic
    print(f"\n[EVAL] === CITATION-FIRST METRICS (Deterministic) ===")

    # 1. Citation Validity: Are our citations supported by their sources?
    validity_results, validity_metrics = evaluate_citation_validity(
        report, sources, citation_map
    )
    print(f"[EVAL] Citation Validity: {validity_metrics.valid_count}/{validity_metrics.unique_citations} "
          f"unique citations verified ({validity_metrics.validity_rate:.0%})")

    if validity_metrics.invalid_count > 0:
        print(f"[EVAL] ⚠️ {validity_metrics.invalid_count} citations could not be verified against sources")
        for r in validity_results:
            if r.status == "INVALID":
                print(f"[EVAL]   - [{r.citation_num}] {r.context[:60]}...")

    if validity_metrics.missing_source_count > 0:
        print(f"[EVAL] ⚠️ {validity_metrics.missing_source_count} citations reference missing sources")

    if validity_metrics.no_content_count > 0:
        print(f"[EVAL] ⚠️ {validity_metrics.no_content_count} sources have no content to verify against")

    # 2. Citation Coverage: What % of sentences have citations?
    coverage = calculate_citation_coverage(report)
    print(f"[EVAL] Citation Coverage: {coverage.cited_sentences}/{coverage.total_sentences} "
          f"sentences cited ({coverage.coverage_rate:.0%})")

    if coverage.uncited_factual:
        print(f"[EVAL] ⚠️ {len(coverage.uncited_factual)} factual statements without citations:")
        for stmt in coverage.uncited_factual[:3]:  # Show first 3
            print(f"[EVAL]   - \"{stmt[:70]}...\"")

    print(f"[EVAL] ==============================================\n")
    # ==========================================================================

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

    # Step 3: Aggregate metrics (uses extracted function for testability)
    claim_metrics = calculate_claim_metrics(per_claim_results)

    # Extract values for use in warnings/output below
    true_count = claim_metrics.true_count
    false_count = claim_metrics.false_count
    total = claim_metrics.total
    uncited_count = claim_metrics.uncited_count

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
        warnings.append(f"Hallucination rate ({claim_metrics.hallucination_rate:.1%}) exceeds 2% target (includes {false_count} FALSE + {uncited_count} UNCITED)")
    if claim_metrics.grounding_rate < 0.85:
        warnings.append(f"Grounding rate ({claim_metrics.grounding_rate:.1%}) below 85% target")

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
    print(f"\n[EVAL] =============== EVALUATION COMPLETE ===============")
    print(f"[EVAL]")
    print(f"[EVAL] --- CITATION-FIRST METRICS (Deterministic) ---")
    print(f"[EVAL] Citation Validity: {validity_metrics.validity_rate:.0%} ({validity_metrics.valid_count}/{validity_metrics.unique_citations} citations verified)")
    print(f"[EVAL] Citation Coverage: {coverage.coverage_rate:.0%} ({coverage.cited_sentences}/{coverage.total_sentences} sentences cited)")
    if coverage.uncited_factual:
        print(f"[EVAL] Uncited Factual Statements: {len(coverage.uncited_factual)}")
    print(f"[EVAL]")
    print(f"[EVAL] --- LLM-BASED METRICS (Legacy) ---")
    print(f"[EVAL] Claims: {true_count}/{total} TRUE ({claim_metrics.grounding_rate:.0%} grounding)")
    print(f"[EVAL] Hallucination rate: {claim_metrics.hallucination_rate:.1%} (FALSE + UNCITED, target: <2%)")
    print(f"[EVAL] Uncited claims: {uncited_count} (high-risk)")
    print(f"[EVAL] Citation accuracy: {citation_metrics.accuracy:.0%}")
    print(f"[EVAL]")
    print(f"[EVAL] Cost: ~${cost_est.estimated_cost_usd:.2f}")
    if warnings:
        print(f"[EVAL] Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"[EVAL]   - {w}")
    print(f"[EVAL] ======================================================\n")

    return result
