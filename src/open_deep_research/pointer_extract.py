"""Pointer-based extraction: LLM points, code extracts.

This module implements a new approach to prevent hallucination:
1. LLM reads sources and outputs "pointers" (what to extract)
2. Code uses fuzzy matching to find actual text in sources
3. If text found → verified. If not → flagged.

The LLM never writes factual content, only points to it.
"""

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


@dataclass
class Pointer:
    """A pointer from LLM indicating what to extract."""
    source_id: str  # Which source to extract from
    keywords: List[str]  # Key terms to find
    context: str  # What this extraction is about (for organization)
    micro_quote: Optional[str] = None  # 8-15 word verbatim phrase for strict matching


@dataclass
class Extraction:
    """Result of attempting to extract based on a pointer."""
    pointer: Pointer
    status: str  # "verified", "partial", "not_found"
    extracted_text: Optional[str] = None
    match_score: float = 0.0
    source_url: Optional[str] = None
    # NEW: Span offsets for reverification (character positions in source)
    span_start: int = -1
    span_end: int = -1
    # NEW: Keywords that were actually matched
    keywords_matched: List[str] = field(default_factory=list)
    # NEW: How this extraction was verified
    verification_method: str = "keyword_window"  # "micro_quote", "strict_substring"
    # NEW: Structured failure diagnostics
    failure_reason: Optional[str] = None  # "keywords_missing", "score_too_low", "quality_reject", "source_missing", "content_empty"
    failure_details: Optional[dict] = None  # {"missing_keywords": [...], "score": 0.2}


@dataclass
class ExtractionReport:
    """Full report with extractions organized by topic."""
    topic: str
    extractions: List[Extraction] = field(default_factory=list)

    @property
    def verified_count(self) -> int:
        return sum(1 for e in self.extractions if e.status == "verified")

    @property
    def total_count(self) -> int:
        return len(self.extractions)


def clean_extracted_text(text: str, max_length: int = 200) -> str:
    """Clean extracted text of HTML/XML tags, markdown, artifacts, and normalize.

    Args:
        text: Raw text to clean
        max_length: Maximum length (will truncate to sentence boundary)
    """
    # Strip HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Strip markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Strip markdown reference links [text](#anchor) → text
    text = re.sub(r'\[([^\]]+)\]\(#[^)]*\)', r'\1', text)

    # Strip orphaned link fragments like "text](#anchor)" or "](#anchor)"
    text = re.sub(r'\w*\]\(#[^)]*\)', '', text)
    text = re.sub(r'\]\([^)]*\)', '', text)

    # Strip markdown images ![alt](url) or broken !Image patterns
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    text = re.sub(r'!Image\s*\d+[^\s]*', '', text)

    # Strip bold **text** or __text__ → text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)

    # Strip orphaned ** anywhere (bold markers without matching pair)
    text = re.sub(r'^\*\*\s*', '', text)  # Start
    text = re.sub(r'\s*\*\*$', '', text)  # End
    text = re.sub(r'\*\*\s+', ' ', text)  # Mid-text with space after
    text = re.sub(r'\s+\*\*', ' ', text)  # Mid-text with space before
    text = re.sub(r'(\w)\*\*\s', r'\1 ', text)  # Word** followed by space

    # Strip italic *text* or _text_ → text (but not bullet points)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)

    # Strip markdown headers ## Header → Header
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Remove separator lines (----, ===, etc.)
    text = re.sub(r'-{3,}', ' ', text)
    text = re.sub(r'={3,}', ' ', text)

    # Strip markdown table syntax (pipe characters and surrounding whitespace)
    # Table row: | cell | cell | → cell cell
    text = re.sub(r'\s*\|\s*', ' ', text)

    # Remove bullet-style prefixes (including markdown *)
    text = re.sub(r'^\s*[-•*]\s*', '', text)
    text = re.sub(r'\s+[-•*]\s+', ' ', text)  # Mid-text bullets

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Truncate to max_length at sentence boundary if too long
    if len(text) > max_length:
        # Find last COMPLETE sentence ending before max_length
        # Avoid cutting mid-URL by looking for sentence endings not preceded by common URL patterns
        truncated = text[:max_length]

        # Find sentence boundaries (. ! ?) not inside URLs
        # Look for period followed by space and capital, or end of string
        best_cut = -1
        for match in re.finditer(r'[.!?](?=\s+[A-Z]|\s*$)', truncated):
            best_cut = match.end()

        if best_cut > max_length // 2:
            text = text[:best_cut].strip()
        else:
            # Fall back: find last period/question/exclaim
            last_period = truncated.rfind('. ')
            last_question = truncated.rfind('? ')
            last_exclaim = truncated.rfind('! ')
            cut_point = max(last_period, last_question, last_exclaim)

            if cut_point > max_length // 3:
                text = text[:cut_point + 1].strip()
            else:
                # Last resort: word boundary
                text = truncated.rsplit(' ', 1)[0]

    return text


def is_quality_extraction(text: str, max_words: int = 50) -> bool:
    """Filter out garbage extractions (tables, metadata, fragments, navigation).

    Args:
        text: Extracted text to evaluate
        max_words: Maximum word count (default 50, facts should be concise)

    Returns:
        True if text is quality content, False if garbage
    """
    if not text or len(text) < 50:  # Minimum 50 chars for substance
        return False

    # Reject if too long (facts should be concise claims, not paragraphs)
    word_count = len(text.split())
    if word_count > max_words:
        return False

    # Reject table fragments (2+ pipe characters = markdown table syntax)
    if text.count('|') >= 2:
        return False

    # Reject metadata blocks
    if 'Metadata' in text and ('License' in text or 'Provider' in text):
        return False

    # --- Navigation pattern rejection ---
    text_lower = text.lower()

    # Reject navigation link patterns
    nav_patterns = [
        '[skip to',           # [Skip to main content]
        '[read more]',        # Action links
        '[contact us]',
        '[learn more]',
        '[sign up]',
        '[log in]',
        '[home]',
        '[about]',
        'log in[',            # Log in[Sign up] combo
        'sign up[',
        '✕dismiss',           # Dismissible banners
        'dismiss this',
        '[products]',
        '[services]',
        '[pricing]',
    ]
    for pattern in nav_patterns:
        if pattern in text_lower:
            return False

    # Reject if multiple bracket links (likely nav menu)
    bracket_links = re.findall(r'\[[^\]]{1,20}\]', text)
    if len(bracket_links) >= 3:
        # Likely navigation: [Home] [About] [Products]
        return False

    # Reject if mostly punctuation/formatting (low alpha ratio)
    # NOTE: 0.35 allows numeric-heavy content (prices, metrics, percentages)
    # Previous 0.5 was rejecting valuable data like "$0.30/1K | $15/1M"
    alpha_count = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_count / max(len(text), 1)
    if alpha_ratio < 0.35:
        return False

    # Reject truncated content ending with incomplete markers
    stripped = text.rstrip()
    if stripped.endswith('*') or stripped.endswith('...') or stripped.endswith(':'):
        # But allow ... if it's after a complete sentence
        if not (stripped.endswith('...') and len(stripped) > 50 and stripped[-4] in '.!?'):
            return False

    # Reject if starts with markdown artifacts
    if text.lstrip().startswith(('##', '**', '| ', '- |', '####', '*What')):
        return False

    # Too many markdown artifacts = formatting, not content
    if text.count('**') > 4 or text.count('](') > 2:
        return False

    # Reject product intro patterns: "CompanyName: description" or "CompanyName is a/the/an"
    # These are overviews, not specific claims
    intro_match = re.match(r'^([A-Z][a-zA-Z0-9]+)\s*[:,]\s*', text)
    if intro_match:
        # Starts with "CompanyName:" or "CompanyName,"
        return False

    intro_match = re.match(r'^([A-Z][a-zA-Z0-9]+)\s+(?:is\s+(?:a|an|the|built|known|designed))', text)
    if intro_match:
        # "CompanyName is a platform..." or "CompanyName is built for..."
        return False

    # Reject header patterns (at start or anywhere in text)
    header_patterns = ['best for:', 'key features:', 'key takeaways:', 'key strengths:', 'pros:', 'cons:', 'pricing:', 'overview:']
    text_lower = text.lower()
    for header in header_patterns:
        if header in text_lower:
            return False

    # Reject questions (text containing '?' early suggests it's a question, not a claim)
    # Only reject if ? is in first 50 chars (question at start) or text ends with ?
    if text.strip().endswith('?'):
        return False
    if '?' in text[:50]:
        return False

    return True


def find_tightest_keyword_window(
    content: str,
    keywords: List[str],
    max_window_chars: int = 500
) -> Tuple[Optional[str], int, int, List[str], float]:
    """Find minimal span covering most keywords using sliding window.

    This is more robust than sentence splitting because it doesn't break on
    abbreviations like "Dr.", "vs.", "Inc.", etc.

    Args:
        content: Normalized source content (lowercase for matching)
        keywords: List of lowercase keywords to find
        max_window_chars: Maximum window size to consider

    Returns:
        (text, span_start, span_end, keywords_matched, coverage_ratio) or (None, -1, -1, [], 0.0)
    """
    if not content or not keywords:
        return None, -1, -1, [], 0.0

    content_lower = content.lower()

    # Find all positions of each keyword
    keyword_positions = {}  # keyword -> list of (start, end) positions
    for kw in keywords:
        positions = []
        start = 0
        while True:
            idx = content_lower.find(kw, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(kw)))
            start = idx + 1
        if positions:
            keyword_positions[kw] = positions

    if not keyword_positions:
        return None, -1, -1, [], 0.0

    # If only one keyword found, return its first occurrence
    if len(keyword_positions) == 1:
        kw = list(keyword_positions.keys())[0]
        pos = keyword_positions[kw][0]
        return content[pos[0]:pos[1]], pos[0], pos[1], [kw], 1.0 / len(keywords)

    # Build list of all keyword occurrences with their keyword index
    # Format: (position, keyword)
    all_positions = []
    for kw, positions in keyword_positions.items():
        for start, end in positions:
            all_positions.append((start, kw, end))

    # Sort by position
    all_positions.sort(key=lambda x: x[0])

    # Sliding window to find tightest span covering most keywords
    best_window = None
    best_coverage = 0
    best_length = float('inf')

    n = len(all_positions)
    for i in range(n):
        # Start a window from position i
        window_start = all_positions[i][0]
        covered_kws = set()

        for j in range(i, n):
            pos_start, kw, pos_end = all_positions[j]
            window_end = pos_end

            # Check if window is too large
            if window_end - window_start > max_window_chars:
                break

            covered_kws.add(kw)
            coverage = len(covered_kws)
            window_length = window_end - window_start

            # Better coverage, or same coverage with shorter window
            if coverage > best_coverage or (coverage == best_coverage and window_length < best_length):
                best_coverage = coverage
                best_length = window_length
                best_window = (window_start, window_end, list(covered_kws))

    if not best_window:
        return None, -1, -1, list(keyword_positions.keys()), 0.0

    start, end, matched = best_window
    coverage_ratio = len(matched) / len(keywords)

    return content[start:end], start, end, matched, coverage_ratio


def expand_to_sentence_bounds(
    content: str,
    span_start: int,
    span_end: int,
    max_expand: int = 100
) -> Tuple[int, int]:
    """Expand span to sentence boundaries for better readability.

    Args:
        content: Source content
        span_start: Start of keyword window
        span_end: End of keyword window
        max_expand: Maximum chars to expand in each direction

    Returns:
        (expanded_start, expanded_end)
    """
    # Sentence ending patterns
    sentence_ends = '.!?'

    # Expand backwards to find sentence start
    search_start = max(0, span_start - max_expand)
    expanded_start = span_start
    for i in range(span_start - 1, search_start - 1, -1):
        if content[i] in sentence_ends:
            # Found sentence end - next char is sentence start
            expanded_start = i + 1
            # Skip whitespace
            while expanded_start < span_start and content[expanded_start].isspace():
                expanded_start += 1
            break
    else:
        # No sentence end found - go to search_start
        expanded_start = search_start

    # Expand forwards to find sentence end
    search_end = min(len(content), span_end + max_expand)
    expanded_end = span_end
    for i in range(span_end, search_end):
        if content[i] in sentence_ends:
            expanded_end = i + 1
            break
    else:
        # No sentence end found - go to search_end
        expanded_end = search_end

    return expanded_start, expanded_end


def find_best_match(
    keywords: List[str],
    source_content: str,
    min_score: float = 0.6,
    micro_quote: Optional[str] = None
) -> Tuple[Optional[str], float, int, int, List[str], str]:
    """Find the best matching sentence/passage containing keywords.

    Uses micro-quote strict matching first, then tightest keyword window algorithm.

    Args:
        keywords: List of key terms to find
        source_content: Full text of source
        min_score: Minimum match score (0-1)
        micro_quote: Optional verbatim phrase for strict substring matching

    Returns:
        (extracted_text, match_score, span_start, span_end, keywords_matched, method)
        where method is "micro_quote", "keyword_window", or "sentence_fallback"
        or (None, 0.0, -1, -1, [], "") on failure
    """
    if not keywords or not source_content:
        return None, 0.0, -1, -1, [], ""

    # Light cleaning only - strip HTML but keep full length for searching
    source_content = re.sub(r'<[^>]+>', '', source_content)  # Strip HTML tags
    source_content = re.sub(r'\s+', ' ', source_content).strip()  # Normalize whitespace

    # PRIMARY: Try micro-quote strict matching first (highest confidence)
    if micro_quote and len(micro_quote) >= 10:  # Minimum 10 chars for meaningful quote
        # Normalize micro_quote whitespace
        micro_quote_normalized = re.sub(r'\s+', ' ', micro_quote).strip()

        # Try exact match first
        if micro_quote_normalized in source_content:
            start = source_content.find(micro_quote_normalized)
            end = start + len(micro_quote_normalized)

            # Expand to sentence boundaries for context
            exp_start, exp_end = expand_to_sentence_bounds(source_content, start, end)
            expanded_text = source_content[exp_start:exp_end].strip()

            # Quality check
            if expanded_text.count('|') < 2:  # Not a table
                cleaned = clean_extracted_text(expanded_text, max_length=500)
                if is_quality_extraction(cleaned):
                    # Determine keywords in the matched text
                    keywords_lower = [k.lower().strip() for k in keywords if k.strip()]
                    matched_kws = [kw for kw in keywords_lower if kw in cleaned.lower()]
                    if cleaned in source_content:
                        clean_start = source_content.find(cleaned)
                        clean_end = clean_start + len(cleaned)
                        return cleaned, 1.0, clean_start, clean_end, matched_kws, "micro_quote"
                    return cleaned, 1.0, exp_start, exp_end, matched_kws, "micro_quote"

        # Try case-insensitive match
        source_lower = source_content.lower()
        micro_lower = micro_quote_normalized.lower()
        if micro_lower in source_lower:
            start = source_lower.find(micro_lower)
            end = start + len(micro_lower)

            exp_start, exp_end = expand_to_sentence_bounds(source_content, start, end)
            expanded_text = source_content[exp_start:exp_end].strip()

            if expanded_text.count('|') < 2:
                cleaned = clean_extracted_text(expanded_text, max_length=500)
                if is_quality_extraction(cleaned):
                    keywords_lower = [k.lower().strip() for k in keywords if k.strip()]
                    matched_kws = [kw for kw in keywords_lower if kw in cleaned.lower()]
                    if cleaned in source_content:
                        clean_start = source_content.find(cleaned)
                        clean_end = clean_start + len(cleaned)
                        return cleaned, 1.0, clean_start, clean_end, matched_kws, "micro_quote"
                    return cleaned, 1.0, exp_start, exp_end, matched_kws, "micro_quote"

    # Normalize keywords
    keywords_lower = [k.lower().strip() for k in keywords if k.strip()]

    if not keywords_lower:
        return None, 0.0, -1, -1, [], ""

    # Check which keywords exist in source
    content_lower = source_content.lower()
    keywords_found = [kw for kw in keywords_lower if kw in content_lower]

    if not keywords_found:
        return None, 0.0, -1, -1, [], ""

    match_ratio = len(keywords_found) / len(keywords_lower)

    if match_ratio < min_score:
        return None, match_ratio, -1, -1, keywords_found, ""

    # SECONDARY: Use tightest keyword window algorithm
    # This is more robust than sentence splitting (handles "Dr.", "vs.", etc.)
    window_text, span_start, span_end, matched_kws, coverage = find_tightest_keyword_window(
        source_content, keywords_lower
    )

    if window_text and coverage >= min_score:
        # Expand window to sentence boundaries for readability
        exp_start, exp_end = expand_to_sentence_bounds(source_content, span_start, span_end)
        expanded_text = source_content[exp_start:exp_end].strip()

        # Early reject: table rows (2+ pipes = markdown table syntax)
        if expanded_text.count('|') < 2:
            cleaned = clean_extracted_text(expanded_text, max_length=500)
            if is_quality_extraction(cleaned):
                # Recalculate span for cleaned text if it's a substring
                if cleaned in source_content:
                    clean_start = source_content.find(cleaned)
                    clean_end = clean_start + len(cleaned)
                    return cleaned, coverage, clean_start, clean_end, matched_kws, "keyword_window"
                return cleaned, coverage, exp_start, exp_end, matched_kws, "keyword_window"

    # FALLBACK: Sentence-based approach for cases where keyword window didn't work
    # Split on: paragraphs, markdown headers, table rows, then sentences
    chunks = re.split(r'\n\n+|\n(?=##?\s)|\n(?=\|)', source_content)
    sentences = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk:
            sents = re.split(r'(?<=[.!?])\s+', chunk)
            sentences.extend([s.strip() for s in sents if s.strip()])

    # Score all sentences
    candidates = []
    for sent in sentences:
        sent_lower = sent.lower()
        sent_keywords = [kw for kw in keywords_found if kw in sent_lower]
        if sent_keywords:
            score = len(sent_keywords) / len(keywords_lower)
            span_start = source_content.find(sent)
            span_end = span_start + len(sent) if span_start >= 0 else -1
            candidates.append((score, sent.strip(), span_start, span_end, sent_keywords))

    # Also try sentence pairs and triplets
    for i in range(len(sentences) - 1):
        passage = sentences[i] + " " + sentences[i + 1]
        passage_lower = passage.lower()
        passage_keywords = [kw for kw in keywords_found if kw in passage_lower]
        if passage_keywords:
            score = len(passage_keywords) / len(keywords_lower)
            span_start = source_content.find(sentences[i])
            span_end = span_start + len(passage) if span_start >= 0 else -1
            candidates.append((score, passage.strip(), span_start, span_end, passage_keywords))

    for i in range(len(sentences) - 2):
        passage = sentences[i] + " " + sentences[i + 1] + " " + sentences[i + 2]
        passage_lower = passage.lower()
        passage_keywords = [kw for kw in keywords_found if kw in passage_lower]
        if passage_keywords:
            score = len(passage_keywords) / len(keywords_lower)
            span_start = source_content.find(sentences[i])
            span_end = span_start + len(passage) if span_start >= 0 else -1
            candidates.append((score, passage.strip(), span_start, span_end, passage_keywords))

    # Sort by score descending, then by length ascending
    candidates.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)

    # Return first candidate that passes quality filter
    for score, text, span_start, span_end, matched_kws in candidates:
        if score >= min_score:
            if text.count('|') >= 2:
                continue
            cleaned = clean_extracted_text(text, max_length=500)
            if is_quality_extraction(cleaned):
                if cleaned in source_content:
                    clean_start = source_content.find(cleaned)
                    clean_end = clean_start + len(cleaned)
                    return cleaned, score, clean_start, clean_end, matched_kws, "sentence_fallback"
                return cleaned, score, span_start, span_end, matched_kws, "sentence_fallback"

    # No fallback - if quality filter rejects all candidates, return None
    best_score = candidates[0][0] if candidates else 0.0
    best_kws = candidates[0][4] if candidates else []
    return None, best_score, -1, -1, best_kws, ""


def extract_from_pointer(
    pointer: Pointer,
    sources: Dict[str, dict],
    min_score: float = 0.6
) -> Extraction:
    """Extract text from source based on pointer.

    Args:
        pointer: The extraction pointer from LLM
        sources: Dict mapping source_id to source data (must have 'content' key)
        min_score: Minimum match score for verification

    Returns:
        Extraction result with status, text, spans, and failure diagnostics
    """
    source = sources.get(pointer.source_id)

    if not source:
        return Extraction(
            pointer=pointer,
            status="not_found",
            match_score=0.0,
            failure_reason="source_missing",
            failure_details={"source_id": pointer.source_id}
        )

    content = source.get("content", "") or source.get("raw_content", "")
    url = source.get("url", "")

    if not content:
        return Extraction(
            pointer=pointer,
            status="not_found",
            source_url=url,
            match_score=0.0,
            failure_reason="content_empty",
            failure_details={"source_id": pointer.source_id, "url": url}
        )

    # Check keyword presence first for diagnostic purposes
    content_lower = content.lower()
    keywords_lower = [k.lower().strip() for k in pointer.keywords if k.strip()]
    missing_keywords = [kw for kw in keywords_lower if kw not in content_lower]

    if len(missing_keywords) > len(keywords_lower) * 0.5:
        # More than half of keywords missing - early reject with diagnostics
        return Extraction(
            pointer=pointer,
            status="not_found",
            source_url=url,
            match_score=0.0,
            failure_reason="keywords_missing",
            failure_details={
                "missing": missing_keywords,
                "total": len(keywords_lower),
                "missing_ratio": len(missing_keywords) / len(keywords_lower) if keywords_lower else 1.0
            }
        )

    extracted_text, score, span_start, span_end, keywords_matched, method = find_best_match(
        pointer.keywords,
        content,
        min_score=min_score,
        micro_quote=pointer.micro_quote  # NEW: Pass micro_quote for strict matching
    )

    # Apply quality filter to extracted text
    if extracted_text and not is_quality_extraction(extracted_text):
        # Garbage extraction - mark as not found with diagnostics
        return Extraction(
            pointer=pointer,
            status="not_found",
            extracted_text=None,
            match_score=score,
            source_url=url,
            span_start=span_start,
            span_end=span_end,
            keywords_matched=keywords_matched,
            verification_method=method or "keyword_window",
            failure_reason="quality_reject",
            failure_details={"rejected_text_preview": extracted_text[:100] if extracted_text else None}
        )

    if extracted_text and score >= min_score:
        status = "verified"
        failure_reason = None
        failure_details = None
    elif score > 0:
        status = "partial"
        failure_reason = "score_too_low"
        failure_details = {"score": score, "min_required": min_score}
    else:
        status = "not_found"
        failure_reason = "no_match"
        failure_details = {"keywords": pointer.keywords}

    return Extraction(
        pointer=pointer,
        status=status,
        extracted_text=extracted_text,
        match_score=score,
        source_url=url,
        span_start=span_start,
        span_end=span_end,
        keywords_matched=keywords_matched,
        verification_method=method or "keyword_window",  # NEW: Use actual method from find_best_match
        failure_reason=failure_reason,
        failure_details=failure_details
    )


def verify_span(extraction: Extraction, source_content: str) -> bool:
    """Deterministic verification: check extracted_text is at the span position.

    Args:
        extraction: Extraction with span_start and span_end populated
        source_content: Original source content (normalized)

    Returns:
        True if extracted_text matches the span in source, False otherwise
    """
    if extraction.span_start < 0 or extraction.span_end <= extraction.span_start:
        return False

    if not extraction.extracted_text:
        return False

    # Normalize source content same way as find_best_match
    source_content = re.sub(r'<[^>]+>', '', source_content)
    source_content = re.sub(r'\s+', ' ', source_content).strip()

    # Check if extracted text can be found at the span position
    if extraction.span_end > len(source_content):
        return False

    span_text = source_content[extraction.span_start:extraction.span_end]

    # The extracted text should match (or be within) the span
    return extraction.extracted_text in span_text or span_text in extraction.extracted_text


# Prompt for LLM to clean extractions - outputs clean text, code verifies substring
CLEANUP_PROMPT = '''For each text, output ONLY the meaningful content with navigation/UI garbage removed.

Rules:
- Remove navigation links: [Skip to...], [Read more], [Contact us], Log in, Sign up
- Remove UI artifacts: Search K, menu items, keyboard shortcuts
- Remove image markdown: ![](...)
- Remove header artifacts: # Title, [Site Name](/), page titles with |
- Remove formatting artifacts: * **Date** ###, changelog prefixes
- Remove unrelated content: FAQ questions in brackets, promotional text
- Keep the actual informative content about the topic
- If there's no meaningful content, output "NO_CONTENT"
- CRITICAL: Output must be an EXACT substring of the original (copy-paste, don't rephrase!)

Texts:
{facts}

Output JSON array:
[
  {{"index": 0, "cleaned": "the exact meaningful content here"}},
  {{"index": 1, "cleaned": "NO_CONTENT"}},
  ...
]

Output ONLY the JSON array.'''


def parse_cleanup_response(response: str) -> List[dict]:
    """Parse LLM cleanup response."""
    try:
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return []


# Tokens that should NEVER be removed during cleanup (semantic loss)
NEGATION_TOKENS = {"not", "never", "no", "without", "except", "unless", "don't", "doesn't", "can't", "won't", "isn't", "aren't"}
QUALIFIER_TOKENS = {"only", "just", "approximately", "about", "up to", "at least", "nearly", "almost", "less than", "more than"}


def verify_and_apply_cleanup(original: str, cleaned: str) -> Optional[str]:
    """Verify cleaned text is exact substring of original without semantic loss.

    Guards against:
    - Removing negation tokens (not, never, without, etc.)
    - Removing numbers/metrics
    - Removing qualifiers (only, approximately, etc.)

    Returns cleaned text if valid, None if should reject.
    """
    if not cleaned or cleaned == "NO_CONTENT":
        return None  # Reject - no meaningful content

    if cleaned in original:
        # Valid - it's an exact substring, but check for semantic loss
        if len(cleaned) >= 50:  # Minimum length for meaningful content
            # Check for negation loss
            orig_tokens = set(original.lower().split())
            clean_tokens = set(cleaned.lower().split())

            # Reject if negation removed
            orig_negations = orig_tokens & NEGATION_TOKENS
            clean_negations = clean_tokens & NEGATION_TOKENS
            if orig_negations - clean_negations:
                # Lost a negation - this changes meaning, keep original
                return original

            # Reject if qualifier removed
            orig_qualifiers = orig_tokens & QUALIFIER_TOKENS
            clean_qualifiers = clean_tokens & QUALIFIER_TOKENS
            if orig_qualifiers - clean_qualifiers:
                # Lost a qualifier - this changes precision, keep original
                return original

            # Reject if numbers removed
            orig_numbers = set(re.findall(r'\d+(?:\.\d+)?(?:%|ms|k|M|GB|MB|TB)?', original))
            clean_numbers = set(re.findall(r'\d+(?:\.\d+)?(?:%|ms|k|M|GB|MB|TB)?', cleaned))
            if orig_numbers - clean_numbers:
                # Lost a number - keep original
                return original

            return cleaned
        else:
            return None  # Too short after cleaning
    else:
        # LLM modified the text - reject, keep original
        return original


def format_facts_for_cleanup(extractions: List['Extraction']) -> str:
    """Format extractions for the cleanup prompt."""
    lines = []
    for i, ext in enumerate(extractions):
        if ext.status == "verified" and ext.extracted_text:
            lines.append(f"[{i}] {ext.extracted_text[:500]}")
    return "\n\n".join(lines)


# Prompt for LLM to generate pointers
POINTER_PROMPT = '''Topic: {topic}

Extract FACTUAL CLAIMS - single sentences with specific, verifiable information.

CRITICAL: Each fact = ONE sentence. Keywords must all appear in the SAME sentence.

A fact is a SINGLE DECLARATIVE SENTENCE (under 40 words) that:
- States a specific claim ("X does Y", "X achieves Z metric")
- Contains evidence: numbers, metrics, percentages, comparisons
- Can be verified as true or false

GOOD (one sentence, specific claim):
✓ "Hamming runs 10,000 concurrent test calls with sub-200ms latency"
✓ "Coval applies autonomous vehicle testing methodology with 95% coverage"
✓ "The platform tracks 40+ metrics including latency and error rates"

BAD - DO NOT EXTRACT:
✗ Product intros: "CompanyName is a platform that..." or "CompanyName: Description..."
✗ Headers/titles: "Key Features:", "What to Expect", "Overview", section headings
✗ Questions: "How does X work?" "What should you consider?"
✗ Multi-sentence passages: If keywords span 2+ sentences, TOO MUCH
✗ Promotional fluff: "Learn more", "Try our platform", "Best-in-class"
✗ Vague claims: "Very fast", "Great performance", "Easy to use" (no metrics)
✗ Buzzword salad (sounds technical, says nothing):
  - "Advanced natural language understanding transformations..."
  - "Leveraging innovative AI to deliver seamless enterprise solutions"
  - "Revolutionizing comprehension capabilities through integration"
  - If text has "revolutionizing/transforming/leveraging/synergy" + no numbers → SKIP
✗ Tautologies (obvious statements):
  - "Voice AI systems benefit from AI improvements"
  - "Modern platforms use modern technology"

If text has "ProductName: claim", point to keywords in the claim AFTER the colon.

For each fact, provide:
- source_id: exactly as shown (src_000, src_001, etc)
- keywords: 3-5 distinctive words from ONE sentence only
- micro_quote: 8-15 word phrase that MUST appear VERBATIM in the source (copy-paste exactly!)
- context: brief 3-6 word label

The micro_quote is CRITICAL - it anchors the extraction to exact text. Copy it character-for-character.

Sources:
{sources}

Output JSON array:
[
  {{"source_id": "src_000", "keywords": ["researchers", "discovered", "2024"], "micro_quote": "researchers discovered a breakthrough in 2024", "context": "Research discovery findings"}},
  {{"source_id": "src_001", "keywords": ["study", "participants", "improved"], "micro_quote": "study participants improved by 35%", "context": "Study participant outcomes"}}
]'''


def parse_pointer_response(response: str, min_relevance: int = 3) -> List[Pointer]:
    """Parse LLM response into Pointer objects.

    Args:
        response: LLM response containing JSON array
        min_relevance: Minimum relevance score to include (1-5, default 3)

    Returns:
        List of Pointer objects with relevance >= min_relevance
    """
    # Try to extract JSON array
    try:
        # Find JSON array in response
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            data = json.loads(match.group())
            pointers = []
            for item in data:
                if isinstance(item, dict):
                    # Filter by relevance score
                    relevance = item.get("relevance", 5)  # Default high for backwards compat
                    if relevance < min_relevance:
                        continue  # Skip low-relevance pointers

                    pointers.append(Pointer(
                        source_id=str(item.get("source_id", "")),
                        keywords=item.get("keywords", []),
                        context=item.get("context", ""),
                        micro_quote=item.get("micro_quote")  # NEW: Parse micro_quote
                    ))
            return pointers
    except json.JSONDecodeError:
        pass

    return []


def format_sources_for_prompt(sources: Dict[str, dict], max_chars: int = 2000) -> str:
    """Format sources for the pointer prompt."""
    lines = []
    for src_id, src in sources.items():
        content = src.get("content", "") or src.get("raw_content", "")
        title = src.get("title", "") or src.get("source_title", "Unknown")

        # Truncate content
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        lines.append(f"[{src_id}] {title}\n{content}\n")

    return "\n---\n".join(lines)


def format_extraction_markdown(
    extractions: List[Extraction],
    use_color: bool = True
) -> str:
    """Format extractions as markdown with optional color styling.

    Args:
        extractions: List of extraction results
        use_color: If True, use HTML color spans for verified/unverified

    Returns:
        Markdown string
    """
    lines = []

    for ext in extractions:
        if ext.status == "verified" and ext.extracted_text:
            if use_color:
                # Green-tinted for verified
                text = f'<span style="color: #166534; background: #dcfce7; padding: 2px 4px; border-radius: 3px;">{ext.extracted_text}</span>'
            else:
                text = ext.extracted_text

            # Add source citation
            if ext.source_url:
                lines.append(f"> {text}\n> — [{ext.pointer.context}]({ext.source_url})\n")
            else:
                lines.append(f"> {text}\n")

        elif ext.status == "partial":
            if use_color:
                # Yellow-tinted for partial
                text = f'<span style="color: #854d0e; background: #fef9c3; padding: 2px 4px; border-radius: 3px;">⚠️ Partial match ({ext.match_score:.0%}): {ext.extracted_text or "keywords found but no clean extraction"}</span>'
            else:
                text = f"[Partial: {ext.match_score:.0%}] {ext.extracted_text or 'N/A'}"
            lines.append(f"{text}\n")

        else:
            if use_color:
                # Red-tinted for not found
                text = f'<span style="color: #991b1b; background: #fee2e2; padding: 2px 4px; border-radius: 3px;">❌ Not found: {ext.pointer.context}</span>'
            else:
                text = f"[NOT FOUND] {ext.pointer.context}"
            lines.append(f"{text}\n")

    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    # Test data
    test_sources = {
        "src_001": {
            "content": "The RAND Corporation released a comprehensive security report in October 2025 recommending multi-layered security approaches for frontier AI systems.",
            "url": "https://rand.org/report",
            "title": "RAND AI Security Report"
        },
        "src_002": {
            "content": "OpenAI announced new safety measures including defense-in-depth strategies and the formation of the Frontier Risk Council to oversee model deployment.",
            "url": "https://openai.com/safety",
            "title": "OpenAI Safety Update"
        }
    }

    # Test pointers
    test_pointers = [
        Pointer(source_id="src_001", keywords=["RAND", "October 2025", "multi-layered", "security"], context="RAND recommendations"),
        Pointer(source_id="src_002", keywords=["OpenAI", "defense-in-depth", "Frontier Risk Council"], context="OpenAI safety"),
        Pointer(source_id="src_001", keywords=["hallucination", "fake", "wrong"], context="Should fail - not in source"),
    ]

    print("Testing pointer extraction...\n")

    for pointer in test_pointers:
        result = extract_from_pointer(pointer, test_sources)
        print(f"Pointer: {pointer.context}")
        print(f"  Status: {result.status}")
        print(f"  Score: {result.match_score:.1%}")
        print(f"  Text: {result.extracted_text[:80] if result.extracted_text else 'N/A'}...")
        print()

    # Format as markdown
    results = [extract_from_pointer(p, test_sources) for p in test_pointers]
    print("\n--- Markdown Output ---\n")
    print(format_extraction_markdown(results, use_color=False))
