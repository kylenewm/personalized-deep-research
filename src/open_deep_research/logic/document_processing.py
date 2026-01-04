"""Document processing with spacy-based sentence extraction.

Replaces brittle regex-based extraction with NLP-aware chunking.
Used for extracting quote-worthy passages from Tavily Extract API content.
"""

import re
from typing import List, Tuple, Optional

import spacy
from spacy.tokens import Span


# Lazy-load spacy model
_nlp: Optional[spacy.Language] = None


def get_nlp() -> spacy.Language:
    """Get spacy NLP model (lazy-loaded on first call)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def strip_navigation(content: str) -> str:
    """Remove navigation artifacts commonly found in Extract API output.

    Handles:
    - Markdown-style links: [text](url)
    - Navigation sections: "Skip to content", "Jump to", etc.
    - Table of contents headers
    - Image references: ![alt](url)
    - Header-only lines (# Title without content)

    Args:
        content: Raw content from Tavily Extract API

    Returns:
        Cleaned content with navigation removed
    """
    # Remove markdown images ![alt](url)
    content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', content)

    # Remove markdown links but keep text [text](url) -> text
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)

    # Remove anchor links like (#section)
    content = re.sub(r'\(#[^)]+\)', '', content)

    # Common navigation line patterns (case-insensitive)
    nav_line_patterns = [
        r'^#{1,6}\s*(Navigation|Navigation Menu|Contents|Table of Contents|Menu|Search)\s*$',
        r'^\s*(Skip to content|Jump to content|Jump to navigation)\s*$',
        r'^\s*(Previous topic|Next topic|This page)\s*$',
        r'^\s*(Search|Sign in|Register|Log in|Sign up)\s*$',
        r'^\s*\*{3,}\s*$',  # Horizontal rules (*** or more)
        r'^-{3,}$',  # Dashes as separator
        r'^\s*Add topic\s*$',
    ]

    lines = content.split('\n')
    filtered_lines = []
    in_nav_section = False
    nav_section_depth = 0

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at start
        if not filtered_lines and not stripped:
            continue

        # Check if this is a navigation header (starts a nav section)
        is_nav_header = any(re.match(p, stripped, re.IGNORECASE) for p in nav_line_patterns)

        # Check if this is a markdown header
        header_match = re.match(r'^(#{1,6})\s+', stripped)

        if is_nav_header:
            in_nav_section = True
            if header_match:
                nav_section_depth = len(header_match.group(1))
            continue

        # If we're in a nav section, check if we've exited
        if in_nav_section:
            # Exit nav section if we hit content header of same/higher level or blank line followed by content
            if header_match:
                header_depth = len(header_match.group(1))
                if header_depth <= nav_section_depth:
                    in_nav_section = False
                    # Include this header as it starts real content
                    filtered_lines.append(line)
                continue
            elif not stripped:
                # Blank line - potential end of nav section
                continue
            elif len(stripped) > 50:
                # Long line suggests real content, exit nav section
                in_nav_section = False
                filtered_lines.append(line)
                continue
            else:
                # Short line in nav section, skip
                continue

        filtered_lines.append(line)

    content = '\n'.join(filtered_lines)

    # Remove orphaned header markers at start (# alone on line)
    content = re.sub(r'^#{1,6}\s*\n+', '', content)

    # Normalize whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()

    return content


def score_sentence(sent: Span, word_count: int) -> float:
    """Score sentence informativeness (0.0 - 1.0).

    Scoring criteria:
    - Named entities (ORG, PERSON, GPE, MONEY, PERCENT, DATE, CARDINAL)
    - Contains numbers/statistics
    - Contains quotation marks (embedded quotes)
    - Moderate length (sweet spot: 20-60 words)

    Args:
        sent: spacy Span representing a sentence
        word_count: Pre-computed word count

    Returns:
        Informativeness score between 0.0 and 1.0
    """
    score = 0.0

    # Named entities boost (high-value entity types)
    entity_types = {ent.label_ for ent in sent.ents}
    high_value_ents = {"ORG", "PERSON", "GPE", "MONEY", "PERCENT", "DATE", "CARDINAL", "QUANTITY"}
    ent_overlap = len(entity_types & high_value_ents)
    score += min(ent_overlap * 0.15, 0.45)  # Max 0.45 from entities

    # Numbers/statistics boost
    if any(token.like_num for token in sent):
        score += 0.2

    # Quotation marks (contains embedded quote)
    quote_chars = {'"', '"', '"', "'", ''', '''}
    if any(c in sent.text for c in quote_chars):
        score += 0.15

    # Moderate length bonus (optimal range: 20-60 words)
    if 20 <= word_count <= 60:
        score += 0.15
    elif 15 <= word_count < 20 or 60 < word_count <= 80:
        score += 0.05

    # Penalize very short sentences (likely fragments)
    if word_count < 10:
        score -= 0.3

    return max(0.0, min(1.0, score))


def chunk_by_sentences(
    content: str,
    min_words: int = 10,
    max_words: int = 100,
    min_score: float = 0.3,
    max_sentences: int = 50
) -> List[str]:
    """Extract quote-worthy sentences using spacy NLP.

    Replaces the brittle 15-60 word + capitalization heuristic with
    proper sentence boundary detection and informativeness scoring.

    Args:
        content: Clean text content (preferably from Extract API)
        min_words: Minimum words per sentence (default: 10)
        max_words: Maximum words per sentence (default: 100)
        min_score: Minimum informativeness score (default: 0.3)
        max_sentences: Maximum sentences to return (default: 50)

    Returns:
        List of sentences sorted by informativeness score (descending)
    """
    # Clean navigation artifacts if present
    content = strip_navigation(content)

    if not content or len(content) < 50:
        return []

    # Process with spacy
    nlp = get_nlp()

    # Handle very long content by processing in chunks
    max_chars = 100000
    if len(content) > max_chars:
        content = content[:max_chars]

    doc = nlp(content)

    scored_sentences: List[Tuple[str, float]] = []

    for sent in doc.sents:
        text = sent.text.strip()

        # Skip empty or whitespace-only
        if not text:
            continue

        word_count = len(text.split())

        # Skip if outside word bounds
        if word_count < min_words or word_count > max_words:
            continue

        # Calculate informativeness score
        score = score_sentence(sent, word_count)

        if score >= min_score:
            scored_sentences.append((text, score))

    # Sort by score descending
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Return top sentences up to max
    return [s[0] for s in scored_sentences[:max_sentences]]


def extract_passages_from_content(
    content: str,
    extraction_method: str = "extract_api",
    min_words: int = 10,
    max_words: int = 100,
    min_score: float = 0.3
) -> List[str]:
    """Extract quote-worthy passages from content based on extraction method.

    This is the main entry point that routes to appropriate extraction logic.

    Args:
        content: Text content from source
        extraction_method: How content was obtained ("extract_api", "search_raw", "fallback")
        min_words: Minimum words per passage
        max_words: Maximum words per passage
        min_score: Minimum informativeness score

    Returns:
        List of quote-worthy passages
    """
    if extraction_method == "extract_api":
        # Extract API gives clean content - use spacy chunking
        return chunk_by_sentences(
            content,
            min_words=min_words,
            max_words=max_words,
            min_score=min_score
        )
    else:
        # For raw HTML content, fall back to legacy sanitize + extract
        # Import here to avoid circular dependency
        from open_deep_research.logic.sanitize import sanitize_for_quotes, extract_paragraphs

        clean_text = sanitize_for_quotes(content)
        return extract_paragraphs(clean_text, min_words=15, max_words=60)
