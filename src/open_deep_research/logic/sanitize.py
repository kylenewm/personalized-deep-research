"""HTML sanitization logic for evidence extraction.

This module implements deterministic HTML sanitization that preserves
paragraph structure for quote extraction per TRUST_ARCH.md.
"""

import re
from typing import List


def sanitize_for_quotes(html: str) -> str:
    """Sanitize HTML content for quote extraction.

    Per TRUST_ARCH.md Section B (Evidence Extraction):
    1. Regex strip <script>, <style>, and all tags
    2. Replace block-level tags (</div>, </p>, </li>, </h1-6>) with \\n\\n
       to preserve paragraph structure

    Args:
        html: Raw HTML content from source

    Returns:
        Clean text with paragraph breaks preserved
    """
    if not html:
        return ""

    text = html

    # Step 1: Remove script and style blocks entirely (including content)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Step 2: Remove comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Step 3: Replace block-level closing tags with double newlines to preserve structure
    block_tags = r'</(?:div|p|li|ul|ol|h[1-6]|article|section|header|footer|blockquote|tr|td|th)>'
    text = re.sub(block_tags, '\n\n', text, flags=re.IGNORECASE)

    # Step 4: Replace <br> tags with single newlines
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

    # Step 5: Strip all remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Step 6: Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&apos;', "'")

    # Step 6b: Strip markdown formatting (content may contain markdown from APIs)
    # Headers: # Heading, ## Heading, etc.
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # *italic*
    text = re.sub(r'__(.+?)__', r'\1', text)       # __bold__
    text = re.sub(r'_(.+?)_', r'\1', text)         # _italic_

    # List markers at start of line
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Links: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Inline code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Step 7: Normalize whitespace within lines (but preserve paragraph breaks)
    # First, normalize multiple spaces to single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Step 8: Normalize multiple newlines to exactly two (paragraph break)
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Step 9: Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Step 10: Final cleanup - remove leading/trailing whitespace
    text = text.strip()

    return text


def extract_paragraphs(clean_text: str, min_words: int = 15, max_words: int = 60) -> List[str]:
    """Extract paragraphs that meet word count criteria.

    Per TRUST_ARCH.md Section B:
    - Split by \\n\\n
    - Filter paragraphs: Length 15 to 60 words
    - Content: Must contain at least one noun phrase or number (simple heuristic)

    Args:
        clean_text: Sanitized text from sanitize_for_quotes()
        min_words: Minimum word count (default 15)
        max_words: Maximum word count (default 60)

    Returns:
        List of paragraph strings meeting the criteria
    """
    if not clean_text:
        return []

    # Split by double newlines (paragraph breaks)
    paragraphs = clean_text.split('\n\n')

    valid_paragraphs = []
    for para in paragraphs:
        # Clean up the paragraph
        para = para.strip()
        if not para:
            continue

        # Count words
        words = para.split()
        word_count = len(words)

        # Filter by word count
        if word_count < min_words or word_count > max_words:
            continue

        # Simple heuristic: must contain a number OR capitalized word (noun phrase proxy)
        has_number = bool(re.search(r'\d', para))
        has_capitalized = bool(re.search(r'\b[A-Z][a-z]+\b', para))

        if has_number or has_capitalized:
            valid_paragraphs.append(para)

    return valid_paragraphs
