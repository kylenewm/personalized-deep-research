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


@dataclass
class Extraction:
    """Result of attempting to extract based on a pointer."""
    pointer: Pointer
    status: str  # "verified", "partial", "not_found"
    extracted_text: Optional[str] = None
    match_score: float = 0.0
    source_url: Optional[str] = None


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
    """Clean extracted text of HTML/XML tags, artifacts, and normalize.

    Args:
        text: Raw text to clean
        max_length: Maximum length (will truncate to sentence boundary)
    """
    # Strip HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove separator lines (----, ===, etc.)
    text = re.sub(r'-{3,}', ' ', text)
    text = re.sub(r'={3,}', ' ', text)

    # Remove bullet-style prefixes
    text = re.sub(r'^\s*[-•*]\s*', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Truncate to max_length at sentence boundary if too long
    if len(text) > max_length:
        # Find last sentence ending before max_length
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')

        cut_point = max(last_period, last_question, last_exclaim)
        if cut_point > max_length // 2:  # Only use if reasonably far in
            text = text[:cut_point + 1]
        else:
            # Fall back to word boundary
            text = truncated.rsplit(' ', 1)[0] + '...'

    return text


def is_quality_extraction(text: str) -> bool:
    """Filter out garbage extractions (tables, metadata, fragments, navigation).

    Args:
        text: Extracted text to evaluate

    Returns:
        True if text is quality content, False if garbage
    """
    if not text or len(text) < 50:  # Minimum 50 chars for substance
        return False

    # Reject table fragments (multiple pipe characters)
    if text.count('|') > 3:
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
    alpha_count = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_count / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False

    # Reject truncated content ending with incomplete markers
    stripped = text.rstrip()
    if stripped.endswith('*') or stripped.endswith('...') or stripped.endswith(':'):
        # But allow ... if it's after a complete sentence
        if not (stripped.endswith('...') and len(stripped) > 50 and stripped[-4] in '.!?'):
            return False

    # Reject if starts with markdown artifacts
    if text.lstrip().startswith(('##', '**', '| ', '- |')):
        return False

    return True


def find_best_match(
    keywords: List[str],
    source_content: str,
    min_score: float = 0.6
) -> Tuple[Optional[str], float]:
    """Find the best matching sentence/passage containing keywords.

    Args:
        keywords: List of key terms to find
        source_content: Full text of source
        min_score: Minimum match score (0-1)

    Returns:
        (extracted_text, match_score) or (None, 0.0)
    """
    if not keywords or not source_content:
        return None, 0.0

    # Light cleaning only - strip HTML but keep full length for searching
    source_content = re.sub(r'<[^>]+>', '', source_content)  # Strip HTML tags
    source_content = re.sub(r'\s+', ' ', source_content).strip()  # Normalize whitespace

    # Normalize
    content_lower = source_content.lower()
    keywords_lower = [k.lower().strip() for k in keywords if k.strip()]

    if not keywords_lower:
        return None, 0.0

    # Check if all keywords exist in source
    keywords_found = []
    for kw in keywords_lower:
        if kw in content_lower:
            keywords_found.append(kw)

    if not keywords_found:
        return None, 0.0

    match_ratio = len(keywords_found) / len(keywords_lower)

    if match_ratio < min_score:
        return None, match_ratio

    # Find the sentence(s) containing the most keywords
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', source_content)

    best_sentence = None
    best_score = 0.0

    for sent in sentences:
        sent_lower = sent.lower()
        sent_keywords = sum(1 for kw in keywords_found if kw in sent_lower)
        if sent_keywords > 0:
            score = sent_keywords / len(keywords_lower)
            if score > best_score:
                best_score = score
                best_sentence = sent.strip()

    # If single sentence doesn't have enough, try to find a passage
    if best_score < min_score and len(sentences) > 1:
        # Try pairs of consecutive sentences
        for i in range(len(sentences) - 1):
            passage = sentences[i] + " " + sentences[i + 1]
            passage_lower = passage.lower()
            passage_keywords = sum(1 for kw in keywords_found if kw in passage_lower)
            score = passage_keywords / len(keywords_lower)
            if score > best_score:
                best_score = score
                best_sentence = passage.strip()

        # Try triplets if still not enough
        if best_score < min_score and len(sentences) > 2:
            for i in range(len(sentences) - 2):
                passage = sentences[i] + " " + sentences[i + 1] + " " + sentences[i + 2]
                passage_lower = passage.lower()
                passage_keywords = sum(1 for kw in keywords_found if kw in passage_lower)
                score = passage_keywords / len(keywords_lower)
                if score > best_score:
                    best_score = score
                    best_sentence = passage.strip()

    if best_score >= min_score:
        return clean_extracted_text(best_sentence, max_length=300), best_score

    return None, best_score


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
        Extraction result with status and text
    """
    source = sources.get(pointer.source_id)

    if not source:
        return Extraction(
            pointer=pointer,
            status="not_found",
            match_score=0.0
        )

    content = source.get("content", "") or source.get("raw_content", "")
    url = source.get("url", "")

    if not content:
        return Extraction(
            pointer=pointer,
            status="not_found",
            source_url=url,
            match_score=0.0
        )

    extracted_text, score = find_best_match(
        pointer.keywords,
        content,
        min_score=min_score
    )

    # Apply quality filter to extracted text
    if extracted_text and not is_quality_extraction(extracted_text):
        # Garbage extraction - mark as not found
        return Extraction(
            pointer=pointer,
            status="not_found",
            extracted_text=None,
            match_score=0.0,
            source_url=url
        )

    if extracted_text and score >= min_score:
        status = "verified"
    elif score > 0:
        status = "partial"
    else:
        status = "not_found"

    return Extraction(
        pointer=pointer,
        status=status,
        extracted_text=extracted_text,
        match_score=score,
        source_url=url
    )


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


def verify_and_apply_cleanup(original: str, cleaned: str) -> Optional[str]:
    """Verify cleaned text is exact substring of original.

    Returns cleaned text if valid, None if should reject.
    """
    if not cleaned or cleaned == "NO_CONTENT":
        return None  # Reject - no meaningful content

    if cleaned in original:
        # Valid - it's an exact substring
        if len(cleaned) >= 50:  # Minimum length for meaningful content
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
POINTER_PROMPT = '''Extract facts from these sources that DIRECTLY answer: {topic}

RELEVANCE CHECK (critical):
- Only extract facts that help answer the specific question
- Skip sources that don't contain relevant information
- Skip generic/promotional content ("We'll show you how to...", "In this article...")
- Skip tutorial intros, marketing claims, and filler text

For each RELEVANT fact, output:
- source_id: Match exactly (e.g., "src_001")
- keywords: 3-5 SINGLE words that appear in that source (not phrases)
- context: What this fact is about (3-5 words)
- relevance: 1-5 score (5=directly answers question, 3=somewhat relevant, 1=tangential)

ONLY include facts with relevance >= 3.

CRITICAL: Use single distinctive words, not phrases. Example:
- Good: ["Biden", "October", "2023", "Executive", "Order"]
- Bad: ["Executive Order", "October 2023"] (these are phrases)

Sources:
{sources}

Output JSON array:
[
  {{"source_id": "src_001", "keywords": ["latency", "200ms", "L40S"], "context": "Speech model latency", "relevance": 5}},
  {{"source_id": "src_002", "keywords": ["ElevenLabs", "cloning", "accuracy"], "context": "Voice cloning quality", "relevance": 4}}
]

Output ONLY the JSON array. Skip sources with no relevant facts.'''


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
                        context=item.get("context", "")
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
