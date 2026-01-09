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


# Prompt for LLM to generate pointers
POINTER_PROMPT = '''Extract key facts from these sources about: {topic}

For each source, identify 1-2 important facts. Output pointers with:
- source_id: Match exactly (e.g., "src_001")
- keywords: 3-5 SINGLE words that appear in that source (not phrases)
- context: What this fact is about (3-5 words)

CRITICAL: Use single distinctive words, not phrases. Example:
- Good: ["Biden", "October", "2023", "Executive", "Order"]
- Bad: ["Executive Order", "October 2023"] (these are phrases)

Sources:
{sources}

Output JSON array:
[
  {{"source_id": "src_001", "keywords": ["RAND", "security", "2025", "framework"], "context": "RAND security report"}},
  {{"source_id": "src_002", "keywords": ["OpenAI", "Council", "deployment", "safety"], "context": "OpenAI governance"}}
]

Output ONLY the JSON array.'''


def parse_pointer_response(response: str) -> List[Pointer]:
    """Parse LLM response into Pointer objects."""
    # Try to extract JSON array
    try:
        # Find JSON array in response
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            data = json.loads(match.group())
            pointers = []
            for item in data:
                if isinstance(item, dict):
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
