"""Pipeline v2: Three-stage safeguarded generation.

Stage 1: Batched pointer extraction from all sources
Stage 2: Arranger groups by theme + curates (drops ~30-50%)
Stage 3: Per-theme synthesis with fine curation

Output: Hybrid report with verified facts (green) + AI analysis (gray)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Handle both relative and absolute imports for testing
try:
    from .pointer_extract import (
        Pointer,
        Extraction,
        POINTER_PROMPT,
        CLEANUP_PROMPT,
        extract_from_pointer,
        format_sources_for_prompt,
        parse_pointer_response,
        format_facts_for_cleanup,
        parse_cleanup_response,
        verify_and_apply_cleanup,
    )
except ImportError:
    from pointer_extract import (
        Pointer,
        Extraction,
        POINTER_PROMPT,
        CLEANUP_PROMPT,
        extract_from_pointer,
        format_sources_for_prompt,
        parse_pointer_response,
        format_facts_for_cleanup,
        parse_cleanup_response,
        verify_and_apply_cleanup,
    )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ThemeGroup:
    """A group of facts under a common theme."""
    theme: str
    fact_ids: List[int]
    intro: str = ""  # AI-written intro for this section


@dataclass
class CuratedFacts:
    """Result of arranger curation."""
    groups: List[ThemeGroup]
    excluded_ids: List[int]  # Facts dropped as irrelevant/redundant


@dataclass
class ThemedSection:
    """A synthesized section for one theme."""
    theme: str
    intro: str  # AI intro (gray)
    facts: List[Extraction]  # Verified facts (green)
    transitions: List[str]  # Between facts (gray)


@dataclass
class HybridReport:
    """Final report with verified facts + AI analysis."""
    title: str
    executive_summary: str  # AI synthesis (gray)
    sections: List[ThemedSection]  # Themed verified facts
    analysis: str  # AI analysis section (gray)
    conclusion: str  # AI conclusion (gray)

    # Stats
    total_extracted: int = 0
    total_verified: int = 0
    total_used: int = 0

    @property
    def verified_count(self) -> int:
        return sum(len(s.facts) for s in self.sections)


# =============================================================================
# Stage 1: Batched Pointer Extraction
# =============================================================================

BATCH_SIZE = 1  # Process one source at a time for thoroughness
MAX_CHARS_PER_SOURCE = 50000  # Send full source content (models have 128k context)
CHUNK_SIZE = 100000  # Effectively disabled - only chunk if > 100k chars
CHUNK_THRESHOLD = 100000  # Effectively disabled - almost no sources are this large
DEFAULT_MIN_SCORE = 0.3  # Lower threshold - need 30% of keywords to match


def batch_sources(sources: Dict[str, dict], batch_size: int = BATCH_SIZE) -> List[Dict[str, dict]]:
    """Split sources into batches."""
    items = list(sources.items())
    batches = []
    for i in range(0, len(items), batch_size):
        batch = dict(items[i:i + batch_size])
        batches.append(batch)
    return batches


def chunk_content(content: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split content into chunks, trying to break at paragraph boundaries."""
    if len(content) <= chunk_size:
        return [content]

    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        if end >= len(content):
            chunks.append(content[start:])
            break

        # Try to break at paragraph boundary
        para_break = content.rfind('\n\n', start, end)
        if para_break > start + chunk_size // 2:
            end = para_break
        else:
            # Try sentence boundary
            sent_break = content.rfind('. ', start, end)
            if sent_break > start + chunk_size // 2:
                end = sent_break + 1

        chunks.append(content[start:end])
        start = end

    return chunks


async def extract_from_source_chunked(
    source_id: str,
    source: dict,
    topic: str,
    llm_call,
    min_score: float = DEFAULT_MIN_SCORE
) -> List[Extraction]:
    """Extract from a single source, chunking if needed for thoroughness."""
    import asyncio

    content = source.get("content", "") or source.get("raw_content", "")
    if not content:
        return []

    chunks = chunk_content(content, CHUNK_SIZE)

    # Question-aware extraction prompt
    chunk_prompt_template = '''Research question: {topic}

A fact is a specific, verifiable claim. Examples:
- "Model X has 150ms latency" ✓
- "Pricing starts at $0.01 per 1000 chars" ✓
- "The API supports 50 languages" ✓
- "This is where things get interesting" ✗ (intro fluff)
- "Best for enterprise use" ✗ (opinion without evidence)

Extract facts that help answer the research question. For each, output 3-5 unique keywords:

Text:
{chunk}

Output JSON array (empty [] if no facts):
[{{"keywords": ["Model", "X", "150ms", "latency"]}}]'''

    async def extract_chunk(chunk_text):
        prompt = chunk_prompt_template.format(topic=topic, chunk=chunk_text)
        response = await llm_call(prompt)
        return parse_pointer_response(response)

    # Extract from all chunks in parallel
    chunk_results = await asyncio.gather(*[extract_chunk(c) for c in chunks])

    # Combine all pointers
    all_pointers = []
    for pointers in chunk_results:
        for p in pointers:
            # Add source_id since chunk prompt doesn't include it
            p.source_id = source_id
            all_pointers.append(p)

    # Verify against full source content
    source_dict = {source_id: source}
    extractions = []
    for pointer in all_pointers:
        result = extract_from_pointer(pointer, source_dict, min_score=min_score)
        extractions.append(result)

    return extractions


async def extract_batch(
    batch: Dict[str, dict],
    topic: str,
    llm_call,  # async function(prompt) -> response
    min_score: float = DEFAULT_MIN_SCORE
) -> List[Extraction]:
    """Extract verified facts from a batch of sources.

    Args:
        batch: Dict of source_id -> source data
        topic: Research topic
        llm_call: Async function to call LLM
        min_score: Minimum match score for verification

    Returns:
        List of Extraction results
    """
    import asyncio

    # For single-source batches with large content, use chunked extraction
    # Only chunk if content exceeds threshold (saves cost for smaller sources)
    if len(batch) == 1:
        source_id, source = list(batch.items())[0]
        content = source.get("content", "") or source.get("raw_content", "")
        if len(content) > CHUNK_THRESHOLD:
            return await extract_from_source_chunked(
                source_id, source, topic, llm_call, min_score
            )

    # Standard batch extraction for small content or multi-source batches
    formatted = format_sources_for_prompt(batch, max_chars=MAX_CHARS_PER_SOURCE)
    prompt = POINTER_PROMPT.format(sources=formatted, topic=topic)

    # Get pointers from LLM
    response = await llm_call(prompt)
    pointers = parse_pointer_response(response)

    # Extract with code verification
    extractions = []
    for pointer in pointers:
        result = extract_from_pointer(pointer, batch, min_score=min_score)
        extractions.append(result)

    return extractions


async def extract_all_batched(
    sources: Dict[str, dict],
    topic: str,
    llm_call,
    batch_size: int = BATCH_SIZE,
    min_score: float = DEFAULT_MIN_SCORE,
    on_batch_complete=None  # callback(batch_num, total_batches, extractions)
) -> List[Extraction]:
    """Extract verified facts from all sources in batches.

    Args:
        sources: All sources
        topic: Research topic
        llm_call: Async LLM function
        batch_size: Sources per batch
        min_score: Verification threshold
        on_batch_complete: Optional progress callback

    Returns:
        All extractions (verified + partial + not_found)
    """
    batches = batch_sources(sources, batch_size)
    all_extractions = []

    for i, batch in enumerate(batches):
        extractions = await extract_batch(batch, topic, llm_call, min_score)
        all_extractions.extend(extractions)

        if on_batch_complete:
            on_batch_complete(i + 1, len(batches), extractions)

    return all_extractions


# =============================================================================
# Deduplication (between Stage 1 and Stage 2)
# =============================================================================

def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    if not text1 or not text2:
        return 0.0

    # Tokenize to words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def normalize_for_comparison(text: str) -> str:
    """Normalize text for deduplication comparison."""
    import re
    # Strip markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def deduplicate_extractions(
    extractions: List[Extraction],
    similarity_threshold: float = 0.5
) -> List[Extraction]:
    """Remove near-duplicate extractions based on content similarity.

    Keeps multiple facts from the same source - only removes actual duplicate content.

    Args:
        extractions: List of verified extractions
        similarity_threshold: Jaccard similarity above which to consider duplicate

    Returns:
        Deduplicated list (keeps higher-scoring extraction when duplicates found)
    """
    if not extractions:
        return []

    # Sort by match_score descending (keep best version when duplicates found)
    sorted_ext = sorted(extractions, key=lambda x: x.match_score, reverse=True)

    # Semantic deduplication - remove actually duplicate content
    # (same or very similar text, regardless of source)
    kept = []
    for ext in sorted_ext:
        if not ext.extracted_text:
            continue

        # Normalize for comparison
        ext_normalized = normalize_for_comparison(ext.extracted_text)

        # Check if similar to any already kept
        is_duplicate = False
        for kept_ext in kept:
            kept_normalized = normalize_for_comparison(kept_ext.extracted_text)
            similarity = compute_text_similarity(ext_normalized, kept_normalized)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(ext)

    return kept


# =============================================================================
# Cleanup Stage: LLM Points, Code Removes
# =============================================================================

async def cleanup_extractions(
    extractions: List[Extraction],
    llm_call,
    batch_size: int = 20
) -> List[Extraction]:
    """Clean navigation artifacts from extractions.

    LLM outputs cleaned text, code verifies it's an exact substring of original.
    If verification fails, keeps original. If no content, rejects extraction.

    Args:
        extractions: Verified extractions to clean
        llm_call: Async LLM function
        batch_size: Extractions per cleanup batch

    Returns:
        Cleaned extractions (rejects pure garbage, cleans others)
    """
    if not extractions:
        return extractions

    cleaned_extractions = []

    # Process in batches
    for i in range(0, len(extractions), batch_size):
        batch = extractions[i:i + batch_size]

        # Format for cleanup prompt
        facts_text = format_facts_for_cleanup(batch)
        if not facts_text:
            cleaned_extractions.extend(batch)
            continue

        prompt = CLEANUP_PROMPT.format(facts=facts_text)
        response = await llm_call(prompt)
        cleanup_results = parse_cleanup_response(response)

        # Build lookup for results
        results_map = {r.get("index"): r.get("cleaned") for r in cleanup_results}

        # Apply and verify cleanup
        for j, ext in enumerate(batch):
            if not ext.extracted_text:
                continue

            cleaned = results_map.get(j)
            if cleaned is None:
                # No cleanup result for this one - keep original
                cleaned_extractions.append(ext)
                continue

            # Verify and apply
            result = verify_and_apply_cleanup(ext.extracted_text, cleaned)

            if result is None:
                # Rejected (pure garbage or too short)
                continue
            else:
                # Update extraction with cleaned text
                ext.extracted_text = result
                cleaned_extractions.append(ext)

    return cleaned_extractions


# =============================================================================
# Stage 2: Arranger (Grouping + Curation)
# =============================================================================

ARRANGER_PROMPT = '''You are organizing research findings to answer a specific question.

QUESTION: {topic}

You have {num_facts} verified facts. Your tasks:

1. AGGRESSIVE QUALITY FILTER (drop liberally):
   - DROP tutorial/promo content: "We'll show you...", "In this guide...", "Learn how to..."
   - DROP vague claims: "is a versatile tool", "offers many features"
   - DROP facts that don't contain specific information (names, numbers, comparisons)
   - DROP anything that reads like marketing copy
   - KEEP only facts with concrete details that help answer the question

2. RELEVANCE FILTER:
   - Does this fact DIRECTLY help answer "{topic}"?
   - If you have to stretch to make it relevant, drop it

3. GROUP remaining facts by theme (3-5 themes)
   - Themes should map to parts of the answer
   - For "best X" questions: "Top Models", "Performance Metrics", "Selection Criteria"

4. FINAL CHECK per fact:
   - Would you cite this in a professional research report?
   - If embarrassing to include, drop it

VERIFIED FACTS:
{facts}

Output JSON:
{{
  "groups": [{{"theme": "Theme Name", "fact_ids": [1, 2, 5]}}],
  "excluded": [{{"id": 3, "reason": "tutorial intro - no specific info"}}]
}}

CRITICAL:
- Be ruthless. It's better to have 5 strong facts than 15 weak ones.
- Exclude anything vague, promotional, or tangential.
- Each fact_id in exactly ONE group OR excluded.

Output ONLY valid JSON.'''


def format_facts_for_arranger(extractions: List[Extraction]) -> str:
    """Format verified facts for arranger prompt."""
    lines = []
    for i, ext in enumerate(extractions, 1):
        if ext.status == "verified" and ext.extracted_text:
            # Truncate long facts for arranger context
            text = ext.extracted_text[:300] + "..." if len(ext.extracted_text) > 300 else ext.extracted_text
            lines.append(f"[{i}] {text}")
            lines.append(f"    Source: {ext.pointer.context}")
            lines.append("")
    return "\n".join(lines)


def parse_arranger_response(response: str, verified_facts: List[Extraction]) -> CuratedFacts:
    """Parse arranger LLM response into structured groups."""
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            data = json.loads(match.group())

            groups = []
            for g in data.get("groups", []):
                groups.append(ThemeGroup(
                    theme=g.get("theme", "Findings"),
                    fact_ids=g.get("fact_ids", [])
                ))

            # Handle excluded - could be list of ints or list of objects
            excluded_raw = data.get("excluded", [])
            excluded_ids = []
            for item in excluded_raw:
                if isinstance(item, int):
                    excluded_ids.append(item)
                elif isinstance(item, dict):
                    excluded_ids.append(item.get("id", item.get("fact_id", 0)))

            return CuratedFacts(groups=groups, excluded_ids=excluded_ids)

    except json.JSONDecodeError:
        pass

    # Fallback: single group with all facts
    all_ids = list(range(1, len(verified_facts) + 1))
    return CuratedFacts(
        groups=[ThemeGroup(theme="Key Findings", fact_ids=all_ids)],
        excluded_ids=[]
    )


async def arrange_facts(
    verified_extractions: List[Extraction],
    topic: str,
    llm_call
) -> CuratedFacts:
    """Group and curate verified facts.

    Args:
        verified_extractions: List of verified extractions only
        topic: Research topic
        llm_call: Async LLM function

    Returns:
        CuratedFacts with theme groups and excluded list
    """
    facts_text = format_facts_for_arranger(verified_extractions)
    prompt = ARRANGER_PROMPT.format(
        topic=topic,
        num_facts=len(verified_extractions),
        facts=facts_text
    )

    response = await llm_call(prompt)
    return parse_arranger_response(response, verified_extractions)


# =============================================================================
# Stage 3: Per-Theme Synthesis
# =============================================================================

THEME_SYNTHESIS_PROMPT = '''You are writing a section of a research report.

Theme: {theme}
Research Topic: {topic}

VERIFIED FACTS for this section (do NOT modify these):
{facts}

Write:
1. INTRO: 2-3 sentences introducing this theme
2. TRANSITIONS: One short transition sentence before each fact (to connect ideas)

You may DROP 1-2 facts if they don't fit the flow (list their IDs in "dropped").

CRITICAL:
- Do NOT rewrite the facts themselves
- Transitions only CONNECT, they don't add new information
- Keep transitions to 1 sentence each
- If a fact contains marketing superlatives ("best", "most", "#1", "leading"),
  your transition should attribute it: "The source describes..." or soften it:
  "among the leading..." Do NOT present vendor marketing as objective fact.

Output JSON:
{{
  "intro": "Your theme introduction...",
  "transitions": ["Transition before fact 1", "Transition before fact 2", ...],
  "dropped": []  // fact IDs that don't fit (optional, max 2)
}}'''


def format_theme_facts(facts: List[Extraction], fact_ids: List[int], all_verified: List[Extraction]) -> str:
    """Format facts for a theme section."""
    lines = []
    for i, fid in enumerate(fact_ids, 1):
        if 1 <= fid <= len(all_verified):
            ext = all_verified[fid - 1]
            if ext.extracted_text:
                lines.append(f"FACT {i} (ID {fid}): {ext.extracted_text}")
                lines.append(f"  Source: {ext.pointer.context}")
                lines.append("")
    return "\n".join(lines)


async def synthesize_theme(
    theme: str,
    fact_ids: List[int],
    all_verified: List[Extraction],
    topic: str,
    llm_call
) -> ThemedSection:
    """Synthesize a single theme section.

    Args:
        theme: Theme name
        fact_ids: IDs of facts in this theme
        all_verified: All verified extractions (for lookup)
        topic: Research topic
        llm_call: Async LLM function

    Returns:
        ThemedSection with intro, facts, transitions
    """
    facts_text = format_theme_facts([], fact_ids, all_verified)
    prompt = THEME_SYNTHESIS_PROMPT.format(
        theme=theme,
        topic=topic,
        facts=facts_text
    )

    response = await llm_call(prompt)

    # Parse response
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            data = json.loads(match.group())
            intro = data.get("intro", "")
            transitions = data.get("transitions", [""] * len(fact_ids))
            dropped = data.get("dropped", [])
        else:
            intro = ""
            transitions = [""] * len(fact_ids)
            dropped = []
    except json.JSONDecodeError:
        intro = ""
        transitions = [""] * len(fact_ids)
        dropped = []

    # Build facts list (excluding dropped)
    kept_ids = [fid for fid in fact_ids if fid not in dropped]
    facts = []
    for fid in kept_ids:
        if 1 <= fid <= len(all_verified):
            facts.append(all_verified[fid - 1])

    # Ensure transitions match facts count
    while len(transitions) < len(facts):
        transitions.append("")

    return ThemedSection(
        theme=theme,
        intro=intro,
        facts=facts,
        transitions=transitions[:len(facts)]
    )


# =============================================================================
# Final Assembly
# =============================================================================

EXECUTIVE_SUMMARY_PROMPT = '''Write an executive summary for this research report.

Topic: {topic}

The report has these themed sections:
{sections_overview}

Write 3-4 sentences summarizing the key findings across all themes.
Do NOT make up facts - only summarize what's in the sections.

Output ONLY the summary text (no JSON).'''


ANALYSIS_PROMPT = '''Write an analysis section for this research report.

Topic: {topic}

Key findings by theme:
{findings_summary}

Write 2-3 paragraphs analyzing:
1. Key patterns or trends across the findings
2. Implications of these findings
3. Notable gaps or areas needing more research

CRITICAL:
- This is YOUR interpretation (will be styled as AI analysis)
- Reference specific themes/findings but don't repeat them
- Be honest about limitations

Output ONLY the analysis text (no JSON).'''


CONCLUSION_PROMPT = '''Write a conclusion for this research report.

Topic: {topic}

Key themes covered: {themes}

Write 2-3 sentences with key takeaways.
Do NOT make up facts - only conclude based on what was found.

Output ONLY the conclusion text (no JSON).'''


async def generate_executive_summary(
    sections: List[ThemedSection],
    topic: str,
    llm_call
) -> str:
    """Generate executive summary."""
    overview = "\n".join([
        f"- {s.theme}: {len(s.facts)} verified findings"
        for s in sections
    ])
    prompt = EXECUTIVE_SUMMARY_PROMPT.format(topic=topic, sections_overview=overview)
    return await llm_call(prompt)


async def generate_analysis(
    sections: List[ThemedSection],
    topic: str,
    llm_call
) -> str:
    """Generate analysis section."""
    summary_parts = []
    for s in sections:
        summary_parts.append(f"\n{s.theme}:")
        for f in s.facts[:3]:  # First 3 facts per theme for context
            summary_parts.append(f"  - {f.extracted_text[:150]}...")

    prompt = ANALYSIS_PROMPT.format(
        topic=topic,
        findings_summary="\n".join(summary_parts)
    )
    return await llm_call(prompt)


async def generate_conclusion(
    sections: List[ThemedSection],
    topic: str,
    llm_call
) -> str:
    """Generate conclusion."""
    themes = ", ".join(s.theme for s in sections)
    prompt = CONCLUSION_PROMPT.format(topic=topic, themes=themes)
    return await llm_call(prompt)


async def assemble_report(
    sections: List[ThemedSection],
    topic: str,
    title: str,
    llm_call,
    total_extracted: int,
    total_verified: int
) -> HybridReport:
    """Assemble final hybrid report.

    Args:
        sections: Synthesized theme sections
        topic: Research topic
        title: Report title
        llm_call: Async LLM function
        total_extracted: Total extractions attempted
        total_verified: Total verified before curation

    Returns:
        Complete HybridReport
    """
    # Generate AI sections (can be parallel)
    exec_summary = await generate_executive_summary(sections, topic, llm_call)
    analysis = await generate_analysis(sections, topic, llm_call)
    conclusion = await generate_conclusion(sections, topic, llm_call)

    total_used = sum(len(s.facts) for s in sections)

    return HybridReport(
        title=title,
        executive_summary=exec_summary,
        sections=sections,
        analysis=analysis,
        conclusion=conclusion,
        total_extracted=total_extracted,
        total_verified=total_verified,
        total_used=total_used
    )


# =============================================================================
# Rendering
# =============================================================================

def render_hybrid_report(report: HybridReport, use_color: bool = True) -> str:
    """Render hybrid report as HTML or plain markdown."""
    if not use_color:
        # Plain markdown mode
        lines = [f"# {report.title}\n"]
        lines.append("## Executive Summary\n")
        lines.append(f"*{report.executive_summary}*\n")
        lines.append("## Verified Findings\n")
        for section in report.sections:
            lines.append(f"### {section.theme}\n")
            if section.intro:
                lines.append(f"*{section.intro}*\n")
            for i, fact in enumerate(section.facts):
                if i < len(section.transitions) and section.transitions[i]:
                    lines.append(f"*{section.transitions[i]}*\n")
                lines.append(f"> {fact.extracted_text}")
                if fact.source_url:
                    lines.append(f"> — [{fact.pointer.context}]({fact.source_url})")
                lines.append("")
        lines.append("## Analysis & Implications\n")
        lines.append(f"*{report.analysis}*\n")
        lines.append("## Conclusion\n")
        lines.append(f"*{report.conclusion}*\n")
        return "\n".join(lines)

    # HTML mode
    lines = [f'<h1>{report.title}</h1>']

    # Executive Summary
    lines.append('<h2>Executive Summary</h2>')
    lines.append(f'<p class="synthesis">{report.executive_summary}</p>')

    # Themed sections
    lines.append('<h2>Verified Findings</h2>')

    for section in report.sections:
        lines.append(f'<h3>{section.theme}</h3>')

        # Theme intro
        if section.intro:
            lines.append(f'<p class="synthesis">{section.intro}</p>')

        # Facts with transitions
        for i, fact in enumerate(section.facts):
            # Transition
            if i < len(section.transitions) and section.transitions[i]:
                lines.append(f'<p class="synthesis">{section.transitions[i]}</p>')

            # Verified fact
            lines.append('<div class="verified-fact">')
            lines.append(f'<p>{fact.extracted_text}</p>')
            if fact.source_url:
                lines.append(f'<a href="{fact.source_url}" class="source-link">{fact.pointer.context}</a>')
            lines.append('</div>')

    # Analysis section
    lines.append('<h2>Analysis & Implications</h2>')
    paragraphs = report.analysis.split('\n\n')
    for p in paragraphs:
        if p.strip():
            lines.append(f'<p class="synthesis">{p.strip()}</p>')

    # Conclusion
    lines.append('<h2>Conclusion</h2>')
    lines.append(f'<p class="synthesis">{report.conclusion}</p>')

    # Stats footer
    lines.append('<hr>')
    lines.append('<div class="stats">')
    lines.append(f'Sources: {report.total_extracted} · Verified: {report.total_verified} · In report: {report.total_used} · Themes: {len(report.sections)}')
    lines.append('</div>')

    return "\n".join(lines)


# HTML template with polished CSS
HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
        :root {{
            --text-primary: #1a1a2e;
            --text-secondary: #4a5568;
            --text-muted: #718096;
            --bg-primary: #ffffff;
            --bg-subtle: #f7fafc;
            --border-light: #e2e8f0;
            --accent: #4a6fa5;
            --accent-light: #eef2f7;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 720px;
            margin: 0 auto;
            padding: 3rem 2rem;
            line-height: 1.75;
            color: var(--text-primary);
            background: var(--bg-primary);
            font-size: 16px;
            -webkit-font-smoothing: antialiased;
        }}

        h1 {{
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 2rem;
            color: var(--text-primary);
        }}

        h2 {{
            font-size: 1.375rem;
            font-weight: 600;
            margin-top: 3rem;
            margin-bottom: 1.25rem;
            color: var(--text-primary);
            letter-spacing: -0.015em;
        }}

        h3 {{
            font-size: 1.125rem;
            font-weight: 600;
            margin-top: 2.5rem;
            margin-bottom: 1rem;
            color: var(--text-secondary);
        }}

        p {{
            margin: 0 0 1.25rem 0;
        }}

        .verified-fact {{
            background: var(--bg-subtle);
            border-left: 3px solid var(--accent);
            padding: 1.25rem 1.5rem;
            margin: 1.5rem 0;
            border-radius: 0 6px 6px 0;
        }}

        .verified-fact p {{
            margin: 0;
            font-size: 0.9375rem;
            color: var(--text-primary);
            line-height: 1.7;
        }}

        .source-link {{
            display: inline-block;
            margin-top: 0.75rem;
            font-size: 0.8125rem;
            color: var(--accent);
            text-decoration: none;
            opacity: 0.85;
        }}

        .source-link:hover {{
            opacity: 1;
            text-decoration: underline;
        }}

        .synthesis {{
            color: var(--text-muted);
            font-size: 0.9375rem;
            margin: 1rem 0;
            line-height: 1.7;
        }}

        .stats {{
            margin-top: 3rem;
            padding: 1.25rem 1.5rem;
            background: var(--bg-subtle);
            border-radius: 8px;
            font-size: 0.875rem;
            color: var(--text-muted);
            border: 1px solid var(--border-light);
        }}

        hr {{
            border: none;
            border-top: 1px solid var(--border-light);
            margin: 3rem 0;
        }}

        a {{
            color: var(--accent);
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        /* Print styles */
        @media print {{
            body {{
                padding: 1rem;
                font-size: 14px;
            }}
            .verified-fact {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{content}
</body>
</html>'''


def render_html(report: HybridReport) -> str:
    """Render report as complete HTML with embedded CSS.

    Uses the new template-based renderer from render.py.
    """
    try:
        from .render import render_report
    except ImportError:
        from render import render_report
    return render_report(report)


# =============================================================================
# Main Pipeline
# =============================================================================

async def run_pipeline_v2(
    sources: Dict[str, dict],
    topic: str,
    title: str,
    llm_call,  # async function(prompt) -> response
    batch_size: int = BATCH_SIZE,
    min_score: float = DEFAULT_MIN_SCORE,
    on_progress=None  # callback(stage, message)
) -> HybridReport:
    """Run the full three-stage pipeline.

    Args:
        sources: Dict of source_id -> {content, url, title}
        topic: Research topic
        title: Report title
        llm_call: Async function to call LLM
        batch_size: Sources per extraction batch
        min_score: Verification threshold
        on_progress: Progress callback

    Returns:
        HybridReport ready for rendering
    """
    def progress(stage: str, msg: str):
        if on_progress:
            on_progress(stage, msg)
        else:
            print(f"[{stage}] {msg}")

    # Stage 1: Batched extraction
    progress("EXTRACT", f"Extracting from {len(sources)} sources in batches of {batch_size}...")

    all_extractions = await extract_all_batched(
        sources, topic, llm_call, batch_size, min_score,
        on_batch_complete=lambda i, t, e: progress("EXTRACT", f"Batch {i}/{t}: {len([x for x in e if x.status == 'verified'])} verified")
    )

    verified = [e for e in all_extractions if e.status == "verified"]
    progress("EXTRACT", f"Total: {len(verified)} verified out of {len(all_extractions)} extractions")

    if not verified:
        raise ValueError("No verified extractions - cannot generate report")

    # Deduplication: Remove near-duplicate extractions
    deduped = deduplicate_extractions(verified)  # Uses default threshold=0.4
    progress("DEDUP", f"Deduplicated: {len(verified)} → {len(deduped)} facts ({len(verified) - len(deduped)} duplicates removed)")
    verified = deduped

    if not verified:
        raise ValueError("No facts remaining after deduplication")

    # Cleanup: LLM points to garbage, code removes it
    progress("CLEANUP", f"Cleaning {len(verified)} facts (LLM points, code removes)...")
    cleaned = await cleanup_extractions(verified, llm_call)
    progress("CLEANUP", f"Cleaned: {len(verified)} → {len(cleaned)} facts ({len(verified) - len(cleaned)} removed as too short)")
    verified = cleaned

    if not verified:
        raise ValueError("No facts remaining after cleanup")

    # Stage 2: Arrange and curate
    progress("ARRANGE", f"Grouping {len(verified)} facts by theme...")
    curated = await arrange_facts(verified, topic, llm_call)

    total_grouped = sum(len(g.fact_ids) for g in curated.groups)
    progress("ARRANGE", f"Created {len(curated.groups)} themes, {total_grouped} facts kept, {len(curated.excluded_ids)} excluded")

    # Stage 3: Per-theme synthesis
    progress("SYNTHESIZE", "Synthesizing themed sections...")
    sections = []
    for group in curated.groups:
        progress("SYNTHESIZE", f"  Processing '{group.theme}' ({len(group.fact_ids)} facts)...")
        section = await synthesize_theme(
            group.theme, group.fact_ids, verified, topic, llm_call
        )
        sections.append(section)

    # Final assembly
    progress("ASSEMBLE", "Generating executive summary, analysis, conclusion...")
    report = await assemble_report(
        sections, topic, title, llm_call,
        total_extracted=len(all_extractions),
        total_verified=len(verified)
    )

    progress("DONE", f"Report complete: {report.verified_count} verified facts in {len(sections)} themes")

    return report
