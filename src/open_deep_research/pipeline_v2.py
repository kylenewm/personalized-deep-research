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
        extract_from_pointer,
        format_sources_for_prompt,
        parse_pointer_response,
    )
except ImportError:
    from pointer_extract import (
        Pointer,
        Extraction,
        POINTER_PROMPT,
        extract_from_pointer,
        format_sources_for_prompt,
        parse_pointer_response,
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

BATCH_SIZE = 10  # Sources per batch (smaller for better extraction)
MAX_CHARS_PER_SOURCE = 5000  # More content per source for better matching
DEFAULT_MIN_SCORE = 0.3  # Lower threshold - need 30% of keywords to match


def batch_sources(sources: Dict[str, dict], batch_size: int = BATCH_SIZE) -> List[Dict[str, dict]]:
    """Split sources into batches."""
    items = list(sources.items())
    batches = []
    for i in range(0, len(items), batch_size):
        batch = dict(items[i:i + batch_size])
        batches.append(batch)
    return batches


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
    # Format sources for prompt
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
# Stage 2: Arranger (Grouping + Curation)
# =============================================================================

ARRANGER_PROMPT = '''You are organizing research findings into a coherent structure.

Research Topic: {topic}

You have {num_facts} verified facts to organize. Your tasks:
1. GROUP facts by theme (4-6 themes)
2. DROP facts that are redundant, off-topic, or low-value (~30-50%)
3. ORDER facts within each theme for logical flow

VERIFIED FACTS:
{facts}

Output a JSON object with:
- "groups": array of {{"theme": "Theme Name", "fact_ids": [1, 2, 5, ...]}}
- "excluded": array of fact IDs to drop (with brief reason)

Theme names should be 2-4 words, like:
- "Governance & Regulation"
- "Technical Safety Measures"
- "Industry Initiatives"
- "Research & Evaluation"

CRITICAL:
- Each fact_id should appear in exactly ONE group OR in excluded
- Keep at least 50% of facts (drop at most 50%)
- Order fact_ids within each group for narrative flow

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
    """Render hybrid report as markdown."""
    lines = [f"# {report.title}\n"]

    # Styles
    if use_color:
        gray = 'style="color: #6b7280; font-style: italic;"'
        green = 'style="background: #dcfce7; padding: 8px; border-left: 3px solid #16a34a; margin: 8px 0;"'

    # Executive Summary (gray)
    lines.append("## Executive Summary\n")
    if use_color:
        lines.append(f'<p {gray}>{report.executive_summary}</p>\n')
    else:
        lines.append(f"*{report.executive_summary}*\n")

    # Themed sections
    lines.append("## Verified Findings\n")

    for section in report.sections:
        lines.append(f"### {section.theme}\n")

        # Theme intro (gray)
        if section.intro:
            if use_color:
                lines.append(f'<p {gray}>{section.intro}</p>\n')
            else:
                lines.append(f"*{section.intro}*\n")

        # Facts with transitions
        for i, fact in enumerate(section.facts):
            # Transition (gray)
            if i < len(section.transitions) and section.transitions[i]:
                if use_color:
                    lines.append(f'<p {gray}>{section.transitions[i]}</p>\n')
                else:
                    lines.append(f"*{section.transitions[i]}*\n")

            # Verified fact (green)
            if use_color:
                lines.append(f'<div {green}>')
                lines.append(fact.extracted_text)
                if fact.source_url:
                    lines.append(f'<br><small><a href="{fact.source_url}">[Source: {fact.pointer.context}]</a></small>')
                lines.append('</div>\n')
            else:
                lines.append(f"> {fact.extracted_text}")
                if fact.source_url:
                    lines.append(f"> — [{fact.pointer.context}]({fact.source_url})")
                lines.append("")

    # Analysis section (gray)
    lines.append("## Analysis & Implications\n")
    if use_color:
        # Split into paragraphs
        paragraphs = report.analysis.split('\n\n')
        for p in paragraphs:
            if p.strip():
                lines.append(f'<p {gray}>{p.strip()}</p>\n')
    else:
        lines.append(f"*{report.analysis}*\n")

    # Conclusion (gray)
    lines.append("## Conclusion\n")
    if use_color:
        lines.append(f'<p {gray}>{report.conclusion}</p>\n')
    else:
        lines.append(f"*{report.conclusion}*\n")

    # Stats footer
    lines.append("\n---\n")
    lines.append("**Report Statistics:**")
    lines.append(f"- Sources processed: {report.total_extracted}")
    lines.append(f"- Verified facts: {report.total_verified}")
    lines.append(f"- Facts in report: {report.total_used}")
    lines.append(f"- Themes: {len(report.sections)}")
    if use_color:
        lines.append("- <span style='background: #dcfce7; padding: 2px 6px;'>Green = verified from source</span>")
        lines.append("- <span style='color: #6b7280; font-style: italic;'>Gray italic = AI synthesis</span>")

    return "\n".join(lines)


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
