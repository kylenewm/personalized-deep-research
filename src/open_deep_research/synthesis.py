"""Synthesis layer: Connect verified facts into readable prose.

This module takes verified extractions and synthesizes them into
a coherent report. The key principle:
- Verified facts are LOCKED (styled differently)
- LLM writes ONLY transitions and introductions
- Reader can visually distinguish verified from synthesized
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .pointer_extract import Extraction


@dataclass
class SynthesisBlock:
    """A block in the synthesized output."""
    type: str  # "verified", "transition", "intro", "conclusion"
    content: str
    source_url: Optional[str] = None
    context: Optional[str] = None


@dataclass
class SynthesizedReport:
    """A complete synthesized report."""
    title: str
    blocks: List[SynthesisBlock] = field(default_factory=list)

    @property
    def verified_count(self) -> int:
        return sum(1 for b in self.blocks if b.type == "verified")

    @property
    def synthesis_count(self) -> int:
        return sum(1 for b in self.blocks if b.type in ("transition", "intro", "conclusion"))


# Prompt for synthesis
SYNTHESIS_PROMPT = '''You are synthesizing verified research findings into a coherent report.

You have these VERIFIED FACTS (extracted directly from sources - do not modify):

{verified_facts}

Your task:
1. Write a brief INTRO (2-3 sentences) introducing the topic
2. For each fact, write a TRANSITION (1 sentence) connecting it to the previous
3. Write a brief CONCLUSION (2-3 sentences) summarizing key takeaways

CRITICAL RULES:
- Do NOT rewrite or paraphrase the verified facts
- Do NOT add new factual claims in transitions
- Transitions should ONLY connect ideas, not add information
- Keep transitions short (1 sentence max)

Output format (JSON):
{{
  "intro": "Your introduction here...",
  "transitions": [
    "First transition connecting to fact 1...",
    "Transition connecting fact 1 to fact 2...",
    ...
  ],
  "conclusion": "Your conclusion here..."
}}

Output ONLY valid JSON.'''


def format_facts_for_synthesis(extractions: List[Extraction]) -> str:
    """Format verified extractions for the synthesis prompt."""
    lines = []
    for i, ext in enumerate(extractions, 1):
        if ext.status == "verified" and ext.extracted_text:
            lines.append(f"FACT {i}: {ext.extracted_text}")
            lines.append(f"  Source: {ext.pointer.context}")
            lines.append("")
    return "\n".join(lines)


def parse_synthesis_response(response: str, num_facts: int) -> dict:
    """Parse LLM synthesis response."""
    try:
        # Find JSON in response
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            data = json.loads(match.group())
            return {
                "intro": data.get("intro", ""),
                "transitions": data.get("transitions", [""] * num_facts),
                "conclusion": data.get("conclusion", "")
            }
    except json.JSONDecodeError:
        pass

    return {
        "intro": "",
        "transitions": [""] * num_facts,
        "conclusion": ""
    }


def synthesize_report(
    extractions: List[Extraction],
    title: str,
    synthesis_data: dict
) -> SynthesizedReport:
    """Combine verified extractions with synthesis into a report.

    Args:
        extractions: List of verified extractions
        title: Report title
        synthesis_data: Dict with intro, transitions, conclusion from LLM

    Returns:
        SynthesizedReport with interleaved blocks
    """
    report = SynthesizedReport(title=title)

    # Add intro
    if synthesis_data.get("intro"):
        report.blocks.append(SynthesisBlock(
            type="intro",
            content=synthesis_data["intro"]
        ))

    # Interleave transitions and facts
    verified = [e for e in extractions if e.status == "verified" and e.extracted_text]
    transitions = synthesis_data.get("transitions", [])

    for i, ext in enumerate(verified):
        # Add transition before fact (except first one)
        if i < len(transitions) and transitions[i]:
            report.blocks.append(SynthesisBlock(
                type="transition",
                content=transitions[i]
            ))

        # Add verified fact
        report.blocks.append(SynthesisBlock(
            type="verified",
            content=ext.extracted_text,
            source_url=ext.source_url,
            context=ext.pointer.context
        ))

    # Add conclusion
    if synthesis_data.get("conclusion"):
        report.blocks.append(SynthesisBlock(
            type="conclusion",
            content=synthesis_data["conclusion"]
        ))

    return report


def render_report_markdown(report: SynthesizedReport, use_color: bool = True) -> str:
    """Render synthesized report as markdown with color styling.

    Args:
        report: The synthesized report
        use_color: If True, use HTML styling for verified vs synthesized

    Returns:
        Markdown string
    """
    lines = [f"# {report.title}\n"]

    # Color styles
    if use_color:
        verified_style = 'style="background: #dcfce7; padding: 8px; border-left: 3px solid #16a34a; margin: 8px 0;"'
        synth_style = 'style="color: #6b7280; font-style: italic;"'

    for block in report.blocks:
        if block.type == "verified":
            if use_color:
                lines.append(f'<div {verified_style}>')
                lines.append(f'{block.content}')
                if block.source_url:
                    lines.append(f'<br><small><a href="{block.source_url}">[Source: {block.context}]</a></small>')
                lines.append('</div>\n')
            else:
                lines.append(f"> {block.content}")
                if block.source_url:
                    lines.append(f"> — [{block.context}]({block.source_url})")
                lines.append("")

        elif block.type in ("intro", "conclusion"):
            if use_color:
                lines.append(f'<p {synth_style}>{block.content}</p>\n')
            else:
                lines.append(f"*{block.content}*\n")

        elif block.type == "transition":
            if use_color:
                lines.append(f'<p {synth_style}>{block.content}</p>\n')
            else:
                lines.append(f"{block.content}\n")

    # Stats footer
    lines.append("\n---\n")
    lines.append(f"**Report Statistics:**")
    lines.append(f"- Verified facts: {report.verified_count}")
    lines.append(f"- Synthesized sections: {report.synthesis_count}")
    if use_color:
        lines.append(f"- <span style='background: #dcfce7; padding: 2px 6px;'>Green = verified from source</span>")
        lines.append(f"- <span style='color: #6b7280; font-style: italic;'>Gray italic = AI synthesis</span>")

    return "\n".join(lines)


def render_report_plain(report: SynthesizedReport) -> str:
    """Render report as plain text with markers."""
    lines = [f"{report.title}", "=" * len(report.title), ""]

    for block in report.blocks:
        if block.type == "verified":
            lines.append(f"[VERIFIED] {block.content}")
            if block.context:
                lines.append(f"           Source: {block.context}")
            lines.append("")

        elif block.type in ("intro", "conclusion"):
            lines.append(f"[{block.type.upper()}] {block.content}")
            lines.append("")

        elif block.type == "transition":
            lines.append(f"{block.content}")
            lines.append("")

    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    from .pointer_extract import Pointer, Extraction

    # Mock extractions
    test_extractions = [
        Extraction(
            pointer=Pointer("src_001", ["RAND", "security"], "RAND security report"),
            status="verified",
            extracted_text="The RAND Corporation released a comprehensive security report in October 2025 recommending multi-layered security approaches.",
            match_score=1.0,
            source_url="https://rand.org/report"
        ),
        Extraction(
            pointer=Pointer("src_002", ["OpenAI", "safety"], "OpenAI initiatives"),
            status="verified",
            extracted_text="OpenAI announced the formation of the Frontier Risk Council to oversee safe model deployment.",
            match_score=1.0,
            source_url="https://openai.com/safety"
        ),
    ]

    # Mock synthesis
    synthesis = {
        "intro": "This report examines recent developments in AI safety from leading organizations.",
        "transitions": [
            "Among the key findings,",
            "Building on these recommendations,"
        ],
        "conclusion": "These developments signal a maturing approach to AI safety across the industry."
    }

    report = synthesize_report(test_extractions, "AI Safety Developments", synthesis)

    print("=== Plain Text ===\n")
    print(render_report_plain(report))

    print("\n=== Markdown (no color) ===\n")
    print(render_report_markdown(report, use_color=False))
