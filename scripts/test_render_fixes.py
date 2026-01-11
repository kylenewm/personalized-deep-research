#!/usr/bin/env python3
"""Test render fixes without LLM calls - uses mock data."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from open_deep_research.pipeline_v2 import HybridReport, ThemedSection, Citation
from open_deep_research.pointer_extract import Pointer, Extraction
from open_deep_research.render import render_report


def create_mock_extraction(text: str, url: str, context: str, score: float = 0.8) -> Extraction:
    """Create a mock Extraction object."""
    pointer = Pointer(
        source_id="test",
        keywords=text.split()[:5],  # First 5 words as keywords
        context=context
    )
    return Extraction(
        pointer=pointer,
        extracted_text=text,
        source_url=url,
        match_score=score,
        status="verified"
    )


def create_mock_report() -> tuple[HybridReport, List[Extraction]]:
    """Create a mock report with excluded facts to test render fixes."""

    # Create some facts for themes
    fact1 = create_mock_extraction(
        text="Multi-agent orchestration using upstream routing layers allows specialized downstream agents to handle specific tasks with appropriate context and logic at every turn.",
        url="https://deepgram.com/webinars/orchestrating-multi-agent-voice-ai-systems",
        context="Deepgram multi-agent orchestration"
    )

    fact2 = create_mock_extraction(
        text="Real-time monitoring of latency, throughput, token consumption for cost control, and model error rates is critical for production voice agent systems.",
        url="https://research.aimultiple.com/llm-orchestration/",
        context="LLM orchestration observability"
    )

    fact3 = create_mock_extraction(
        text="TEN Framework emphasizes extensibility, turn detection, and agent composability for building modular voice AI systems that can scale across multiple use cases.",
        url="https://medium.com/@mahadise0011/top-voice-ai-agent-frameworks-in-2026",
        context="TEN Framework modular architecture"
    )

    fact4 = create_mock_extraction(
        text="Bolna AI is well-suited for turn-based voice agents and domain-specific assistants where conversation flow follows predictable patterns with clear handoffs between states.",
        url="https://medium.com/@mahadise0011/top-voice-ai-agent-frameworks-in-2026",
        context="Bolna AI turn-based agents"
    )

    # Create a long fact to test truncation fix
    fact5 = create_mock_extraction(
        text="Retrieval-Augmented Generation (RAG) pipelines combined with embedding-based vector search enable voice agents to access enterprise knowledge bases in real-time, supporting multi-modal inputs including text, code, and images while maintaining low latency response times critical for conversational interfaces. This architectural pattern has emerged as a best practice for production deployments where agents need access to constantly updated information sources.",
        url="https://research.aimultiple.com/llm-orchestration/",
        context="RAG and multi-modal support"
    )

    # Create sections with citations
    section1 = ThemedSection(
        theme="Multi-Agent Orchestration Architecture",
        facts=[fact1, fact2],
        prose="Modern voice agent systems increasingly rely on multi-agent orchestration patterns. [1] This approach enables better separation of concerns and more maintainable systems. [2] The trend is accelerating across the industry.",
        citations=[
            Citation(marker="[1]", fact_index=0),
            Citation(marker="[2]", fact_index=1),
        ]
    )

    section2 = ThemedSection(
        theme="Framework Ecosystem and Tools",
        facts=[fact3, fact4, fact5],
        prose="Several frameworks have emerged for building voice agents. [1] Turn-based conversation handling is a key capability. [2] RAG integration enables knowledge-grounded responses. [3]",
        citations=[
            Citation(marker="[1]", fact_index=0),
            Citation(marker="[2]", fact_index=1),
            Citation(marker="[3]", fact_index=2),
        ]
    )

    # Create excluded facts (verified but not used in themes)
    excluded1 = create_mock_extraction(
        text="Voice agent latency should target sub-500ms response times for natural conversational flow, with optimal systems achieving 200-300ms end-to-end latency.",
        url="https://voiceflow.com/blog/voice-agent-latency",
        context="Latency requirements"
    )

    excluded2 = create_mock_extraction(
        text="WebSocket-based architectures outperform REST APIs for real-time voice streaming, reducing connection overhead and enabling bidirectional communication essential for interruption handling.",
        url="https://livekit.io/blog/voice-ai-architecture",
        context="WebSocket architecture"
    )

    report = HybridReport(
        title="Voice Agent Orchestration Methods in 2026",
        executive_summary="This report examines the best methods for complex orchestration of voice agents in 2026, covering multi-agent architectures, framework ecosystems, and production best practices.",
        sections=[section1, section2],
        analysis="The voice agent landscape in 2026 shows maturation toward multi-agent orchestration patterns that enable better scalability and maintainability.",
        conclusion="Organizations building voice agents should adopt multi-agent architectures with proper observability and leverage emerging frameworks like TEN and Bolna AI.",
        total_extracted=10,
        total_verified=7,
        total_used=5,
        excluded_facts=[excluded1, excluded2]
    )

    return report, [excluded1, excluded2]


def main():
    print("=" * 70)
    print("TESTING RENDER FIXES")
    print("=" * 70)

    report, excluded = create_mock_report()

    print(f"\n[MOCK DATA]")
    print(f"  Sections: {len(report.sections)}")
    print(f"  Facts in themes: {report.verified_count}")
    print(f"  Excluded facts: {len(report.excluded_facts)}")

    # Render
    html = render_report(report, excluded_facts=report.excluded_facts)

    # Save
    output_path = project_root / "test_render_fixes.html"
    output_path.write_text(html)
    print(f"\n[SAVED] {output_path}")

    # Verify fixes
    print("\n[VERIFICATION]")

    # Check 1: Additional Sources section
    if "Additional Sources" in html:
        print("  [PASS] Additional Sources section present")
    else:
        print("  [FAIL] Additional Sources section MISSING")

    # Check 2: Excluded fact content appears
    if "WebSocket-based architectures" in html:
        print("  [PASS] Excluded fact content appears in footnotes")
    else:
        print("  [FAIL] Excluded fact content MISSING")

    # Check 3: Long fact not truncated (the RAG fact is > 200 chars)
    long_phrase = "constantly updated information sources"
    if long_phrase in html:
        print("  [PASS] Long evidence text NOT truncated")
    else:
        print("  [FAIL] Long evidence text was truncated")

    # Check 4: [u] markers for unverified sentences
    if 'class="unverified"' in html:
        print("  [PASS] Unverified sentence markers present")
    else:
        print("  [FAIL] Unverified sentence markers MISSING")

    # Count footnotes
    fn_count = html.count('class="footnote"')
    print(f"\n[STATS]")
    print(f"  Total footnotes: {fn_count}")
    print(f"  Expected: {report.verified_count + len(excluded)} ({report.verified_count} cited + {len(excluded)} excluded)")

    print("\n" + "=" * 70)
    print("Open test_render_fixes.html to visually verify the changes")
    print("=" * 70)


if __name__ == "__main__":
    main()
