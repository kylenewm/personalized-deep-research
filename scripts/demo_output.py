#!/usr/bin/env python3
"""Demo output rendering with mock data.

Shows what a synthesized report looks like - NO API calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_deep_research.pointer_extract import Pointer, Extraction
from open_deep_research.synthesis import (
    synthesize_report,
    render_report_markdown,
    render_report_plain,
)


def main():
    # Realistic mock extractions (simulating what we'd get from real sources)
    extractions = [
        Extraction(
            pointer=Pointer("src_001", ["RAND", "security", "October 2025"], "RAND security framework"),
            status="verified",
            extracted_text="The RAND Corporation released a comprehensive security report in October 2025 recommending multi-layered security approaches for frontier AI systems, including tiered access controls and continuous monitoring.",
            match_score=1.0,
            source_url="https://rand.org/pubs/research_reports/RRA4159-1.html"
        ),
        Extraction(
            pointer=Pointer("src_002", ["OpenAI", "Frontier Risk Council", "defense-in-depth"], "OpenAI safety measures"),
            status="verified",
            extracted_text="OpenAI announced the formation of the Frontier Risk Council to advise on balancing capability and misuse risk, alongside defense-in-depth strategies including intensified red teaming with external experts.",
            match_score=0.95,
            source_url="https://openai.com/index/strengthening-cyber-resilience/"
        ),
        Extraction(
            pointer=Pointer("src_003", ["LLM-judges", "safety evaluations", "biases"], "Evaluation benchmarks"),
            status="verified",
            extracted_text="LLM-based judges have become the default for safety evaluations, with recent research addressing their known biases through calibration techniques and multi-model consensus approaches.",
            match_score=0.9,
            source_url="https://aievaluation.substack.com/p/2025-december-ai-evaluation-digest"
        ),
        Extraction(
            pointer=Pointer("src_004", ["Singapore", "AI Safety Institute", "governance"], "Singapore initiatives"),
            status="verified",
            extracted_text="Singapore established the AI Safety Institute under IMDA to lead regional AI governance initiatives, focusing on developing practical safety frameworks for enterprise deployment.",
            match_score=0.85,
            source_url="https://www.imda.gov.sg/ai-safety"
        ),
        Extraction(
            pointer=Pointer("src_005", ["chain-of-thought", "monitoring", "transparency"], "CoT monitoring"),
            status="verified",
            extracted_text="Chain-of-thought monitoring has emerged as a leading transparency technique, with major labs implementing real-time reasoning inspection to detect potential misalignment.",
            match_score=0.92,
            source_url="https://techcrunch.com/2025/07/15/research-leaders-urge-cot-monitoring"
        ),
    ]

    # Mock synthesis (simulating LLM output)
    synthesis = {
        "intro": "The AI safety landscape in 2025 has seen significant developments across technical, organizational, and regulatory dimensions. This report synthesizes key findings from leading institutions and research organizations.",
        "transitions": [
            "On the technical front,",
            "Complementing these technical measures,",
            "The evaluation ecosystem has also evolved significantly.",
            "Internationally,",
            "Finally, transparency mechanisms have gained prominence.",
        ],
        "conclusion": "These developments indicate a maturing approach to AI safety, with increased coordination between technical research, organizational governance, and regulatory frameworks. The emphasis on verifiable safety measures suggests the field is moving beyond aspirational principles toward actionable practices."
    }

    # Build report
    report = synthesize_report(
        extractions,
        "AI Safety Developments: 2025 Overview",
        synthesis
    )

    # Render and display
    print("=" * 70)
    print("PLAIN TEXT OUTPUT")
    print("=" * 70)
    print()
    print(render_report_plain(report))

    print()
    print("=" * 70)
    print("MARKDOWN OUTPUT (saved to demo_output.md)")
    print("=" * 70)
    print()

    md_output = render_report_markdown(report, use_color=True)
    print(md_output[:1500] + "\n...[truncated]...")

    # Save full markdown
    output_path = Path(__file__).parent.parent / "demo_output.md"
    output_path.write_text(md_output)
    print(f"\n[Full output saved to {output_path}]")

    # Stats
    print()
    print("=" * 70)
    print("REPORT STATISTICS")
    print("=" * 70)
    print(f"  Verified facts:      {report.verified_count}")
    print(f"  Synthesized blocks:  {report.synthesis_count}")
    print(f"  Total blocks:        {len(report.blocks)}")
    print()
    print("  Legend:")
    print("    Green background = Verified (extracted from source)")
    print("    Gray italic = Synthesized (AI-written transitions)")
    print("=" * 70)


if __name__ == "__main__":
    main()
