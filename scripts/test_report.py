"""Test report generation with saved state or mock data.

This script tests report generation in isolation using either:
1. Saved state from test_research.py
2. Mock research findings

Usage:
    python scripts/test_report.py --state test_research_state.json
    python scripts/test_report.py --mock

Output:
    test_report_output.md - Generated report
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load .env file
env_file = project_root / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from staged_config import MINIMAL_CONFIG, print_config_summary

# Mock research data for testing report generation
MOCK_RESEARCH_BRIEF = """Research the latest developments in quantum computing, focusing on:
1. Current hardware approaches (superconducting qubits, trapped ions, photonic)
2. Error correction progress
3. Practical applications and timeline predictions"""

MOCK_FINDINGS = [
    """## Quantum Hardware Progress (IBM)
IBM announced their 1,121-qubit Condor processor in December 2023.
The company claims it represents a significant milestone in scaling.
However, more qubits doesn't directly translate to more computational power
due to error rates and connectivity limitations.""",

    """## Google Quantum AI
Google's Sycamore processor (72 qubits) demonstrated "quantum supremacy"
in 2019 by completing a calculation in 200 seconds that would take
classical supercomputers 10,000 years. Critics debate whether this
represents practical advantage.""",

    """## Error Correction Advances
Microsoft and Quantinuum demonstrated the first "logical qubit" that
operates below the threshold for error correction. This is a critical
milestone for practical quantum computing. Current error rates remain
too high for most practical applications.""",
]

MOCK_SOURCES = [
    {
        "url": "https://research.ibm.com/blog/quantum-computing-roadmap",
        "title": "IBM Quantum Computing Roadmap 2024",
        "content": "IBM plans to deliver 100,000 qubit systems by 2033...",
        "extraction_method": "mock"
    },
    {
        "url": "https://quantumai.google/hardware",
        "title": "Google Quantum AI Hardware",
        "content": "Our Sycamore processor contains 72 qubits...",
        "extraction_method": "mock"
    }
]


async def test_report_with_mock():
    """Generate report from mock data."""
    from langchain_core.messages import HumanMessage
    from langchain.chat_models import init_chat_model

    from open_deep_research.prompts import final_report_generation_prompt
    from open_deep_research.utils import get_today_str

    print("\n" + "=" * 60)
    print("REPORT GENERATION TEST (Mock Data)")
    print("=" * 60)
    print_config_summary()

    # Prepare inputs
    research_brief = MOCK_RESEARCH_BRIEF
    findings = "\n\n".join(MOCK_FINDINGS)

    print(f"\nResearch brief: {len(research_brief)} chars")
    print(f"Findings: {len(findings)} chars ({len(MOCK_FINDINGS)} notes)")

    # Build the prompt
    today = get_today_str()
    prompt = final_report_generation_prompt.format(
        research_brief=research_brief,
        messages="User: Tell me about quantum computing developments",
        findings=findings,
        date=today
    )

    # Add verified findings placeholder (mock - no real verification)
    prompt += """

=== VERIFIED FINDINGS (IMMUTABLE - DO NOT EDIT) ===
## Verified Findings

*Verification was disabled for this test run.*

=== END VERIFIED FINDINGS ===
"""

    print(f"\nPrompt length: {len(prompt)} chars")
    print("\n[1/2] Calling model...")

    # Initialize model
    model_name = MINIMAL_CONFIG.get("final_report_model", "openai:gpt-4.1")
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
    else:
        provider, model = "openai", model_name

    start = time.time()

    try:
        report_model = init_chat_model(
            model=model,
            model_provider=provider,
            max_tokens=MINIMAL_CONFIG.get("final_report_model_max_tokens", 4000),
        )

        response = await report_model.ainvoke([HumanMessage(content=prompt)])
        elapsed = time.time() - start

        report = response.content

        print(f"[2/2] Done in {elapsed:.1f}s")
        print(f"Report length: {len(report)} chars")

        return {
            "success": True,
            "elapsed": elapsed,
            "report": report,
            "prompt_length": len(prompt),
            "brief": research_brief,
            "findings": MOCK_FINDINGS
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "elapsed": time.time() - start,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


async def test_report_with_state(state_path: Path):
    """Generate report from saved state file."""
    from langchain_core.messages import HumanMessage
    from langchain.chat_models import init_chat_model

    from open_deep_research.prompts import final_report_generation_prompt
    from open_deep_research.utils import get_today_str

    print("\n" + "=" * 60)
    print("REPORT GENERATION TEST (From State)")
    print("=" * 60)
    print(f"State file: {state_path}")
    print_config_summary()

    # Load state
    with open(state_path) as f:
        state = json.load(f)

    pipeline_result = state.get("pipeline_result", {})

    research_brief = pipeline_result.get("brief", "")
    notes = pipeline_result.get("notes", [])
    findings = "\n\n".join(notes) if notes else ""

    if not research_brief:
        print("[ERROR] No research brief in state file")
        return {"success": False, "error": "No research brief in state"}

    if not findings:
        print("[WARN] No findings in state file, using raw_notes")
        raw_notes = pipeline_result.get("raw_notes", [])
        findings = "\n\n".join(raw_notes) if raw_notes else ""

    print(f"\nResearch brief: {len(research_brief)} chars")
    print(f"Findings: {len(findings)} chars")

    # Build the prompt
    today = get_today_str()
    prompt = final_report_generation_prompt.format(
        research_brief=research_brief,
        messages=f"User: {state.get('query', 'Research query')}",
        findings=findings,
        date=today
    )

    prompt += """

=== VERIFIED FINDINGS (IMMUTABLE - DO NOT EDIT) ===
## Verified Findings

*Verification was disabled for this test run.*

=== END VERIFIED FINDINGS ===
"""

    print(f"Prompt length: {len(prompt)} chars")
    print("\n[1/2] Calling model...")

    # Initialize model
    model_name = MINIMAL_CONFIG.get("final_report_model", "openai:gpt-4.1")
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
    else:
        provider, model = "openai", model_name

    start = time.time()

    try:
        report_model = init_chat_model(
            model=model,
            model_provider=provider,
            max_tokens=MINIMAL_CONFIG.get("final_report_model_max_tokens", 4000),
        )

        response = await report_model.ainvoke([HumanMessage(content=prompt)])
        elapsed = time.time() - start

        report = response.content

        print(f"[2/2] Done in {elapsed:.1f}s")
        print(f"Report length: {len(report)} chars")

        return {
            "success": True,
            "elapsed": elapsed,
            "report": report,
            "prompt_length": len(prompt),
            "brief": research_brief,
            "findings": notes
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "elapsed": time.time() - start,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


async def main_test(use_mock: bool = True, state_path: Path = None):
    """Run report generation test."""
    print("=" * 60)
    print("REPORT GENERATION TEST")
    print("=" * 60)
    print(f"Mode: {'Mock data' if use_mock else f'State file: {state_path}'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if use_mock:
        result = await test_report_with_mock()
    else:
        result = await test_report_with_state(state_path)

    md = []
    md.append("# Report Generation Test\n")
    md.append(f"**Mode:** {'Mock data' if use_mock else 'State file'}\n")
    md.append(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if result["success"]:
        md.append(f"**Status:** Success\n")
        md.append(f"**Generation time:** {result['elapsed']:.1f}s\n")
        md.append(f"**Prompt length:** {result['prompt_length']} chars\n")
        md.append(f"**Report length:** {len(result['report'])} chars\n")

        md.append("\n---\n## Research Brief\n")
        md.append(f"```\n{result['brief']}\n```\n")

        md.append("\n---\n## Input Findings\n")
        if isinstance(result['findings'], list):
            for i, f in enumerate(result['findings'], 1):
                md.append(f"\n### Finding {i}\n")
                md.append(f"```\n{f[:500]}{'...' if len(f) > 500 else ''}\n```\n")
        else:
            md.append(f"```\n{result['findings'][:1000]}...\n```\n")

        md.append("\n---\n## Generated Report\n")
        md.append(result['report'])

        print("\n" + "=" * 60)
        print("GENERATED REPORT (preview)")
        print("=" * 60)
        print(result['report'][:1500])
        print("..." if len(result['report']) > 1500 else "")

    else:
        md.append(f"\n**Status:** FAILED\n")
        md.append(f"**Error:** {result['error']}\n")
        if "traceback" in result:
            md.append(f"```\n{result['traceback']}\n```\n")
        print(f"\n[ERROR] {result['error']}")

    # Save output
    output_path = project_root / "test_report_output.md"
    output_path.write_text("\n".join(md))
    print(f"\nOutput saved to: {output_path}")

    return result["success"]


def main():
    """Entry point."""
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set")
        sys.exit(1)

    # Parse args
    use_mock = True
    state_path = None

    if len(sys.argv) > 1:
        if sys.argv[1] == "--mock":
            use_mock = True
        elif sys.argv[1] == "--state" and len(sys.argv) > 2:
            use_mock = False
            state_path = Path(sys.argv[2])
            if not state_path.exists():
                print(f"[ERROR] State file not found: {state_path}")
                sys.exit(1)
        else:
            print("Usage:")
            print("  python scripts/test_report.py --mock")
            print("  python scripts/test_report.py --state test_research_state.json")
            sys.exit(1)

    success = asyncio.run(main_test(use_mock=use_mock, state_path=state_path))

    if success:
        print("\nTEST PASSED")
    else:
        print("\nTEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
