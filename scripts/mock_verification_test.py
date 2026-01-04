"""
Mock Verification Test - Isolates verification logic from API issues.

Uses hardcoded sources and claims with KNOWN ground truth to test
whether the verification pipeline correctly identifies supported vs
unsupported claims.

Usage:
    source venv/bin/activate
    python scripts/mock_verification_test.py

Output:
    mock_verification_report.md - Detailed analysis of each step
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any

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

# =============================================================================
# MOCK DATA - Sources with known content
# =============================================================================

MOCK_SOURCES = [
    {
        "url": "https://example.com/openai-gpt4",
        "title": "OpenAI Releases GPT-4",
        "content": """OpenAI officially released GPT-4 on March 14, 2023.
The model demonstrated significant improvements over GPT-3.5, including
better reasoning capabilities and multimodal support for images.
CEO Sam Altman announced the release at a press conference in San Francisco.
The model achieved 90% on the bar exam and supports 25,000 tokens of context.""",
        "query": "GPT-4 release",
        "timestamp": "2024-01-01T00:00:00",
    },
    {
        "url": "https://example.com/anthropic-claude3",
        "title": "Anthropic Announces Claude 3",
        "content": """Anthropic unveiled Claude 3 on March 4, 2024.
The Claude 3 family includes three variants: Haiku, Sonnet, and Opus.
CEO Dario Amodei stated the models achieve state-of-the-art performance
on multiple benchmarks including MMLU and HumanEval.
Opus is the most capable model, while Haiku is optimized for speed.""",
        "query": "Claude 3 announcement",
        "timestamp": "2024-01-01T00:00:00",
    },
    {
        "url": "https://example.com/transformer-paper",
        "title": "Attention Is All You Need Paper",
        "content": """The transformer architecture was introduced in the paper
"Attention Is All You Need" published in 2017 by Vaswani et al.
The paper proposed the self-attention mechanism as an alternative to
recurrent neural networks. Google Brain researchers demonstrated that
transformers could achieve superior results on machine translation tasks.""",
        "query": "transformer architecture",
        "timestamp": "2024-01-01T00:00:00",
    },
    {
        "url": "https://example.com/meta-llama",
        "title": "Meta Releases Llama 2",
        "content": """Meta released Llama 2 in July 2023 as an open-source model.
The model is available in 7B, 13B, and 70B parameter variants.
Meta partnered with Microsoft for distribution through Azure.
Llama 2 can be used for commercial purposes under Meta's license.""",
        "query": "Llama 2 release",
        "timestamp": "2024-01-01T00:00:00",
    },
    {
        "url": "https://example.com/google-gemini",
        "title": "Google Announces Gemini",
        "content": """Google DeepMind announced Gemini on December 6, 2023.
The model comes in three sizes: Ultra, Pro, and Nano.
Gemini Ultra achieved a score of 90% on MMLU benchmark.
CEO Sundar Pichai described it as Google's most capable AI model.""",
        "query": "Gemini announcement",
        "timestamp": "2024-01-01T00:00:00",
    },
]

# =============================================================================
# MOCK CLAIMS - With expected outcomes
# =============================================================================

MOCK_CLAIMS = [
    # TRUE CLAIMS - Should be SUPPORTED
    {
        "claim": "OpenAI released GPT-4 on March 14, 2023.",
        "expected": "SUPPORTED",
        "reason": "Exact match in Source 1",
    },
    {
        "claim": "The Claude 3 family includes Haiku, Sonnet, and Opus variants.",
        "expected": "SUPPORTED",
        "reason": "Exact match in Source 2",
    },
    {
        "claim": "The transformer architecture was introduced in 2017.",
        "expected": "SUPPORTED",
        "reason": "Match in Source 3",
    },
    # FALSE CLAIMS - Should be UNSUPPORTED
    {
        "claim": "Google released Gemini 2.0 in December 2024.",
        "expected": "UNSUPPORTED",
        "reason": "Source 5 says Gemini (not 2.0) in December 2023 (not 2024)",
    },
    {
        "claim": "OpenAI released GPT-4 in January 2023.",
        "expected": "UNSUPPORTED",
        "reason": "Source 1 says March 14, 2023 (not January)",
    },
]


async def run_mock_test():
    """Run the mock verification test and generate detailed report."""

    md = []
    md.append("# Mock Verification Test Report\n")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("\nThis test uses hardcoded sources and claims with known ground truth.\n")

    # ==========================================================================
    # SECTION 1: Test Data Overview
    # ==========================================================================
    md.append("\n---\n## 1. Test Data\n")

    md.append("\n### Sources (5)\n")
    md.append("| # | Title | Content Length | Key Facts |\n")
    md.append("|---|-------|----------------|------------|\n")
    for i, source in enumerate(MOCK_SOURCES, 1):
        key_fact = source["content"].split(".")[0][:50] + "..."
        md.append(f"| {i} | {source['title']} | {len(source['content'])} chars | {key_fact} |\n")

    md.append("\n### Claims (5)\n")
    md.append("| # | Claim | Expected | Why |\n")
    md.append("|---|-------|----------|-----|\n")
    for i, claim_data in enumerate(MOCK_CLAIMS, 1):
        md.append(f"| {i} | {claim_data['claim'][:50]}... | {claim_data['expected']} | {claim_data['reason'][:30]}... |\n")

    # ==========================================================================
    # SECTION 2: Import and run verification functions
    # ==========================================================================
    md.append("\n---\n## 2. Embedding + Entity Matching\n")

    try:
        from open_deep_research.verification import (
            find_relevant_passages,
            extract_entities,
            cosine_similarity,
            get_embeddings,
        )

        embeddings = get_embeddings()

        # Store results for later
        matching_results = []

        for i, claim_data in enumerate(MOCK_CLAIMS, 1):
            claim = claim_data["claim"]
            md.append(f"\n### Claim {i}: \"{claim}\"\n")

            # Extract entities from claim
            claim_entities = extract_entities(claim)
            md.append(f"\n**Claim entities:** {', '.join(sorted(claim_entities)) if claim_entities else 'None'}\n")

            # Get embeddings and scores for each source
            md.append("\n| Source | Embed Score | Source Entities | Overlap | Final Score |\n")
            md.append("|--------|-------------|-----------------|---------|-------------|\n")

            # Run the actual matching
            top_sources = await find_relevant_passages(claim, MOCK_SOURCES, top_k=3)

            # Also compute scores for ALL sources (for the table)
            claim_embedding = await embeddings.aembed_query(claim)

            source_scores = []
            for source in MOCK_SOURCES:
                content = source.get("content", "")
                title = source.get("title", "")

                # Get embedding for source
                source_text = f"{title} {content[:1000]}"
                source_embedding = await embeddings.aembed_query(source_text)
                embed_score = cosine_similarity(claim_embedding, source_embedding)

                # Get entities
                source_entities = extract_entities(title + " " + content)
                overlap = claim_entities & source_entities
                entity_bonus = min(0.3, len(overlap) * 0.05)

                final_score = embed_score + entity_bonus
                source_scores.append((source, embed_score, source_entities, overlap, final_score))

            # Sort and display
            source_scores.sort(key=lambda x: x[4], reverse=True)
            for source, embed_score, source_ents, overlap, final_score in source_scores:
                overlap_str = ', '.join(sorted(overlap)[:3]) if overlap else '-'
                md.append(f"| {source['title'][:25]}... | {embed_score:.3f} | {len(source_ents)} | {overlap_str} | {final_score:.3f} |\n")

            # Show winner
            if top_sources:
                winner = top_sources[0]
                md.append(f"\n**Winner:** {winner[0]['title']} (score: {winner[2]:.3f})\n")
                md.append(f"\n**Selected passage:**\n```\n{winner[1][:300]}...\n```\n")
                matching_results.append((claim_data, winner))
            else:
                md.append(f"\n**Winner:** No match found\n")
                matching_results.append((claim_data, None))

        # ==========================================================================
        # SECTION 3: LLM Verification
        # ==========================================================================
        md.append("\n---\n## 3. LLM Verification\n")
        md.append("\nThis step uses an LLM to judge if the passage supports the claim.\n")

        from open_deep_research.verification import verify_single_claim, VerificationVote
        from langchain.chat_models import init_chat_model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            md.append("\n**ERROR:** OPENAI_API_KEY not set. Skipping LLM verification.\n")
            verification_results = []
        else:
            verification_llm = init_chat_model(
                model="gpt-4o-mini",
                api_key=api_key,
            ).with_structured_output(VerificationVote)

            verification_results = []

            for i, (claim_data, match) in enumerate(matching_results, 1):
                claim = claim_data["claim"]
                md.append(f"\n### Claim {i}: \"{claim[:50]}...\"\n")

                if match is None:
                    md.append("\n**No source match - marking as UNCERTAIN**\n")
                    verification_results.append({
                        "claim": claim_data,
                        "status": "UNCERTAIN",
                        "confidence": 0.0,
                        "evidence": "No matching source",
                    })
                    continue

                source, passage, score = match

                md.append(f"\n**Input to LLM:**\n")
                md.append(f"```\nCLAIM: {claim}\n\nSOURCE PASSAGE:\n{passage[:400]}...\n\nSOURCE URL: {source['url']}\n```\n")

                # Run verification
                result = await verify_single_claim(
                    claim=claim,
                    passage=passage,
                    source=source,
                    llm=verification_llm,
                    claim_id=f"claim_{i}"
                )

                md.append(f"\n**LLM Decision:**\n")
                md.append(f"- **Status:** {result['status']}\n")
                md.append(f"- **Confidence:** {result['confidence']:.2f}\n")
                md.append(f"- **Evidence:** \"{result.get('evidence_snippet', 'N/A')}\"\n")

                verification_results.append({
                    "claim": claim_data,
                    "status": result["status"],
                    "confidence": result["confidence"],
                    "evidence": result.get("evidence_snippet", ""),
                })

        # ==========================================================================
        # SECTION 4: Summary
        # ==========================================================================
        md.append("\n---\n## 4. Summary\n")

        md.append("\n| # | Claim | Expected | Actual | Confidence | Match? |\n")
        md.append("|---|-------|----------|--------|------------|--------|\n")

        correct = 0
        for i, vr in enumerate(verification_results, 1):
            expected = vr["claim"]["expected"]
            actual = vr["status"]
            conf = vr["confidence"]

            # Check if result matches expectation
            # SUPPORTED matches SUPPORTED
            # UNSUPPORTED/UNCERTAIN matches UNSUPPORTED
            if expected == "SUPPORTED":
                match = actual == "SUPPORTED"
            else:
                match = actual in ["UNSUPPORTED", "UNCERTAIN"]

            if match:
                correct += 1

            match_str = "**pass**" if match else "**FAIL**"
            md.append(f"| {i} | {vr['claim']['claim'][:40]}... | {expected} | {actual} | {conf:.2f} | {match_str} |\n")

        total = len(verification_results)
        accuracy = correct / total * 100 if total > 0 else 0

        md.append(f"\n### Results\n")
        md.append(f"- **Correct:** {correct}/{total}\n")
        md.append(f"- **Accuracy:** {accuracy:.0f}%\n")

        if accuracy == 100:
            md.append(f"\n**Verification logic is working correctly on mock data.**\n")
        else:
            md.append(f"\n**Some tests failed - verification logic needs investigation.**\n")
            md.append("\n### Analysis of Failures\n")
            for i, vr in enumerate(verification_results, 1):
                expected = vr["claim"]["expected"]
                actual = vr["status"]
                if expected == "SUPPORTED" and actual != "SUPPORTED":
                    md.append(f"- Claim {i}: Expected SUPPORTED but got {actual}. The LLM may be too strict.\n")
                elif expected == "UNSUPPORTED" and actual == "SUPPORTED":
                    md.append(f"- Claim {i}: Expected UNSUPPORTED but got SUPPORTED. The LLM may be too lenient.\n")

    except Exception as e:
        import traceback
        md.append(f"\n**ERROR:** {e}\n")
        md.append(f"```\n{traceback.format_exc()}\n```\n")

    # ==========================================================================
    # Write report
    # ==========================================================================
    report = "\n".join(md)
    output_path = project_root / "mock_verification_report.md"
    output_path.write_text(report)
    print(f"\nReport written to: {output_path}")
    print(f"\nSummary: {correct}/{total} tests passed ({accuracy:.0f}% accuracy)")

    return report


def main():
    """Entry point."""
    # Check for OpenAI key
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set")
        sys.exit(1)

    print("=" * 60)
    print("Mock Verification Test")
    print("=" * 60)
    print(f"Sources: {len(MOCK_SOURCES)}")
    print(f"Claims: {len(MOCK_CLAIMS)} ({sum(1 for c in MOCK_CLAIMS if c['expected'] == 'SUPPORTED')} true, {sum(1 for c in MOCK_CLAIMS if c['expected'] == 'UNSUPPORTED')} false)")
    print("=" * 60)

    asyncio.run(run_mock_test())


if __name__ == "__main__":
    main()
