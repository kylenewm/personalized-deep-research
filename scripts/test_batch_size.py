"""Quick test for BATCH_SIZE=5 quality.

Runs extraction on 10 sources to verify batching works correctly.
Cost: ~$0.05-0.10
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_deep_research.pipeline_v2 import (
    extract_all_batched,
    batch_sources,
    BATCH_SIZE,
)
from open_deep_research.utils import get_api_key_for_model
from langchain.chat_models import init_chat_model


async def main():
    print(f"Testing BATCH_SIZE={BATCH_SIZE}")

    # Load a fixture with real sources
    fixture_path = Path(__file__).parent.parent / "tests/fixtures/gold_queries/latest_research.json"

    # Use small synthetic sources to minimize cost and time
    print("Using synthetic sources for quick test")
    sources = {
        f"src_{i}": {
            "content": f"""AI Safety Report {i}

OpenAI announced new safety measures in January 2025. The Frontier Model Forum released guidelines for responsible AI deployment. Key metrics include:
- 95% accuracy on safety benchmarks
- 40% reduction in harmful outputs
- 200ms average response latency

Google DeepMind published research on scalable oversight techniques. The EU AI Act became enforceable in February 2025, requiring risk assessments for high-risk AI systems.

RAND Corporation released a security report recommending multi-layered approaches to AI governance. Industry leaders committed to voluntary safety standards at the Munich Security Conference.""",
            "url": f"https://example.com/ai-safety-{i}",
            "title": f"AI Safety Report {i}"
        }
        for i in range(5)  # Just 5 sources = 1 batch
    }

    print(f"Using {len(sources)} sources")

    # Show batching
    batches = batch_sources(sources, BATCH_SIZE)
    print(f"Split into {len(batches)} batches of size {BATCH_SIZE}")

    # Setup LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    model = init_chat_model(
        model="gpt-4.1-mini",
        api_key=api_key,
        max_tokens=4000
    )

    async def llm_call(prompt):
        from langchain_core.messages import HumanMessage
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return response.content

    # Run extraction
    print("\nRunning extraction...")
    topic = "AI safety developments in 2025"

    extractions = await extract_all_batched(
        sources=sources,
        topic=topic,
        llm_call=llm_call,
        batch_size=BATCH_SIZE,
        min_score=0.3,
        on_batch_complete=lambda b, t, e: print(f"  Batch {b}/{t}: {len(e)} extractions")
    )

    # Results
    verified = [e for e in extractions if e.status == "verified"]
    not_found = [e for e in extractions if e.status == "not_found"]

    print(f"\n=== Results ===")
    print(f"Total extractions: {len(extractions)}")
    print(f"Verified: {len(verified)}")
    print(f"Not found: {len(not_found)}")
    print(f"Verification rate: {len(verified)/len(extractions)*100:.1f}%" if extractions else "N/A")

    # Show sample extractions
    print(f"\n=== Sample Verified Extractions ===")
    for e in verified[:5]:
        print(f"- {e.extracted_text[:100]}...")
        print(f"  Score: {e.match_score:.2f}, Method: {e.verification_method}")

    if not_found:
        print(f"\n=== Sample Not Found ===")
        for e in not_found[:3]:
            print(f"- Keywords: {e.pointer.keywords}")
            print(f"  Reason: {e.failure_reason}")


if __name__ == "__main__":
    asyncio.run(main())
