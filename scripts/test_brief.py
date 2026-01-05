"""Test brief generation only.

This script tests just the brief generation step in isolation.
Shows exactly what research plan the LLM creates for a given query.

Usage:
    python scripts/test_brief.py "What are the latest AI models?"
    python scripts/test_brief.py  # Uses default query

Output:
    test_brief_output.md - The generated brief
"""

import asyncio
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

# Default test query
DEFAULT_QUERY = "What are the latest developments in AI safety?"


async def test_brief(query: str):
    """Test brief generation for a query."""
    from langchain_core.messages import HumanMessage
    from langchain.chat_models import init_chat_model

    from open_deep_research.state import ResearchQuestion
    from open_deep_research.prompts import transform_messages_into_research_topic_prompt

    print("=" * 60)
    print("BRIEF GENERATION TEST")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_config_summary()

    # Build the prompt (same as write_research_brief does)
    today = datetime.now().strftime("%Y-%m-%d")
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=f"User: {query}",
        date=today
    )

    print("\n[1/3] Prompt being sent to model:")
    print("-" * 40)
    print(prompt_content[:500] + "..." if len(prompt_content) > 500 else prompt_content)
    print("-" * 40)

    # Initialize model (same as write_research_brief)
    model_name = MINIMAL_CONFIG.get("research_model", "openai:gpt-4.1")
    # Parse provider:model format
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
    else:
        provider, model = "openai", model_name

    print(f"\n[2/3] Calling model: {model_name}")
    start_time = time.time()

    try:
        research_model = init_chat_model(
            model=model,
            model_provider=provider,
            max_tokens=MINIMAL_CONFIG.get("research_model_max_tokens", 4000),
        ).with_structured_output(ResearchQuestion)

        response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
        elapsed = time.time() - start_time

        print(f"   Done in {elapsed:.1f}s")

        # Display result
        brief = response.research_brief

        print(f"\n[3/3] Generated Brief ({len(brief)} chars):")
        print("=" * 60)
        print(brief)
        print("=" * 60)

        # Save output
        output_path = project_root / "test_brief_output.md"
        report = f"""# Brief Generation Test

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** {model_name}
**Time:** {elapsed:.1f}s

---

## Generated Brief

{brief}

---

## Prompt Used

```
{prompt_content}
```
"""
        output_path.write_text(report)
        print(f"\nOutput saved to: {output_path}")

        return brief

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Entry point."""
    # Check for OpenAI key
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set")
        sys.exit(1)

    # Get query from args or use default
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    result = asyncio.run(test_brief(query))

    if result:
        print("\nTEST PASSED")
    else:
        print("\nTEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
