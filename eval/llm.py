"""Simple LLM wrapper for eval - no pipeline dependencies."""

import json
import os
from pathlib import Path
from typing import Optional

# Load .env from project root
def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value

_load_env()


def call_llm(prompt: str, model: str = "gpt-4.1-mini") -> dict:
    """Call OpenAI and parse JSON response.

    Args:
        prompt: Full prompt text
        model: Model to use (default: gpt-4o-mini for cost)

    Returns:
        Parsed JSON dict from response
    """
    import openai

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    return json.loads(content)


def estimate_cost(prompt: str, model: str = "gpt-4.1-mini") -> float:
    """Estimate cost of a prompt (rough).

    gpt-4.1-mini: $0.40/1M input, $1.60/1M output (approx)
    gpt-4.1: $2.00/1M input, $8.00/1M output (approx)
    """
    # Rough token estimate: 4 chars per token
    input_tokens = len(prompt) / 4
    output_tokens = 1000  # Assume ~1K output

    if model == "gpt-4.1-mini":
        return (input_tokens * 0.40 + output_tokens * 1.60) / 1_000_000
    elif model == "gpt-4.1":
        return (input_tokens * 2.00 + output_tokens * 8.00) / 1_000_000
    else:
        return 0.01  # Unknown model, guess
