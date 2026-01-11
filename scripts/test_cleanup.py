#!/usr/bin/env python3
"""Test the LLM cleanup on actual garbage from report_preview.html"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import AsyncOpenAI

# Sample garbage extractions from the actual report
GARBAGE_SAMPLES = [
    # Line 58 - nav at start
    'Introducing next-generation audio models in the API | OpenAI [Skip to main content](https://openai.com/index/introducing-our-next-generation-audio-models/#main) Log in [](https://openai.com/) Switch to * [ChatGPT(opens in a new window)](https://chatgpt.com/?',

    # Line 72 - pure nav garbage
    'Changelog | OpenAI API [](https://platform.openai.com/docs/overview) [Docs Docs](https://platform.openai.com/docs)[API reference API](https://platform.openai.com/docs/api-reference/introduction) Log in[Sign up](https://platform.openai.com/signup) Search K Get started [Overview](https://platform.',

    # Line 79 - header + image
    '# ElevenLabs Documentation ![](https://files.buildwithfern.com/https://elevenlabs.docs.buildwithfern.com/docs/12097a437e55f60c199946cf59c9528eb8349d110142394833d67fe93b50e68d/assets/images/overview/voice-library-bg.',

    # Line 86 - nav + unrelated questions
    '[The Keyword](/) Improving Gemini Text-to-Speech models for better control and capabilities ["How does Gemini work in Google Maps?", "What is quantum computing?", "What are the camera features on Pixel 10?',

    # Line 166 - read more + dates
    '[Read more](/release-notes/23-new-telnyx-ai-assistant-third-party-integrations) * **17, Dec 2025** ### Telnyx speech-to-text (STT) is now available as a standalone, real-time API We\'ve expanded our STT capabilities to make the Telnyx Speech-to-Text API available as a standalone service.',

    # Line 337 - pure nav links (worst case)
    '[Contact us](https://telnyx.com/contact-us)[Log in](https://portal.telnyx.com) [Contact us](https://telnyx.com/contact-us)[Log in](https://portal.telnyx.',

    # Clean sample - should remain unchanged
    'Using Gemini-TTS, you can synthesize single or multi-speaker speech from short snippets to long-form narratives, precisely dictating style, accent, pace, tone, and even emotional expression, all steerable through natural-language prompts.',
]

# Import our cleanup functions
import importlib.util
src_dir = Path(__file__).parent.parent / "src" / "open_deep_research"

def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

pointer_extract = load_mod("pointer_extract", src_dir / "pointer_extract.py")


CLEANUP_PROMPT = '''Review these extracted facts and identify any garbage text that should be removed.

Garbage includes:
- Navigation elements: [Skip to...], Log in, Sign up, menu links
- UI artifacts: Search K, keyboard shortcuts, dismiss buttons
- Image markdown: ![...](...)
- Header/footer boilerplate: site titles, page headers
- Incomplete fragments that don't convey meaning
- Markdown link syntax that's just navigation: [Text](url) patterns for nav
- Date/changelog formatting artifacts: * **Date** ###

For each fact, output the EXACT substring(s) to remove. If the fact is clean, output empty array.
If the entire fact is garbage with no salvageable content, output ["ENTIRE_FACT_IS_GARBAGE"].

Facts to review:
{facts}

Output JSON array:
[
  {{"fact_index": 0, "remove": ["[Skip to main content]", "Log in[Sign up]"]}},
  {{"fact_index": 1, "remove": []}},
  ...
]

Output ONLY the JSON array.'''


async def main():
    client = AsyncOpenAI()

    print("=" * 70)
    print("CLEANUP TEST: LLM Points, Code Removes")
    print("=" * 70)

    # Format facts for prompt
    facts_text = "\n\n".join([f"[{i}] {text[:500]}" for i, text in enumerate(GARBAGE_SAMPLES)])
    prompt = CLEANUP_PROMPT.format(facts=facts_text)

    print("\nCalling LLM to identify garbage...")
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.1
    )
    response = resp.choices[0].message.content
    print(f"\nLLM Response:\n{response}\n")

    # Parse response
    import json
    import re
    match = re.search(r'\[[\s\S]*\]', response)
    if match:
        cleanup_instructions = json.loads(match.group())
    else:
        print("Failed to parse response")
        return

    # Apply cleanup
    print("=" * 70)
    print("BEFORE/AFTER COMPARISON")
    print("=" * 70)

    for i, original in enumerate(GARBAGE_SAMPLES):
        instruction = next((x for x in cleanup_instructions if x.get("fact_index") == i), None)
        removals = instruction.get("remove", []) if instruction else []

        print(f"\n--- Sample {i} ---")
        print(f"BEFORE ({len(original)} chars):")
        print(f"  {original[:100]}...")

        if "ENTIRE_FACT_IS_GARBAGE" in removals:
            print(f"\nREMOVALS: [ENTIRE FACT REJECTED]")
            print(f"AFTER: <removed entirely>")
        elif removals:
            cleaned = original
            for removal in removals:
                if removal and len(removal) > 2:
                    cleaned = cleaned.replace(removal, '')
            # Normalize whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            print(f"\nREMOVALS: {removals}")
            print(f"AFTER ({len(cleaned)} chars):")
            print(f"  {cleaned[:100]}..." if len(cleaned) > 100 else f"  {cleaned}")
        else:
            print(f"\nREMOVALS: [] (clean)")
            print(f"AFTER: <unchanged>")


if __name__ == "__main__":
    asyncio.run(main())
