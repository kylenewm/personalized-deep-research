#!/usr/bin/env python3
"""Test cleanup v2: LLM outputs clean text, code verifies it's a contiguous substring."""

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

CLEANUP_PROMPT_V2 = '''For each text, output ONLY the meaningful content with navigation/UI garbage removed.

Rules:
- Remove navigation links: [Skip to...], [Read more], [Contact us], etc.
- Remove UI artifacts: Log in, Sign up, Search K, menu items
- Remove image markdown: ![](...)
- Remove header artifacts: # Title, [Site Name](/)
- Remove formatting artifacts: * **Date** ###
- Keep the actual informative content
- If there's no meaningful content, output "NO_CONTENT"
- Output must be an EXACT substring of the original (don't rephrase!)

Texts:
{texts}

Output JSON array with cleaned versions:
[
  {{"index": 0, "cleaned": "the meaningful content here"}},
  {{"index": 1, "cleaned": "NO_CONTENT"}},
  ...
]

Output ONLY the JSON array.'''


async def main():
    client = AsyncOpenAI()

    print("=" * 70)
    print("CLEANUP V2: LLM outputs clean text, code verifies substring")
    print("=" * 70)

    # Format texts for prompt
    texts = "\n\n".join([f"[{i}] {text}" for i, text in enumerate(GARBAGE_SAMPLES)])
    prompt = CLEANUP_PROMPT_V2.format(texts=texts)

    print("\nCalling LLM...")
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.1
    )
    response = resp.choices[0].message.content
    print(f"\nLLM Response:\n{response}\n")

    # Parse response
    import json
    import re
    match = re.search(r'\[[\s\S]*\]', response)
    if not match:
        print("Failed to parse response")
        return

    results = json.loads(match.group())

    # Verify and apply
    print("=" * 70)
    print("VERIFICATION: Is cleaned text a substring of original?")
    print("=" * 70)

    for result in results:
        idx = result.get("index", -1)
        cleaned = result.get("cleaned", "")

        if idx < 0 or idx >= len(GARBAGE_SAMPLES):
            continue

        original = GARBAGE_SAMPLES[idx]

        print(f"\n--- Sample {idx} ---")
        print(f"ORIGINAL ({len(original)} chars):")
        print(f"  {original[:80]}...")

        if cleaned == "NO_CONTENT":
            print(f"\nLLM: NO_CONTENT (pure garbage)")
            print(f"RESULT: ❌ Reject entire extraction")
        elif cleaned in original:
            print(f"\nLLM OUTPUT ({len(cleaned)} chars):")
            print(f"  {cleaned[:80]}..." if len(cleaned) > 80 else f"  {cleaned}")
            print(f"\nVERIFY: '{cleaned[:30]}...' in original? ✅ YES")
            print(f"RESULT: ✅ Use cleaned version")
        else:
            print(f"\nLLM OUTPUT ({len(cleaned)} chars):")
            print(f"  {cleaned[:80]}..." if len(cleaned) > 80 else f"  {cleaned}")
            print(f"\nVERIFY: substring in original? ❌ NO (LLM modified text)")
            print(f"RESULT: ⚠️ Keep original (safety fallback)")


if __name__ == "__main__":
    asyncio.run(main())
