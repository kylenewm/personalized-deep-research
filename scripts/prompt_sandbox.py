#!/usr/bin/env python3
"""Prompt iteration sandbox - autonomous testing without API search costs.

Runs extraction/quality filter on cached fixtures, evaluates quality metrics,
and logs results for rapid prompt iteration.

Usage:
    python scripts/prompt_sandbox.py                    # Single run, latest fixture
    python scripts/prompt_sandbox.py voice_agent_eval   # Use specific fixture
    python scripts/prompt_sandbox.py --loop             # Continuous iteration mode
"""

import asyncio
import json
import re
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from openai import AsyncOpenAI


# =============================================================================
# QUALITY METRICS
# =============================================================================

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def has_formatting_artifacts(text: str) -> bool:
    """Check for markdown/formatting artifacts."""
    patterns = [
        r'\*\*',           # Bold markers
        r'####',           # Headers
        r'\]\(#',          # Link fragments
        r'^\s*[-*•]\s',    # Bullet points at start
        r'^\d+\.\s',       # Numbered lists at start
        r'!Image',         # Image markers
        r'\[.*?\]\(.*?\)', # Markdown links
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def is_question(text: str) -> bool:
    """Check if text is a question."""
    text = text.strip()
    if text.endswith('?'):
        return True
    question_starts = ['how ', 'what ', 'why ', 'when ', 'where ', 'who ', 'which ', 'is ', 'are ', 'does ', 'do ', 'can ', 'could ', 'would ', 'should ']
    lower = text.lower()
    return any(lower.startswith(q) and '?' in text for q in question_starts)


def is_header_or_title(text: str) -> bool:
    """Check if text looks like a header/title rather than a fact."""
    text = text.strip()
    # Ends with colon
    if text.endswith(':'):
        return True
    # Very short and capitalized
    words = text.split()
    if len(words) <= 5 and text == text.title():
        return True
    # Header patterns
    header_patterns = ['Key Features', 'Key Strengths', 'What to Expect', 'Overview', 'Summary', 'Conclusion', 'Introduction']
    for pattern in header_patterns:
        if pattern.lower() in text.lower():
            return True
    return False


def simple_semantic_hash(text: str) -> str:
    """Create a simple semantic hash for duplicate detection."""
    # Normalize: lowercase, remove punctuation, sort words
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    # Take first 10 significant words (skip common words)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'any', 'this', 'that', 'these', 'those', 'it', 'its'}
    significant = [w for w in words if w not in stopwords][:10]
    return ' '.join(sorted(significant))


def evaluate_facts(facts: list[str]) -> tuple[dict, bool]:
    """Evaluate extracted facts against quality thresholds.

    Returns:
        (scores_dict, passed_bool)
    """
    if not facts:
        return {"error": "No facts extracted"}, False

    word_counts = [count_words(f) for f in facts]

    scores = {
        "total_facts": len(facts),
        "avg_words": sum(word_counts) / len(word_counts),
        "max_words": max(word_counts),
        "over_50_words": sum(1 for wc in word_counts if wc > 50),
        "over_40_words": sum(1 for wc in word_counts if wc > 40),
        "has_artifacts": sum(1 for f in facts if has_formatting_artifacts(f)),
        "is_question": sum(1 for f in facts if is_question(f)),
        "is_header": sum(1 for f in facts if is_header_or_title(f)),
    }

    # Detect duplicates via semantic hash
    hashes = [simple_semantic_hash(f) for f in facts]
    hash_counts = Counter(hashes)
    duplicate_count = sum(c - 1 for c in hash_counts.values() if c > 1)
    scores["duplicates"] = duplicate_count
    scores["duplicate_rate"] = duplicate_count / len(facts) if facts else 0

    # Quality thresholds
    passed = (
        scores["avg_words"] < 40 and
        scores["over_50_words"] == 0 and
        scores["has_artifacts"] == 0 and
        scores["is_question"] == 0 and
        scores["is_header"] == 0 and
        scores["duplicate_rate"] < 0.10
    )

    return scores, passed


def print_scores(scores: dict, passed: bool):
    """Pretty print evaluation scores."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{'='*60}")
    print(f"EVALUATION: {status}")
    print(f"{'='*60}")
    print(f"  Total facts:      {scores.get('total_facts', 0)}")
    print(f"  Avg words:        {scores.get('avg_words', 0):.1f} (target: <40)")
    print(f"  Max words:        {scores.get('max_words', 0)}")
    print(f"  Over 50 words:    {scores.get('over_50_words', 0)} (target: 0)")
    print(f"  Over 40 words:    {scores.get('over_40_words', 0)}")
    print(f"  Artifacts:        {scores.get('has_artifacts', 0)} (target: 0)")
    print(f"  Questions:        {scores.get('is_question', 0)} (target: 0)")
    print(f"  Headers:          {scores.get('is_header', 0)} (target: 0)")
    print(f"  Duplicates:       {scores.get('duplicates', 0)} ({scores.get('duplicate_rate', 0):.1%})")
    print(f"{'='*60}\n")


# =============================================================================
# EXTRACTION (Minimal, for sandbox testing)
# =============================================================================

async def run_extraction_sandbox(sources: dict, topic: str, llm_call) -> list[str]:
    """Run extraction and return list of fact texts."""
    from open_deep_research.pipeline_v2 import extract_all_batched

    extractions = await extract_all_batched(
        sources=sources,
        topic=topic,
        llm_call=llm_call,
        batch_size=12,
        min_score=0.4,
        on_batch_complete=lambda i, t, e: print(f"  Batch {i}/{t}: {len([x for x in e if x.status == 'verified'])} verified")
    )

    verified = [e for e in extractions if e.status == "verified"]
    facts = [e.extracted_text for e in verified if e.extracted_text]
    return facts


# =============================================================================
# MAIN SANDBOX
# =============================================================================

async def run_sandbox(fixture_name: str = None, loop_mode: bool = False):
    """Run the prompt sandbox."""

    # Load fixture
    fixture_dir = project_root / "tests/fixtures/gold_queries"
    if fixture_name:
        fixture_path = fixture_dir / f"{fixture_name}.json"
    else:
        fixtures = sorted(fixture_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not fixtures:
            print("No fixtures found. Run a test first to create one.")
            return
        fixture_path = fixtures[0]

    print(f"Loading: {fixture_path.name}")
    with open(fixture_path) as f:
        fixture = json.load(f)

    query = fixture.get("query", "Research topic")
    sources_list = fixture.get("sources", [])

    # Use only 3 sources for fast iteration
    sources_list = sources_list[:3]

    # Convert to dict format
    sources = {}
    for i, src in enumerate(sources_list):
        content = src.get("content", "") or src.get("raw_content", "")
        if content:
            sources[f"src_{i:03d}"] = {
                "content": content,
                "url": src.get("url", ""),
                "title": src.get("title", "Unknown"),
            }

    print(f"Query: {query[:80]}...")
    print(f"Sources: {len(sources)} (sandbox mode)")
    print("=" * 60)

    # Setup LLM
    client = AsyncOpenAI()

    async def llm_call(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3
        )
        return resp.choices[0].message.content

    # Run extraction
    print("\nRunning extraction...")
    facts = await run_extraction_sandbox(sources, query, llm_call)

    print(f"\nExtracted {len(facts)} facts")

    # Show sample facts
    print("\n--- SAMPLE FACTS ---")
    for i, fact in enumerate(facts[:10]):
        word_count = count_words(fact)
        artifacts = "⚠️" if has_formatting_artifacts(fact) else ""
        question = "❓" if is_question(fact) else ""
        header = "📋" if is_header_or_title(fact) else ""
        print(f"[{i+1}] ({word_count}w) {artifacts}{question}{header} {fact[:150]}...")

    if len(facts) > 10:
        print(f"... and {len(facts) - 10} more")

    # Evaluate
    scores, passed = evaluate_facts(facts)
    print_scores(scores, passed)

    # Log results
    log_path = project_root / "sandbox_iterations.jsonl"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "fixture": fixture_path.name,
        "sources": len(sources),
        "scores": scores,
        "passed": passed,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Logged to: {log_path}")

    # Show problem facts
    if not passed:
        print("\n--- PROBLEM FACTS ---")
        for i, fact in enumerate(facts):
            problems = []
            wc = count_words(fact)
            if wc > 50:
                problems.append(f"too long ({wc}w)")
            if has_formatting_artifacts(fact):
                problems.append("artifacts")
            if is_question(fact):
                problems.append("question")
            if is_header_or_title(fact):
                problems.append("header")

            if problems:
                print(f"[{i+1}] {', '.join(problems)}")
                print(f"    {fact[:200]}...")
                print()

    return passed


def main():
    args = sys.argv[1:]
    loop_mode = "--loop" in args
    args = [a for a in args if not a.startswith("-")]
    fixture_name = args[0] if args else None

    passed = asyncio.run(run_sandbox(fixture_name, loop_mode))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
