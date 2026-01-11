#!/usr/bin/env python3
"""Brief generation sandbox - analyze research brief quality.

This sandbox tests:
1. Query → Brief transformation quality
2. Context injection effectiveness
3. Brief specificity and coverage
4. Council feedback patterns

Usage:
    python scripts/brief_sandbox.py --generate "query"     # Generate brief
    python scripts/brief_sandbox.py --analyze briefs.json  # Analyze saved briefs
    python scripts/brief_sandbox.py --batch                # Test batch of queries
    python scripts/brief_sandbox.py --stats                # Show all stats
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

BRIEFS_DIR = project_root / "tests/fixtures/briefs"


def ensure_briefs_dir():
    """Create briefs directory if needed."""
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)


# Test queries with expected quality attributes
TEST_QUERIES = [
    {
        "query": "What are the best voice AI agents in 2025?",
        "expected": {
            "should_contain": ["voice", "AI", "2025"],
            "type": "product_comparison",
            "specificity": "medium"
        }
    },
    {
        "query": "Compare Retell AI vs Vapi for building voice agents",
        "expected": {
            "should_contain": ["Retell", "Vapi", "voice", "agent"],
            "type": "product_comparison",
            "specificity": "high"
        }
    },
    {
        "query": "How does async work in Python?",
        "expected": {
            "should_contain": ["async", "Python"],
            "type": "technical",
            "specificity": "low"
        }
    },
    {
        "query": "Recent developments in quantum computing",
        "expected": {
            "should_contain": ["quantum", "computing"],
            "type": "news",
            "specificity": "low"
        }
    },
    {
        "query": "I want to build a voice assistant that can handle customer support calls for my e-commerce store. It should be able to answer questions about orders, returns, and product availability. Budget is around $500/month.",
        "expected": {
            "should_contain": ["voice", "customer support", "e-commerce", "orders", "returns", "$500"],
            "type": "implementation",
            "specificity": "high"
        }
    },
]


async def generate_brief(query: str, use_context: bool = True) -> dict:
    """Generate a research brief from a query.

    Returns:
        Dict with query, brief, context_used, metrics
    """
    from langchain_core.messages import HumanMessage
    from open_deep_research.models import configurable_model
    from open_deep_research.state import ResearchQuestion
    from open_deep_research.prompts import transform_messages_into_research_topic_prompt
    from open_deep_research.utils import gather_brief_context, format_brief_context, get_today_str

    print(f"Generating brief for: {query[:60]}...")

    # Set up model
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "No OPENAI_API_KEY"}

    model_config = {
        "model": "gpt-4.1-mini",
        "max_tokens": 2000,
        "api_key": api_key,
    }

    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_config(model_config)
    )

    # Build prompt
    prompt = transform_messages_into_research_topic_prompt.format(
        messages=f"Human: {query}",
        date=get_today_str()
    )

    # Add context if enabled
    context_info = None
    if use_context:
        try:
            from langgraph.config import RunnableConfig
            config = RunnableConfig(configurable={})

            context = await gather_brief_context(
                user_messages=query,
                config=config,
                max_queries=2,
                max_results=3,
                days=30,
                include_news=True
            )

            if context.sources_used:
                context_block = format_brief_context(context, days=30)
                prompt += f"\n\n{context_block}"
                context_info = {
                    "sources_used": len(context.sources_used),
                    "entities": context.key_entities[:5],
                    "dates": context.key_dates[:3]
                }
        except Exception as e:
            print(f"Context gathering failed: {e}")

    # Generate brief
    try:
        response = await research_model.ainvoke([HumanMessage(content=prompt)])
        brief = response.research_brief

        return {
            "query": query,
            "brief": brief,
            "brief_length": len(brief),
            "context_used": context_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "query": query}


def analyze_brief(brief_data: dict) -> dict:
    """Analyze a generated brief for quality metrics.

    Returns:
        Dict with:
        - specificity_score: 0-1 based on named entities/numbers
        - coverage_score: 0-1 based on expected keywords
        - length_appropriate: bool
        - issues: list of potential problems
    """
    brief = brief_data.get("brief", "")
    query = brief_data.get("query", "")

    if not brief:
        return {"error": "No brief to analyze"}

    import re

    # Calculate specificity (presence of specific details)
    numbers = len(re.findall(r'\b\d+\b', brief))
    dates = len(re.findall(r'\b(20\d{2}|January|February|March|April|May|June|July|August|September|October|November|December)\b', brief, re.I))
    proper_nouns = len(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', brief))
    specific_markers = numbers + dates + proper_nouns

    specificity = min(1.0, specific_markers / 20)  # Cap at 1.0

    # Check query keyword coverage
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    brief_words = set(re.findall(r'\b\w+\b', brief.lower()))
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'and', 'or',
                 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'what',
                 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'to', 'for',
                 'of', 'in', 'on', 'at', 'by', 'from', 'with', 'about', 'as', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'between',
                 'under', 'again', 'further', 'than', 'once', 'i', 'me', 'my', 'want'}
    query_keywords = query_words - stopwords
    covered = len(query_keywords & brief_words)
    coverage = covered / len(query_keywords) if query_keywords else 1.0

    # Check length appropriateness
    length = len(brief)
    length_appropriate = 100 <= length <= 2000

    # Identify issues
    issues = []
    if length < 100:
        issues.append("Too short - may lack specificity")
    if length > 2000:
        issues.append("Too long - may be unfocused")
    if specificity < 0.3:
        issues.append("Low specificity - missing concrete details")
    if coverage < 0.5:
        issues.append(f"Poor coverage - missing query keywords")

    # Check for bad patterns
    if "I will" in brief or "I'll" in brief:
        issues.append("Contains future tense - should be research questions")
    if brief.count("?") < 1:
        issues.append("No question marks - may not be framed as research questions")

    return {
        "query": query[:60],
        "brief_length": length,
        "specificity_score": round(specificity, 2),
        "coverage_score": round(coverage, 2),
        "length_appropriate": length_appropriate,
        "numbers_found": numbers,
        "dates_found": dates,
        "proper_nouns": proper_nouns,
        "issues": issues,
        "quality_score": round((specificity + coverage + (1 if length_appropriate else 0)) / 3, 2)
    }


def print_analysis(analysis: dict):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print(f"BRIEF ANALYSIS")
    print(f"{'='*60}")
    print(f"Query: {analysis.get('query', 'unknown')}...")

    print(f"\n  Length: {analysis['brief_length']} chars {'✅' if analysis['length_appropriate'] else '⚠️'}")
    print(f"  Specificity: {analysis['specificity_score']:.0%}")
    print(f"  Coverage: {analysis['coverage_score']:.0%}")
    print(f"  Quality Score: {analysis['quality_score']:.0%}")

    print(f"\n  Details:")
    print(f"    Numbers: {analysis['numbers_found']}")
    print(f"    Dates: {analysis['dates_found']}")
    print(f"    Proper Nouns: {analysis['proper_nouns']}")

    if analysis['issues']:
        print(f"\n  Issues:")
        for issue in analysis['issues']:
            print(f"    ⚠️  {issue}")
    else:
        print(f"\n  ✅ No issues detected")


async def run_batch_test():
    """Run batch test on predefined queries."""
    print(f"\n{'='*60}")
    print(f"BATCH BRIEF GENERATION TEST")
    print(f"{'='*60}")

    results = []
    for test in TEST_QUERIES:
        query = test["query"]
        expected = test["expected"]

        print(f"\n  Testing: {query[:50]}...")

        brief_data = await generate_brief(query, use_context=False)

        if "error" in brief_data:
            print(f"    ❌ Error: {brief_data['error']}")
            continue

        analysis = analyze_brief(brief_data)

        # Check expected keywords
        brief_lower = brief_data["brief"].lower()
        missing = [k for k in expected["should_contain"] if k.lower() not in brief_lower]

        result = {
            "query": query[:50],
            "brief_length": len(brief_data["brief"]),
            "quality_score": analysis["quality_score"],
            "missing_keywords": missing,
            "issues": analysis["issues"]
        }
        results.append(result)

        status = "✅" if not missing and analysis["quality_score"] >= 0.5 else "⚠️"
        print(f"    {status} Quality: {analysis['quality_score']:.0%}, Missing: {missing if missing else 'none'}")

    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")

    passing = sum(1 for r in results if r["quality_score"] >= 0.5 and not r["missing_keywords"])
    print(f"  Passed: {passing}/{len(results)}")
    avg_quality = sum(r["quality_score"] for r in results) / len(results) if results else 0
    print(f"  Avg Quality: {avg_quality:.0%}")

    # Save results
    ensure_briefs_dir()
    results_path = BRIEFS_DIR / f"batch_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


def show_all_stats():
    """Show aggregated stats from all saved briefs."""
    briefs = list(BRIEFS_DIR.glob("*.json"))

    if not briefs:
        print("No saved briefs found.")
        print(f"Run: python scripts/brief_sandbox.py --generate 'your query'")
        return

    print(f"\n{'='*60}")
    print(f"AGGREGATED BRIEF STATS ({len(briefs)} files)")
    print(f"{'='*60}")

    all_quality = []
    all_lengths = []
    all_issues = defaultdict(int)

    for brief_path in briefs:
        with open(brief_path) as f:
            data = json.load(f)

        # Handle both single briefs and batch results
        if isinstance(data, list):
            for item in data:
                if "quality_score" in item:
                    all_quality.append(item["quality_score"])
                if "brief_length" in item:
                    all_lengths.append(item["brief_length"])
                for issue in item.get("issues", []):
                    all_issues[issue] += 1
        else:
            if "brief" in data:
                analysis = analyze_brief(data)
                all_quality.append(analysis["quality_score"])
                all_lengths.append(analysis["brief_length"])
                for issue in analysis.get("issues", []):
                    all_issues[issue] += 1

    if all_quality:
        print(f"\n  Metrics:")
        print(f"    Avg quality score: {sum(all_quality)/len(all_quality):.0%}")
        print(f"    Avg brief length: {sum(all_lengths)//len(all_lengths)} chars")
        print(f"    Total briefs analyzed: {len(all_quality)}")

    if all_issues:
        print(f"\n  Common Issues:")
        for issue, count in sorted(all_issues.items(), key=lambda x: -x[1])[:5]:
            print(f"    {count}x {issue}")


async def main():
    args = sys.argv[1:]

    ensure_briefs_dir()

    if "--generate" in args:
        idx = args.index("--generate")
        if idx + 1 < len(args):
            query = args[idx + 1]
            use_context = "--no-context" not in args

            brief_data = await generate_brief(query, use_context=use_context)

            if "error" in brief_data:
                print(f"Error: {brief_data['error']}")
                return

            # Save brief
            brief_path = BRIEFS_DIR / f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(brief_path, 'w') as f:
                json.dump(brief_data, f, indent=2)
            print(f"\nBrief saved to: {brief_path}")

            # Show brief
            print(f"\n{'='*60}")
            print("GENERATED BRIEF")
            print(f"{'='*60}")
            print(brief_data["brief"])

            if brief_data.get("context_used"):
                ctx = brief_data["context_used"]
                print(f"\nContext injected:")
                print(f"  Sources: {ctx['sources_used']}")
                print(f"  Entities: {ctx['entities']}")

            # Analyze
            analysis = analyze_brief(brief_data)
            print_analysis(analysis)
        else:
            print("Usage: --generate 'query string' [--no-context]")

    elif "--analyze" in args:
        idx = args.index("--analyze")
        if idx + 1 < len(args):
            brief_path = Path(args[idx + 1])
            if brief_path.exists():
                with open(brief_path) as f:
                    brief_data = json.load(f)
                analysis = analyze_brief(brief_data)
                print_analysis(analysis)
            else:
                print(f"Brief not found: {brief_path}")
        else:
            print("Usage: --analyze brief.json")

    elif "--batch" in args:
        await run_batch_test()

    elif "--stats" in args:
        show_all_stats()

    else:
        print(__doc__)
        print("\nAvailable commands:")
        print("  --generate 'query'   Generate and analyze a brief")
        print("  --generate 'query' --no-context   Skip context injection")
        print("  --analyze file.json  Analyze a saved brief")
        print("  --batch              Run batch test on predefined queries")
        print("  --stats              Show aggregated stats")


if __name__ == "__main__":
    asyncio.run(main())
