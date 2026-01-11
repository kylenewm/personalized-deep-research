"""
Sandbox for testing Pipeline v2 without re-running research.

Usage:
  1. First, capture state from a real run:
     python scripts/sandbox_pipeline.py --capture "your query here" --name my_query
     python scripts/sandbox_pipeline.py --capture "your query here" --name my_query --review  # pause to edit brief

  2. Then iterate on report generation:
     python scripts/sandbox_pipeline.py --run my_query                    # uses raw query (default)
     python scripts/sandbox_pipeline.py --run my_query --use-brief        # uses LLM brief
     python scripts/sandbox_pipeline.py --run my_query --topic "custom"   # uses your text
     python scripts/sandbox_pipeline.py --run all

  3. Fast filter tuning (no LLM calls):
     python scripts/sandbox_pipeline.py --save-extractions my_query       # run once, cache LLM outputs
     python scripts/sandbox_pipeline.py --replay my_query                 # instant replay with current filters
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_deep_research.pipeline_v2 import run_pipeline_v2, render_hybrid_report
from open_deep_research.configuration import Configuration
from open_deep_research.utils import get_api_key_for_model
from langchain.chat_models import init_chat_model

# Where to store gold query states
FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "gold_queries"
# Where to cache extraction results for fast replay
CACHE_DIR = Path(__file__).parent.parent / "sandbox_output" / "extraction_cache"


def prompt_for_edit(label: str, current_value: str) -> str:
    """Show current value and let user edit or accept."""
    print(f"\n{'='*60}")
    print(f"{label}:")
    print(f"{'='*60}")
    print(current_value)
    print(f"{'='*60}")
    print("\nOptions:")
    print("  [Enter] Accept as-is")
    print("  [e]     Edit (opens in $EDITOR)")
    print("  [t]     Type new value")
    print("  [q]     Quit")

    choice = input("\nChoice: ").strip().lower()

    if choice == "" or choice == "y":
        return current_value
    elif choice == "e":
        # Write to temp file, open in editor
        import tempfile
        import subprocess
        editor = os.environ.get("EDITOR", "vim")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(current_value)
            temp_path = f.name

        subprocess.call([editor, temp_path])

        with open(temp_path) as f:
            edited = f.read().strip()
        os.unlink(temp_path)

        print(f"\n[EDITED] New value ({len(edited)} chars)")
        return edited
    elif choice == "t":
        print("Enter new value (Ctrl+D or empty line to finish):")
        lines = []
        try:
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
        except EOFError:
            pass
        return "\n".join(lines)
    elif choice == "q":
        print("Aborted.")
        sys.exit(0)
    else:
        return current_value


async def run_research_with_brief(brief: str, config: dict) -> dict:
    """Run just the supervisor subgraph with a specific brief.

    This bypasses write_research_brief and validate_brief, directly
    injecting the brief into the supervisor.

    Args:
        brief: The research brief to use
        config: LangGraph config dict

    Returns:
        Supervisor output with notes, source_store, etc.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from open_deep_research.nodes.supervisor import supervisor_subgraph
    from open_deep_research.prompts import lead_researcher_prompt
    from open_deep_research.utils import get_today_str

    cfg = Configuration()

    # Build supervisor_messages exactly as brief.py does
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=cfg.get_effective_max_concurrent_research_units(),
        max_researcher_iterations=cfg.get_effective_max_researcher_iterations()
    )

    initial_state = {
        "supervisor_messages": [
            SystemMessage(content=supervisor_system_prompt),
            HumanMessage(content=brief)
        ],
        "research_brief": brief,
        "research_iterations": 0,
        "notes": [],
        "raw_notes": [],
        "source_store": []
    }

    print(f"[RESEARCH] Starting supervisor with brief: {brief[:80]}...")

    # Run supervisor subgraph directly
    result = await supervisor_subgraph.ainvoke(initial_state, config)

    return result


async def capture_state_from_run(query: str, name: str, review: bool = False, brief: str = None):
    """Run full pipeline and capture state after research for future testing.

    Args:
        query: The original user query
        name: Name for the fixture
        review: If True, generate brief via LLM and pause for interactive editing
        brief: If provided, use this brief directly (skips LLM generation and review)
    """
    from open_deep_research.graph import deep_researcher

    print(f"[CAPTURE] Query: {query[:80]}...")

    config = {"configurable": {"thread_id": f"capture_{name}"}}

    if brief:
        # Brief provided directly - skip LLM generation and review
        print(f"[CAPTURE] Using provided brief ({len(brief)} chars)...")
        print("[CAPTURE] Running research with your brief (this may take a while)...")

        result = await run_research_with_brief(brief, config)

        # Add metadata
        result["research_brief"] = brief
        result["user_edited_brief"] = True

    elif review:
        # Step 1: Generate brief via LLM
        from open_deep_research.prompts import transform_messages_into_research_topic_prompt
        from open_deep_research.state import ResearchQuestion
        from open_deep_research.utils import get_today_str
        from langchain_core.messages import HumanMessage

        print("[CAPTURE] Generating research brief...")
        cfg = Configuration()
        model = init_chat_model(
            model=cfg.research_model,
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=4000
        ).with_structured_output(ResearchQuestion)

        prompt = transform_messages_into_research_topic_prompt.format(
            messages=query,
            date=get_today_str()
        )
        response = await model.ainvoke([HumanMessage(content=prompt)])
        generated_brief = response.research_brief

        # Step 2: Let user review/edit
        final_brief = prompt_for_edit("LLM-GENERATED RESEARCH BRIEF", generated_brief)

        print(f"\n[CAPTURE] Using brief: {final_brief[:100]}...")
        print("[CAPTURE] Running research with your brief (this may take a while)...")

        # Step 3: Run supervisor subgraph directly with the edited brief
        # This bypasses write_research_brief and validate_brief
        result = await run_research_with_brief(final_brief, config)

        # Add metadata
        result["research_brief"] = final_brief
        result["user_edited_brief"] = True

    else:
        # Run full graph without pause
        print("[CAPTURE] Running full pipeline...")
        result = None
        async for event in deep_researcher.astream(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            stream_mode="values"
        ):
            result = event

        if result:
            result["user_edited_brief"] = False

    if not result:
        print("[CAPTURE] Failed - no result")
        return

    # Extract what we need
    state_to_save = {
        "query": query,
        "research_brief": result.get("research_brief", ""),
        "source_store": result.get("source_store", []),
        "notes": result.get("notes", []),
        "captured_at": datetime.now().isoformat(),
        "user_edited_brief": result.get("user_edited_brief", False)
    }

    # Save
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURES_DIR / f"{name}.json"

    with open(fixture_path, "w") as f:
        json.dump(state_to_save, f, indent=2, default=str)

    print(f"[CAPTURE] Saved to {fixture_path}")
    print(f"[CAPTURE] Sources: {len(state_to_save['source_store'])}")
    print(f"[CAPTURE] Brief: {state_to_save['research_brief'][:100]}...")


async def run_pipeline_only(name: str, use_brief: bool = False, custom_topic: str = None, output_dir: Path = None):
    """Run just Pipeline v2 from saved state."""
    fixture_path = FIXTURES_DIR / f"{name}.json"

    if not fixture_path.exists():
        print(f"[ERROR] No fixture found at {fixture_path}")
        print(f"[ERROR] Available: {list(FIXTURES_DIR.glob('*.json'))}")
        return None

    with open(fixture_path) as f:
        state = json.load(f)

    # Determine which topic to use
    if custom_topic:
        topic = custom_topic
        topic_source = "custom"
    elif use_brief:
        topic = state["research_brief"]
        topic_source = "LLM brief"
    else:
        topic = state["query"]
        topic_source = "raw query"

    print(f"[SANDBOX] Loading: {name}")
    print(f"[SANDBOX] Topic ({topic_source}): {topic[:80]}...")
    print(f"[SANDBOX] Sources: {len(state['source_store'])}")

    # Convert source_store to dict format for pipeline
    sources = {}
    for i, source in enumerate(state["source_store"]):
        src_id = f"src_{i:03d}"
        sources[src_id] = {
            "content": source.get("content", ""),
            "url": source.get("url", ""),
            "title": source.get("title", "")
        }

    # Set up LLM call wrapper
    config = Configuration()
    model = init_chat_model(
        model=config.research_model,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4000
    )

    async def llm_call(prompt: str) -> str:
        response = await model.ainvoke(prompt)
        return response.content

    # Run pipeline
    print(f"[SANDBOX] Running Pipeline v2...")
    start = datetime.now()

    report = await run_pipeline_v2(
        sources=sources,
        topic=topic,
        title=f"Research: {state['query'][:50]}",
        llm_call=llm_call,
        batch_size=10,
        min_score=0.3
    )

    elapsed = (datetime.now() - start).total_seconds()
    print(f"[SANDBOX] Done in {elapsed:.1f}s")

    # Render
    rendered = render_hybrid_report(report)

    # Save output
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "sandbox_output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{name}_{datetime.now().strftime('%H%M%S')}.md"
    with open(output_path, "w") as f:
        f.write(rendered)

    print(f"[SANDBOX] Output: {output_path}")
    print(f"[SANDBOX] Stats: {report.total_extracted} extracted → {report.total_verified} verified → {report.total_used} used")

    return report


def diagnose_quality_rejection(text: str) -> str:
    """Return reason why text fails quality filter."""
    import re
    if not text or len(text) < 50:
        return "too_short"
    if text.count('|') > 3:
        return "table_fragment"
    if 'Metadata' in text and ('License' in text or 'Provider' in text):
        return "metadata_block"

    text_lower = text.lower()
    nav_patterns = ['[skip to', '[read more]', '[contact us]', '[learn more]',
                    '[sign up]', '[log in]', '[home]', '[about]', 'log in[',
                    'sign up[', '✕dismiss', 'dismiss this', '[products]',
                    '[services]', '[pricing]']
    for pattern in nav_patterns:
        if pattern in text_lower:
            return f"nav_pattern:{pattern}"

    bracket_links = re.findall(r'\[[^\]]{1,20}\]', text)
    if len(bracket_links) >= 3:
        return f"multiple_brackets:{len(bracket_links)}"

    alpha_count = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_count / max(len(text), 1)
    if alpha_ratio < 0.5:
        return f"low_alpha:{alpha_ratio:.2f}"

    stripped = text.rstrip()
    if stripped.endswith('*') or stripped.endswith('...') or stripped.endswith(':'):
        return f"truncated_ending:{stripped[-3:]}"

    if stripped.endswith('|') or '---' in text:
        return "markdown_artifact"

    if text.count('#') > 2 or text.count('**') > 4:
        return "heavy_markdown"

    return "unknown"


async def run_diagnostics(name: str, use_brief: bool = False):
    """Run pipeline with detailed diagnostics at each stage."""
    from open_deep_research.pipeline_v2 import (
        batch_sources, BATCH_SIZE, MAX_CHARS_PER_SOURCE, DEFAULT_MIN_SCORE,
        deduplicate_extractions
    )
    from open_deep_research.pointer_extract import (
        POINTER_PROMPT, format_sources_for_prompt, parse_pointer_response,
        extract_from_pointer, is_quality_extraction, find_best_match
    )

    fixture_path = FIXTURES_DIR / f"{name}.json"
    if not fixture_path.exists():
        print(f"[ERROR] No fixture found at {fixture_path}")
        return

    with open(fixture_path) as f:
        state = json.load(f)

    topic = state["research_brief"] if use_brief else state["query"]

    # Convert source_store
    sources = {}
    for i, source in enumerate(state["source_store"]):
        src_id = f"src_{i:03d}"
        sources[src_id] = {
            "content": source.get("content", ""),
            "url": source.get("url", ""),
            "title": source.get("title", "")
        }

    print(f"{'='*70}")
    print(f"DIAGNOSTIC RUN: {name}")
    print(f"{'='*70}")
    print(f"Topic: {topic[:100]}...")
    print(f"Sources: {len(sources)}")

    # Set up LLM
    config = Configuration()
    model = init_chat_model(
        model=config.research_model,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4000
    )

    async def llm_call(prompt: str) -> str:
        response = await model.ainvoke(prompt)
        return response.content

    # Run ALL batches to get aggregate stats
    batches = batch_sources(sources, BATCH_SIZE)

    print(f"\n{'='*70}")
    print(f"STAGE 1: EXTRACTION (All {len(batches)} batches)")
    print(f"{'='*70}")

    total_stats = {
        "pointers": 0,
        "verified": 0,
        "not_found_no_source": 0,
        "not_found_low_score": 0,
        "not_found_quality_filter": 0,
    }
    all_quality_rejections = []
    rejection_reasons = {}

    for batch_idx, batch in enumerate(batches):
        # Format and send to LLM
        formatted = format_sources_for_prompt(batch, max_chars=MAX_CHARS_PER_SOURCE)
        prompt = POINTER_PROMPT.format(sources=formatted, topic=topic)

        response = await llm_call(prompt)
        pointers = parse_pointer_response(response)
        total_stats["pointers"] += len(pointers)

        batch_verified = 0
        batch_rejected = 0

        for pointer in pointers:
            source = batch.get(pointer.source_id)
            if not source:
                total_stats["not_found_no_source"] += 1
                continue

            content = source.get("content", "") or source.get("raw_content", "")
            if not content:
                total_stats["not_found_no_source"] += 1
                continue

            # Try to match
            extracted_text, score = find_best_match(
                pointer.keywords, content, min_score=DEFAULT_MIN_SCORE
            )

            if not extracted_text:
                total_stats["not_found_low_score"] += 1
                continue

            # Check quality
            if not is_quality_extraction(extracted_text):
                rejection_reason = diagnose_quality_rejection(extracted_text)
                total_stats["not_found_quality_filter"] += 1
                rejection_reasons[rejection_reason] = rejection_reasons.get(rejection_reason, 0) + 1
                if len(all_quality_rejections) < 10:  # Keep first 10
                    all_quality_rejections.append({
                        "source_id": pointer.source_id,
                        "text_preview": extracted_text[:200],
                        "reason": rejection_reason
                    })
                batch_rejected += 1
                continue

            total_stats["verified"] += 1
            batch_verified += 1

        print(f"  Batch {batch_idx+1}/{len(batches)}: {len(pointers)} pointers → {batch_verified} verified, {batch_rejected} quality-rejected")

    # Summary
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY (All batches)")
    print(f"{'='*70}")
    print(f"Sources: {len(sources)}")
    print(f"Pointers generated: {total_stats['pointers']}")
    print(f"  → Verified: {total_stats['verified']} ({100*total_stats['verified']/max(total_stats['pointers'],1):.1f}%)")
    print(f"  → No source/content: {total_stats['not_found_no_source']}")
    print(f"  → Low score: {total_stats['not_found_low_score']}")
    print(f"  → Quality filter: {total_stats['not_found_quality_filter']}")

    if rejection_reasons:
        print(f"\nQuality rejection breakdown:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if all_quality_rejections:
        print(f"\nSample quality rejections:")
        for rej in all_quality_rejections[:5]:
            print(f"  - {rej['source_id']} [{rej['reason']}]: '{rej['text_preview'][:60]}...'")

    print(f"\n[!] To see full pipeline, run without --diagnose")


async def run_all(use_brief: bool = False):
    """Run pipeline on all gold queries."""
    fixtures = list(FIXTURES_DIR.glob("*.json"))
    if not fixtures:
        print("[ERROR] No fixtures found. Run --capture first.")
        return

    print(f"[SANDBOX] Running {len(fixtures)} gold queries...")

    for fixture in fixtures:
        name = fixture.stem
        print(f"\n{'='*60}")
        await run_pipeline_only(name, use_brief=use_brief)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline v2 sandbox")

    # Capture options
    parser.add_argument("--capture", type=str, help="Run full pipeline and capture state")
    parser.add_argument("--name", type=str, help="Name for the captured state")
    parser.add_argument("--review", action="store_true", help="Pause to review/edit brief before research")
    parser.add_argument("--brief", type=str, help="Use this brief directly (skips LLM generation)")
    parser.add_argument("--brief-file", type=str, help="Read brief from file (skips LLM generation)")

    # Run options
    parser.add_argument("--run", type=str, help="Run pipeline on saved state (name or 'all')")
    parser.add_argument("--use-brief", action="store_true", help="Use LLM-generated brief instead of raw query")
    parser.add_argument("--topic", type=str, help="Use custom topic text")
    parser.add_argument("--diagnose", type=str, help="Run diagnostics on fixture (shows extraction breakdown)")

    # Utility
    parser.add_argument("--list", action="store_true", help="List available fixtures")
    parser.add_argument("--show", type=str, help="Show details of a fixture")

    args = parser.parse_args()

    if args.list:
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        fixtures = list(FIXTURES_DIR.glob("*.json"))
        if fixtures:
            print("Available fixtures:")
            for f in fixtures:
                with open(f) as file:
                    data = json.load(file)
                edited = " [edited]" if data.get("user_edited_brief") else ""
                print(f"  - {f.stem}: {data['query'][:60]}... ({len(data['source_store'])} sources){edited}")
        else:
            print("No fixtures yet. Run --capture first.")
        return

    if args.show:
        fixture_path = FIXTURES_DIR / f"{args.show}.json"
        if not fixture_path.exists():
            print(f"[ERROR] No fixture: {args.show}")
            return
        with open(fixture_path) as f:
            data = json.load(f)
        print(f"Query: {data['query']}")
        print(f"\nResearch Brief:\n{data['research_brief']}")
        print(f"\nSources: {len(data['source_store'])}")
        print(f"Captured: {data.get('captured_at', 'unknown')}")
        print(f"User edited: {data.get('user_edited_brief', False)}")
        return

    if args.capture:
        name = args.name or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Handle --brief or --brief-file
        brief = None
        if args.brief:
            brief = args.brief
        elif args.brief_file:
            with open(args.brief_file) as f:
                brief = f.read().strip()
        asyncio.run(capture_state_from_run(args.capture, name, review=args.review, brief=brief))
    elif args.diagnose:
        asyncio.run(run_diagnostics(args.diagnose, use_brief=args.use_brief))
    elif args.run:
        if args.run == "all":
            asyncio.run(run_all(use_brief=args.use_brief))
        else:
            asyncio.run(run_pipeline_only(args.run, use_brief=args.use_brief, custom_topic=args.topic))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
