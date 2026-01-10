"""Safeguarded report generation node (Pipeline v2).

This node replaces the legacy extract_evidence → verify_evidence → claim_pre_check
→ final_report chain with a three-stage safeguarded generation pipeline:

1. Batched pointer extraction: LLM points to facts, code extracts them
2. Arranger: Groups facts by theme, curates (~50%+ kept)
3. Per-theme synthesis: LLM writes transitions around verified facts

Key principle: LLM never writes factual content, only points to it.
This structurally prevents hallucination.
"""

import asyncio
import logging
from typing import Dict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from openai import AsyncOpenAI

from open_deep_research.configuration import Configuration
from open_deep_research.state import AgentState
from open_deep_research.pipeline_v2 import (
    run_pipeline_v2,
    render_hybrid_report,
    BATCH_SIZE,
    DEFAULT_MIN_SCORE,
)


async def safeguarded_report_generation(state: AgentState, config: RunnableConfig) -> dict:
    """Generate report using safeguarded three-stage pipeline.

    This replaces the legacy chain:
    - extract_evidence (S03)
    - verify_evidence (S04)
    - claim_pre_check (Layer 3)
    - final_report_generation

    With a single node that:
    1. Extracts verified facts via pointer extraction
    2. Groups facts by theme
    3. Synthesizes report with clear verified/synthesis distinction

    Args:
        state: Current agent state with source_store and research_brief
        config: Runtime configuration

    Returns:
        Dictionary with final_report to add to state
    """
    configurable = Configuration.from_runnable_config(config)

    # Get sources from state
    sources = state.get("source_store", [])

    if not sources:
        logging.warning("[SAFEGUARDED] No sources available for report generation.")
        print("[SAFEGUARDED] ⚠️ No sources found")
        return {"final_report": "Error: No sources available for report generation."}

    # Build sources dict for pipeline
    sources_dict: Dict[str, dict] = {}
    for i, src in enumerate(sources):
        content = src.get("content", "") or src.get("raw_content", "")
        if content:
            sources_dict[f"src_{i:03d}"] = {
                "content": content,
                "url": src.get("url", ""),
                "title": src.get("title", "Unknown"),
            }

    if not sources_dict:
        print("[SAFEGUARDED] ⚠️ No sources with content")
        return {"final_report": "Error: No sources with extractable content."}

    print(f"[SAFEGUARDED] Processing {len(sources_dict)} sources...")

    # Get original user query from messages (not the research brief!)
    # The brief is a PLAN, but the prompts need the actual QUESTION
    messages = state.get("messages", [])
    topic = None
    for msg in messages:
        # Find the first human message - that's the original query
        if isinstance(msg, HumanMessage):
            topic = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            topic = msg.get("content", "")
            break

    # Fallback to brief if no user message found (shouldn't happen)
    if not topic:
        research_brief = state.get("research_brief", "")
        topic = research_brief[:500] if research_brief else "Research findings"
        print(f"[SAFEGUARDED] ⚠️ No user query found, using brief as topic")

    # Get config parameters
    batch_size = getattr(configurable, 'safeguarded_batch_size', BATCH_SIZE)
    min_score = getattr(configurable, 'safeguarded_min_score', DEFAULT_MIN_SCORE)

    # Setup LLM client
    client = AsyncOpenAI()

    async def llm_call(prompt: str) -> str:
        """Call LLM for pipeline stages."""
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",  # Use fast model for extraction
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3
        )
        return resp.choices[0].message.content

    # Progress callback
    def on_progress(stage: str, msg: str):
        print(f"[SAFEGUARDED:{stage}] {msg}")

    try:
        # Run the three-stage pipeline
        report = await run_pipeline_v2(
            sources=sources_dict,
            topic=topic,
            title="Research Report",
            llm_call=llm_call,
            batch_size=batch_size,
            min_score=min_score,
            on_progress=on_progress
        )

        # Render as markdown
        final_report = render_hybrid_report(report, use_color=True)

        print(f"[SAFEGUARDED] ✓ Report generated: {report.verified_count} verified facts in {len(report.sections)} themes")

        return {"final_report": final_report}

    except Exception as e:
        logging.exception(f"[SAFEGUARDED] Pipeline failed: {e}")
        print(f"[SAFEGUARDED] ❌ Error: {e}")

        # Fall back to simple report
        return {
            "final_report": f"Error generating safeguarded report: {e}\n\nSources collected: {len(sources_dict)}"
        }
