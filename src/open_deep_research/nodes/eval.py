"""Optional evaluation node for the Deep Research pipeline.

This node runs post-hoc evaluation on the generated report when enabled.
Controlled by the `run_evaluation` config flag.
"""

import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from open_deep_research.configuration import Configuration
from open_deep_research.evaluation import evaluate_report, EvalConfig
from open_deep_research.state import AgentState


async def run_evaluation_node(state: AgentState, config: RunnableConfig) -> dict:
    """Run evaluation on the generated report if enabled.

    This is an optional post-hoc quality check that:
    1. Extracts claims from the report
    2. Verifies each claim against its cited source
    3. Computes quality metrics (hallucination rate, grounding rate, citation accuracy)

    Args:
        state: Agent state containing final_report and source_store
        config: Runtime configuration

    Returns:
        Dictionary with eval_result added to state
    """
    configurable = Configuration.from_runnable_config(config)

    # Check if evaluation is enabled
    if not configurable.run_evaluation:
        logging.info("[EVAL] Evaluation disabled, skipping")
        return {}

    # Check if we have a report to evaluate
    final_report = state.get("final_report", "")
    if not final_report or len(final_report) < 100:
        logging.warning("[EVAL] No report to evaluate, skipping")
        return {}

    # Get source store for verification
    source_store = state.get("source_store", [])
    if not source_store:
        logging.warning("[EVAL] No sources available for verification, skipping eval")
        return {}

    logging.info(f"[EVAL] Running evaluation on report ({len(final_report)} chars, {len(source_store)} sources)")

    # Build eval config
    eval_config = EvalConfig(
        model=configurable.evaluation_model,
        max_claims=30,
        verify_citations=True,
        check_verified_findings=True,
        dry_run=False,
        parallel_batch_size=5
    )

    # Build state dict for evaluation (evaluation.py expects this format)
    eval_state = {
        "final_report": final_report,
        "source_store": source_store,
        "evidence_snippets": state.get("evidence_snippets", []),
        "messages": state.get("messages", [])
    }

    try:
        # Run evaluation
        result = await evaluate_report(eval_state, eval_config)

        # Log summary
        logging.info(f"[EVAL] Complete: {result.claims.grounding_rate:.0%} grounding, "
                    f"{result.claims.hallucination_rate:.0%} hallucination, "
                    f"{result.citations.accuracy:.0%} citation accuracy")

        if result.warnings:
            for warning in result.warnings:
                logging.warning(f"[EVAL] {warning}")

        # Return eval result and clear source_store (no longer needed)
        return {
            "eval_result": result.to_dict(),
            "source_store": {"type": "override", "value": []}  # Clear sources after eval
        }

    except Exception as e:
        logging.error(f"[EVAL] Evaluation failed: {e}")
        return {
            "eval_result": {"error": str(e)},
            "source_store": {"type": "override", "value": []}  # Clear even on error
        }


def should_run_evaluation(state: AgentState, config: RunnableConfig) -> Literal["run_eval", "skip_eval"]:
    """Conditional routing: decide whether to run evaluation.

    Returns:
        "run_eval" if evaluation is enabled and we have a report
        "skip_eval" otherwise
    """
    configurable = Configuration.from_runnable_config(config)

    if not configurable.run_evaluation:
        return "skip_eval"

    final_report = state.get("final_report", "")
    if not final_report or len(final_report) < 100:
        return "skip_eval"

    return "run_eval"
