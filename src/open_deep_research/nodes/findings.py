"""Findings validation node for the Deep Research agent."""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.models import FindingsReview, configurable_model
from open_deep_research.prompts import fact_check_findings_prompt
from open_deep_research.state import AgentState
from open_deep_research.utils import get_api_key_for_model, get_today_str


async def validate_findings(state: AgentState, config: RunnableConfig) -> dict:
    """Fact-check research findings and flag issues for human review.

    In this architecture, Council = Advisor, Human = Authority:
    - Council fact-checks findings and FLAGS issues (doesn't auto-reject)
    - Issues are stored in state for human review
    - If review_mode == "full" and issues found, human reviews before report
    - Human can: approve anyway, request re-research, or proceed

    Note: S05 wiring - this node now flows to extract_evidence via graph edge.

    Args:
        state: Current agent state with research notes
        config: Runtime configuration with fact-check and review settings

    Returns:
        Dictionary with flagged_issues update (routing handled by graph edges)
    """
    configurable = Configuration.from_runnable_config(config)

    # Skip fact-checking if disabled
    if not configurable.use_findings_council:
        return {}

    # Get research findings from notes
    notes = state.get("notes", [])
    if not notes:
        # No findings to validate - proceed anyway
        return {}

    findings_text = "\n\n".join(notes)

    # Configure the fact-checking model
    model_config = {
        "model": configurable.council_models[0] if configurable.council_models else "openai:gpt-4.1",
        "max_tokens": 4096,
        "api_key": get_api_key_for_model(configurable.council_models[0] if configurable.council_models else "openai:gpt-4.1", config),
        "tags": ["langsmith:fact_check"]
    }

    fact_check_model = (
        configurable_model
        .with_structured_output(FindingsReview)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # Generate fact-check prompt
    prompt = fact_check_findings_prompt.format(
        date=get_today_str(),
        findings=findings_text
    )

    try:
        review: FindingsReview = await fact_check_model.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        # If fact-check fails, log and proceed
        print(f"[FACT-CHECK] Error during fact-check: {e}. Proceeding to extraction.")
        return {}

    # Format flagged issues
    flagged_issues = []
    if review.issues_found:
        flagged_issues = review.issues_found

    # Log the fact-check results
    print(f"\n{'='*60}")
    print(f"FACT-CHECK RESULTS (Advisory - Human has final say)")
    print(f"Assessment: {review.decision.upper()}")
    print(f"Confidence: {review.confidence:.0%}")
    if flagged_issues:
        print(f"Issues Flagged: {len(flagged_issues)}")
        for issue in flagged_issues[:3]:
            print(f"  ⚠ {issue[:100]}...")
    else:
        print(f"No issues flagged.")
    print(f"{'='*60}\n")

    # Log detailed feedback (non-blocking, for offline review)
    if flagged_issues:
        print(f"\n{'='*60}")
        print(f"[FACT-CHECK] ⚠️ {len(flagged_issues)} issues flagged (advisory):")
        for issue in flagged_issues[:5]:
            print(f"  - {issue[:100]}...")
        if review.suggested_fixes:
            print(f"\n[FACT-CHECK] Suggested fixes:")
            for fix in review.suggested_fixes[:3]:
                print(f"  → {fix[:100]}...")
        print(f"\n[FACT-CHECK] Reasoning: {review.reasoning[:200]}...")
        print(f"[FACT-CHECK] Flags stored in state for offline review.")
        print(f"{'='*60}\n")

    # Always proceed to extraction (non-blocking)
    # Flagged issues stored in state for later review
    # Routing handled by graph edge: validate_findings -> extract_evidence
    return {"flagged_issues": flagged_issues if flagged_issues else []}
