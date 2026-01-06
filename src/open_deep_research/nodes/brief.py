"""Research brief creation and validation nodes for the Deep Research agent."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from open_deep_research.configuration import Configuration
from open_deep_research.council import (
    CouncilConfig,
    council_vote_on_brief,
    log_council_decision,
)
from open_deep_research.models import configurable_model
from open_deep_research.prompts import (
    lead_researcher_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import AgentState, ResearchQuestion
from open_deep_research.utils import (
    format_brief_context,
    gather_brief_context,
    get_api_key_for_model,
    get_today_str,
)


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["validate_brief"]]:
    """Transform user messages into a structured research brief and initialize supervisor.

    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.

    If brief context injection is enabled, it first gathers recent context from Tavily
    to make the brief more specific and grounded in current events.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to brief validation (council) with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    user_messages = get_buffer_string(state.get("messages", []))

    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Gather recent context if enabled (pre-search for recency)
    # Skip on revisions - context already gathered on first attempt, no need to re-search
    context_block = ""
    revision_count = state.get("council_revision_count", 0)
    print(f"[BRIEF] Generating research plan...")
    if configurable.enable_brief_context and revision_count == 0:
        print(f"[BRIEF] Gathering context from recent sources...")
        try:
            brief_context = await gather_brief_context(
                user_messages=user_messages,
                config=config,
                max_queries=configurable.brief_context_max_queries,
                max_results=configurable.brief_context_max_results,
                days=configurable.brief_context_days,
                include_news=configurable.brief_context_include_news
            )
            context_block = format_brief_context(brief_context, configurable.brief_context_days)
            if brief_context.sources_used:
                entities_preview = brief_context.key_entities[:3] if brief_context.key_entities else []
                print(f"[BRIEF] Context: {len(brief_context.sources_used)} sources, entities: {entities_preview}")
        except Exception as e:
            # Log but don't fail - context is optional enhancement
            import logging
            logging.warning(f"Brief context gathering failed, proceeding without: {e}")

    # Step 3: Generate structured research brief from user messages + context
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=user_messages,
        date=get_today_str()
    )

    # Inject gathered context if available
    if context_block:
        prompt_content += f"\n\n{context_block}"

    # Include council feedback if this is a revision attempt
    feedback_on_brief = state.get("feedback_on_brief", [])
    if feedback_on_brief:
        prompt_content += f"\n\nPREVIOUS FEEDBACK TO ADDRESS:\n{feedback_on_brief[-1]}"

    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    print(f"[BRIEF] ✓ Brief generated ({len(response.research_brief)} chars)")

    # Step 3: Initialize supervisor with research brief and instructions
    # Use effective values (reduced in test mode)
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.get_effective_max_concurrent_research_units(),
        max_researcher_iterations=configurable.get_effective_max_researcher_iterations()
    )

    return Command(
        goto="validate_brief",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def validate_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor", "write_research_brief"]]:
    """Validate the research brief using the LLM Council and optionally human review.

    In this architecture, Council = Advisor, Human = Authority:
    - Council provides feedback and suggestions (not approve/reject decisions)
    - If review_mode != "none", human reviews brief + council feedback
    - Human can approve, edit, or ignore council feedback
    - Once human approves (or if review_mode == "none"), proceed to research

    Args:
        state: Current agent state with research brief
        config: Runtime configuration with council and review settings

    Returns:
        Command to proceed to research (after human approval if needed)
    """
    configurable = Configuration.from_runnable_config(config)

    # Get current brief (use human-approved version if available)
    brief = state.get("human_approved_brief") or state.get("research_brief", "")
    if not brief:
        # No brief to validate - proceed anyway
        return Command(goto="research_supervisor")

    # Check if human already approved (resuming after interrupt)
    if state.get("human_approved_brief"):
        # Human has approved - proceed to research without re-validation
        print(f"[REVIEW] Using human-approved brief. Proceeding to research.")
        return Command(goto="research_supervisor")

    # Get council feedback (advisory only, not approve/reject)
    council_feedback = ""
    if configurable.use_council:
        print(f"[COUNCIL] Voting on brief...")
        council_config = CouncilConfig(
            models=configurable.council_models,
            min_consensus_for_approve=configurable.council_min_consensus,
            max_revision_rounds=configurable.council_max_revisions,
        )

        verdict = await council_vote_on_brief(brief, council_config, config)
        log_council_decision(verdict)

        print(f"[COUNCIL] Result: {verdict.decision.upper()} ({verdict.consensus_score:.0%} consensus)")

        # Format council feedback for human review
        council_feedback = f"""
COUNCIL FEEDBACK (Advisory):
Decision: {verdict.decision.upper()}
Consensus: {verdict.consensus_score:.0%}

{verdict.synthesized_feedback}
"""

    # Human review checkpoint
    if configurable.review_mode != "none":
        # Format the review request for human
        review_request = f"""
═══════════════════════════════════════════════════════════════
HUMAN REVIEW REQUIRED: Research Brief
═══════════════════════════════════════════════════════════════

RESEARCH BRIEF:
{brief}

{council_feedback if council_feedback else "(Council review disabled)"}

═══════════════════════════════════════════════════════════════
OPTIONS:
1. Reply with "approve" to proceed with this brief
2. Reply with your edited version of the brief to use instead
3. Reply with "ignore" to proceed without council suggestions
═══════════════════════════════════════════════════════════════
"""
        # Interrupt for human review - execution pauses here
        human_response = interrupt(review_request)

        # DEBUG: Log what we received from interrupt
        print(f"\n{'='*60}")
        print(f"[DEBUG BRIEF] Raw interrupt response type: {type(human_response)}")
        print(f"[DEBUG BRIEF] Raw interrupt response repr: {repr(human_response)[:200]}")
        print(f"{'='*60}\n")

        try:
            # Process human response with error handling
            response_str = str(human_response).strip().strip('"\'')  # Strip quotes for JSON/YAML
            response_lower = response_str.lower()

            print(f"[DEBUG BRIEF] After processing: '{response_str}' (len={len(response_str)})")

            # Known commands
            approve_commands = ["approve", "ok", "yes", "y", "proceed", "continue", "go", "accept"]
            ignore_commands = ["ignore", "skip", "no council", "dismiss"]

            if response_lower in approve_commands or response_lower in ignore_commands:
                # Human approved the brief as-is
                print(f"[REVIEW] Human approved brief with command: {response_lower}")
                return Command(
                    goto="research_supervisor",
                    update={
                        "human_approved_brief": brief,
                        "council_brief_feedback": council_feedback
                    }
                )
            elif len(response_str) < 50:
                # Short response that's not a known command - likely a mistake
                # Default to approve with warning
                print(f"[REVIEW] Unknown short response '{response_str}'. Defaulting to approve. "
                      f"Use 'approve', 'ignore', or provide a full edited brief (50+ chars).")
                return Command(
                    goto="research_supervisor",
                    update={
                        "human_approved_brief": brief,
                        "council_brief_feedback": council_feedback
                    }
                )
            else:
                # Human provided an edited brief (substantial text) - use their version
                print(f"[REVIEW] Human provided edited brief ({len(response_str)} chars).")
                return Command(
                    goto="research_supervisor",
                    update={
                        "human_approved_brief": response_str,
                        "council_brief_feedback": council_feedback,
                        "research_brief": response_str,
                        # Update supervisor messages with new brief
                        "supervisor_messages": {
                            "type": "override",
                            "value": [
                                SystemMessage(content=lead_researcher_prompt.format(
                                    date=get_today_str(),
                                    max_concurrent_research_units=configurable.get_effective_max_concurrent_research_units(),
                                    max_researcher_iterations=configurable.get_effective_max_researcher_iterations()
                                )),
                                HumanMessage(content=response_str)
                            ]
                        }
                    }
                )
        except Exception as e:
            # If anything goes wrong, log it and default to approve
            print(f"[ERROR BRIEF] Exception processing interrupt: {e}")
            print(f"[ERROR BRIEF] Response was: {repr(human_response)[:200]}")
            import traceback
            traceback.print_exc()
            # Fallback: default to approve
            return Command(
                goto="research_supervisor",
                update={
                    "human_approved_brief": brief,
                    "council_brief_feedback": council_feedback
                }
            )

    # No human review required (review_mode == "none")
    # In auto mode, still use council decision for routing
    if configurable.use_council:
        revision_count = state.get("council_revision_count", 0)

        if verdict.decision == "approve":
            return Command(
                goto="research_supervisor",
                update={"council_brief_feedback": council_feedback}
            )
        elif verdict.decision == "reject" or revision_count >= configurable.council_max_revisions:
            print(f"[COUNCIL] Auto-mode: Proceeding after {revision_count} revisions.")
            return Command(
                goto="research_supervisor",
                update={"council_brief_feedback": council_feedback}
            )
        else:
            return Command(
                goto="write_research_brief",
                update={
                    "feedback_on_brief": [verdict.synthesized_feedback],
                    "council_revision_count": revision_count + 1,
                    "council_brief_feedback": council_feedback
                }
            )

    # No council, no human review - just proceed
    return Command(goto="research_supervisor")
