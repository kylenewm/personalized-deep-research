"""Supervisor nodes and subgraph for the Deep Research agent."""

import asyncio
from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import Configuration
from open_deep_research.models import configurable_model
from open_deep_research.nodes.researcher import researcher_subgraph
from open_deep_research.state import ConductResearch, ResearchComplete, SupervisorState, SupervisorOutputState
from open_deep_research.utils import (
    get_api_key_for_model,
    get_notes_from_tool_calls,
    is_token_limit_exceeded,
    think_tool,
)


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.

    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.

    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.

    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase

    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings

    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)

    # BUG FIX: Check for empty messages before accessing [-1]
    if not supervisor_messages:
        print("[SUPERVISOR] Warning: No messages in state, ending research")
        return Command(
            goto=END,
            update={
                "notes": [],
                "research_brief": state.get("research_brief", ""),
                "source_store": state.get("source_store", []),
            }
        )

    most_recent_message = supervisor_messages[-1]

    # Define exit criteria for research phase (use effective values for test mode)
    exceeded_allowed_iterations = research_iterations > configurable.get_effective_max_researcher_iterations()
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "source_store": state.get("source_store", []),  # BUG FIX: Propagate sources
            }
        )

    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        print(f"[RESEARCH] Iteration {research_iterations}: {len(conduct_research_calls)} research tasks")
        try:
            # Limit concurrent research units to prevent resource exhaustion (use effective values for test mode)
            max_units = configurable.get_effective_max_concurrent_research_units()
            allowed_conduct_research_calls = conduct_research_calls[:max_units]
            overflow_conduct_research_calls = conduct_research_calls[max_units:]

            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]

            # BUG FIX: Protect gather with return_exceptions=True
            tool_results_raw = await asyncio.gather(*research_tasks, return_exceptions=True)

            # Filter out exceptions and log them
            tool_results = []
            successful_calls = []
            for i, result in enumerate(tool_results_raw):
                if isinstance(result, Exception):
                    print(f"[SUPERVISOR] Research task {i} failed: {result}")
                    # Add error message for failed research
                    all_tool_messages.append(ToolMessage(
                        content=f"Error: Research failed - {result}",
                        name=allowed_conduct_research_calls[i]["name"],
                        tool_call_id=allowed_conduct_research_calls[i]["id"]
                    ))
                else:
                    tool_results.append(result)
                    successful_calls.append(allowed_conduct_research_calls[i])

            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, successful_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {max_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

            # Aggregate sources from all research results for verification
            # With quality filtering, deduplication, and limit enforcement
            max_sources = getattr(configurable, 'max_total_sources', 200)
            min_content_len = getattr(configurable, 'min_source_content_length', 500)

            existing_sources = state.get("source_store", [])
            existing_urls = {s.get("url") for s in existing_sources if s.get("url")}

            new_sources = []
            skipped_low_quality = 0
            for observation in tool_results:
                for source in observation.get("source_store", []):
                    url = source.get("url")
                    content = source.get("content", "") or ""

                    # Skip duplicates
                    if not url or url in existing_urls:
                        continue

                    # Quality filter: skip sources with insufficient content
                    if len(content) < min_content_len:
                        skipped_low_quality += 1
                        continue

                    new_sources.append(source)
                    existing_urls.add(url)

            if skipped_low_quality > 0:
                print(f"[SUPERVISOR] Filtered {skipped_low_quality} low-quality sources (<{min_content_len} chars)")

            # Enforce limit - stop adding when we hit max
            slots_remaining = max(0, max_sources - len(existing_sources))
            if slots_remaining < len(new_sources):
                print(f"[SUPERVISOR] Source limit reached ({max_sources}), dropping {len(new_sources) - slots_remaining} sources")
            new_sources = new_sources[:slots_remaining]

            if new_sources:
                update_payload["source_store"] = existing_sources + new_sources
                print(f"[SUPERVISOR] Sources: {len(existing_sources)} existing + {len(new_sources)} new = {len(existing_sources) + len(new_sources)} total")

        except Exception as e:
            # Handle research execution errors
            print(f"[SUPERVISOR] Research execution error: {e}")

            # BUG FIX: Removed `or True` - only end on token limit, not all errors
            if is_token_limit_exceeded(e, configurable.research_model):
                print("[SUPERVISOR] Token limit exceeded, ending research phase")
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", ""),
                        "source_store": state.get("source_store", []),
                    }
                )
            # For other errors, log and continue (don't crash the whole pipeline)
            print(f"[SUPERVISOR] Non-fatal error, continuing with partial results")

    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    )


# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
# NOTE: output=SupervisorOutputState ensures source_store gets mapped back to parent AgentState
supervisor_builder = StateGraph(SupervisorState, output=SupervisorOutputState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()
