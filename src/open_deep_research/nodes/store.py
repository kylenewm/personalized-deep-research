"""Store gating node for the Deep Research agent.

This module implements the Trust Store gating logic (S02) that ensures
the LangGraph Store is available for verification before proceeding.
"""

import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.types import Command

from open_deep_research.state import AgentState


async def check_store(state: AgentState, config: RunnableConfig) -> Command[Literal["clarify_with_user"]]:
    """Check store availability - now always enables verification since sources flow through state.

    UPDATED: Previously this node disabled verification when external LangGraph Store
    was unavailable. Now that sources flow through state.source_store (via Extract API
    caching), we no longer need external Store for verification.

    The S03 (extract_evidence) and S04 (verify_evidence) nodes already check
    state.source_store first before falling back to external Store.

    Args:
        state: Current agent state
        config: Runtime configuration

    Returns:
        Command to proceed to clarify_with_user with verification ENABLED
    """
    # Sources now flow through state.source_store (via Extract API caching)
    # No need to check for external LangGraph Store
    print("[STORE] ✓ Verification enabled (sources flow through state)")
    return Command(
        goto="clarify_with_user",
        update={
            "verified_disabled": False,
            "verified_disabled_reason": ""
        }
    )
