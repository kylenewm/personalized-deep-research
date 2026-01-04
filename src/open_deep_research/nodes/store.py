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
    """Check if LangGraph Store is available for verification.

    This is a fail-fast gating node that runs at the start of the workflow.
    If the Store is unavailable or thread_id is missing, we disable the
    Verified Findings section to prevent false verification claims.

    Per TRUST_ARCH.md Section A (Store Gating):
    - If store is None OR thread_id is missing:
        - Set state["verified_disabled"] = True
        - Log warning: "Store unavailable; Verified section disabled."
    - Else:
        - Set state["verified_disabled"] = False

    Args:
        state: Current agent state
        config: Runtime configuration with store and thread_id

    Returns:
        Command to proceed to clarify_with_user with verified_disabled flag set
    """
    # Step 1: Check if LangGraph Store is available
    store = get_store()

    # Step 2: Check if thread_id is present in config
    thread_id = config.get("configurable", {}).get("thread_id")

    # Step 3: Determine if verification should be disabled
    if store is None:
        reason = "LangGraph Store unavailable"
        logging.warning(f"[STORE] {reason}; Verified Findings section disabled.")
        print(f"[STORE] ⚠️ {reason}; Verified Findings section will be disabled.")
        return Command(
            goto="clarify_with_user",
            update={
                "verified_disabled": True,
                "verified_disabled_reason": reason
            }
        )

    if not thread_id:
        reason = "No thread_id in config"
        logging.warning(f"[STORE] {reason}; Verified Findings section disabled.")
        print(f"[STORE] ⚠️ {reason}; Verified Findings section will be disabled.")
        return Command(
            goto="clarify_with_user",
            update={
                "verified_disabled": True,
                "verified_disabled_reason": reason
            }
        )

    # Step 4: Store is available and thread_id present - verification enabled
    logging.info(f"[STORE] Store available with thread_id={thread_id}; Verification enabled.")
    print(f"[STORE] ✓ Store available (thread: {thread_id[:8]}...); Verification enabled.")
    return Command(
        goto="clarify_with_user",
        update={
            "verified_disabled": False,
            "verified_disabled_reason": ""
        }
    )
