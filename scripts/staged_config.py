"""Shared minimal configuration for staged testing scripts.

This config is optimized for fast iteration:
- No council validation (saves 4-8 LLM calls)
- Minimal research iterations
- No claim verification
- Reduced token limits
"""

# Minimal config overrides for fast testing
MINIMAL_CONFIG = {
    "test_mode": True,
    "use_council": False,
    "use_findings_council": False,
    "use_claim_verification": False,
    "enable_brief_context": False,  # Skip pre-search
    "allow_clarification": False,
    "max_researcher_iterations": 2,  # Need 2: first for think_tool, second for ConductResearch
    "max_react_tool_calls": 2,
    "max_concurrent_research_units": 1,
    "research_model_max_tokens": 4000,
    "summarization_model_max_tokens": 4000,
    "final_report_model_max_tokens": 4000,
}


def get_minimal_runnable_config():
    """Get a RunnableConfig with minimal settings for fast testing."""
    return {"configurable": MINIMAL_CONFIG}


def print_config_summary():
    """Print current config settings."""
    print("\n" + "=" * 50)
    print("STAGED TEST CONFIG (Minimal)")
    print("=" * 50)
    for key, value in MINIMAL_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")
