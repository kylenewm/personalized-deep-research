"""Main LangGraph construction for the Deep Research agent.

This module builds the complete deep research workflow by composing
modular nodes from the nodes/ directory.

Flow with Safeguarded Generation (default, use_safeguarded_generation=True):
START -> check_store -> clarify -> brief -> [validate_brief] -> supervisor
      -> [validate_findings] -> safeguarded_report -> [eval] -> END

Legacy Flow (use_safeguarded_generation=False):
START -> check_store -> clarify -> brief -> [validate_brief] -> supervisor
      -> [validate_findings] -> extract_evidence -> verify_evidence -> [claim_pre_check] -> final_report -> [eval] -> END

Note: validate_brief and validate_findings are optional (council features, off by default)
Note: claim_pre_check is Layer 3 verification (legacy path only)
Note: eval is optional (run_evaluation config flag)
"""

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from open_deep_research.configuration import Configuration
from open_deep_research.nodes.brief import validate_brief, write_research_brief
from open_deep_research.nodes.claim_gate import claim_pre_check
from open_deep_research.nodes.clarify import clarify_with_user
from open_deep_research.nodes.eval import run_evaluation_node, should_run_evaluation
from open_deep_research.nodes.extract import extract_evidence
from open_deep_research.nodes.findings import validate_findings
from open_deep_research.nodes.report import final_report_generation
from open_deep_research.nodes.safeguarded_report import safeguarded_report_generation
from open_deep_research.nodes.store import check_store
from open_deep_research.nodes.supervisor import supervisor_subgraph
from open_deep_research.nodes.verify import verify_evidence
from open_deep_research.state import AgentInputState, AgentState


def should_run_claim_check(state: AgentState, config: RunnableConfig) -> Literal["run_claim_check", "skip_claim_check"]:
    """Conditional routing: decide whether to run claim pre-check (Layer 3)."""
    configurable = Configuration.from_runnable_config(config)
    if getattr(configurable, 'claim_pre_check', True):
        return "run_claim_check"
    return "skip_claim_check"


def should_use_safeguarded_generation(state: AgentState, config: RunnableConfig) -> Literal["safeguarded", "legacy"]:
    """Conditional routing: decide whether to use safeguarded generation (Pipeline v2).

    Safeguarded generation uses three-stage extraction:
    1. LLM points to facts, code extracts them
    2. Arranger groups facts by theme
    3. Per-theme synthesis with clear verified/AI distinction

    Legacy path uses:
    extract_evidence → verify_evidence → claim_pre_check → final_report
    """
    configurable = Configuration.from_runnable_config(config)
    if getattr(configurable, 'use_safeguarded_generation', True):
        return "safeguarded"
    return "legacy"


# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("check_store", check_store)                       # S02: Trust Store gating
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("validate_brief", validate_brief)                 # Council 1: Brief validation (optional)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("validate_findings", validate_findings)           # Council 2: Fact-check findings (optional)
deep_researcher_builder.add_node("safeguarded_report", safeguarded_report_generation)  # Pipeline v2: Safeguarded generation (default)
deep_researcher_builder.add_node("extract_evidence", extract_evidence)             # S03: Evidence extraction (legacy)
deep_researcher_builder.add_node("verify_evidence", verify_evidence)               # S04: Evidence verification
deep_researcher_builder.add_node("claim_pre_check", claim_pre_check)               # Layer 3: Claim soft-gate (optional)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase
deep_researcher_builder.add_node("run_evaluation", run_evaluation_node)            # Optional: Post-hoc evaluation
# REMOVED: verify_claims - Too expensive (~$0.45/run) and runs after report (too late to help)

# Define main workflow edges for sequential execution
# See docstring at top of file for flow diagrams
deep_researcher_builder.add_edge(START, "check_store")                             # Entry point: Trust Store gating
# Note: check_store -> clarify_with_user handled by Command return
deep_researcher_builder.add_edge("research_supervisor", "validate_findings")       # Research to fact-check

# Conditional routing: safeguarded generation (Pipeline v2) vs legacy path
deep_researcher_builder.add_conditional_edges(
    "validate_findings",
    should_use_safeguarded_generation,
    {
        "safeguarded": "safeguarded_report",
        "legacy": "extract_evidence"
    }
)
deep_researcher_builder.add_edge("extract_evidence", "verify_evidence")            # Extraction to verification

# Conditional routing: run claim pre-check if enabled, otherwise go directly to report
deep_researcher_builder.add_conditional_edges(
    "verify_evidence",
    should_run_claim_check,
    {
        "run_claim_check": "claim_pre_check",
        "skip_claim_check": "final_report_generation"
    }
)
deep_researcher_builder.add_edge("claim_pre_check", "final_report_generation")     # Claim check to report (legacy)

# Conditional routing after report: run eval if enabled, otherwise END
# Both safeguarded and legacy paths converge here
deep_researcher_builder.add_conditional_edges(
    "safeguarded_report",
    should_run_evaluation,
    {
        "run_eval": "run_evaluation",
        "skip_eval": END
    }
)
deep_researcher_builder.add_conditional_edges(
    "final_report_generation",
    should_run_evaluation,
    {
        "run_eval": "run_evaluation",
        "skip_eval": END
    }
)
deep_researcher_builder.add_edge("run_evaluation", END)                            # Eval to END
# Note: write_research_brief -> validate_brief, validate_brief routing handled by Command returns

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
