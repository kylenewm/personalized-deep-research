"""Main LangGraph construction for the Deep Research agent.

This module builds the complete deep research workflow by composing
modular nodes from the nodes/ directory.
"""

from langgraph.graph import END, START, StateGraph

from open_deep_research.configuration import Configuration
from open_deep_research.nodes.brief import validate_brief, write_research_brief
from open_deep_research.nodes.clarify import clarify_with_user
from open_deep_research.nodes.extract import extract_evidence
from open_deep_research.nodes.findings import validate_findings
from open_deep_research.nodes.report import final_report_generation
from open_deep_research.nodes.store import check_store
from open_deep_research.nodes.supervisor import supervisor_subgraph
from open_deep_research.nodes.verify import verify_claims, verify_evidence
from open_deep_research.state import AgentInputState, AgentState


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
deep_researcher_builder.add_node("validate_brief", validate_brief)                 # Council 1: Brief validation
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("validate_findings", validate_findings)           # Council 2: Fact-check findings
deep_researcher_builder.add_node("extract_evidence", extract_evidence)             # S03: Evidence extraction
deep_researcher_builder.add_node("verify_evidence", verify_evidence)               # S04: Evidence verification
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase
deep_researcher_builder.add_node("verify_claims", verify_claims)                   # Claim verification health check

# Define main workflow edges for sequential execution
# Flow: check_store -> clarify -> brief -> validate_brief -> supervisor -> validate_findings
#       -> extract_evidence -> verify_evidence -> final_report -> verify_claims -> END
deep_researcher_builder.add_edge(START, "check_store")                             # Entry point: Trust Store gating
# Note: check_store -> clarify_with_user handled by Command return
deep_researcher_builder.add_edge("research_supervisor", "validate_findings")       # Research to fact-check
deep_researcher_builder.add_edge("validate_findings", "extract_evidence")          # S05: Fact-check to extraction
deep_researcher_builder.add_edge("extract_evidence", "verify_evidence")            # S05: Extraction to verification
deep_researcher_builder.add_edge("verify_evidence", "final_report_generation")     # S05: Verification to report
deep_researcher_builder.add_edge("final_report_generation", "verify_claims")       # Report to claim verification
deep_researcher_builder.add_edge("verify_claims", END)                             # Final exit point
# Note: write_research_brief -> validate_brief, validate_brief routing handled by Command returns

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
