"""Final report generation node for the Deep Research agent."""

import re
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from open_deep_research.configuration import Configuration
from open_deep_research.models import configurable_model
from open_deep_research.prompts import final_report_generation_prompt
from open_deep_research.state import AgentState, EvidenceSnippet
from open_deep_research.utils import (
    get_api_key_for_model,
    get_model_token_limit,
    get_today_str,
    is_token_limit_exceeded,
)

# Default character caps for prompt truncation (token limit handling)
DEFAULT_MESSAGES_CAP_CHARS = 8000
DEFAULT_FINDINGS_CAP_CHARS = 50000
DEFAULT_VERIFIED_CAP_CHARS = 6000


# Verified Findings disabled message (when Store unavailable)
VERIFIED_DISABLED_MESSAGE = """## Verified Findings

*Verification was disabled for this research run.*

The findings in this report are based on research notes but have not been verified against original source documents.
"""

# Verified Findings empty message (when no PASS snippets)
VERIFIED_NO_QUOTES_MESSAGE = """## Verified Findings

*No quotes passed verification for this research.*

The evidence extraction did not yield quotes that could be verified against source documents.
"""

# Selector-only prompt for generating Verified Findings
SELECTOR_ONLY_PROMPT = """You are generating ONLY the "Verified Findings" section for a research report.

AVAILABLE VERIFIED QUOTES (all have been verified to exist in source documents):
{verified_quotes}

YOUR TASK:
Generate a "## Verified Findings" section using ONLY the quotes above.

STRICT RULES:
1. Start with "## Verified Findings" heading
2. Create a bullet list with 3-5 of the most relevant quotes
3. **DIVERSITY REQUIRED**: Select quotes from DIFFERENT sources - do not pick multiple quotes from the same URL
4. Each bullet MUST follow this EXACT format:
   * **[Topic/Claim]** - "[EXACT QUOTE]" — [Source Title](URL)
5. Copy quotes EXACTLY - do not paraphrase, summarize, or modify
6. Do not add any quotes not in the list above
7. Do not add commentary, analysis, or additional text after the quotes
8. End the section after the bullet list

OUTPUT FORMAT:
## Verified Findings

* **[Topic]** - "[Exact quote from list]" — [Source](URL)
* **[Topic]** - "[Exact quote from list]" — [Source](URL)
* **[Topic]** - "[Exact quote from list]" — [Source](URL)
"""


def format_verified_quotes(snippets: List[EvidenceSnippet], max_quotes: int = 20, max_per_source: int = 3) -> str:
    """Format verified snippets for the selector prompt with source diversity.

    Ensures quotes come from diverse sources by round-robin selection across
    source URLs, preventing all quotes from coming from a single source.

    Args:
        snippets: List of evidence snippets (only PASS status will be used)
        max_quotes: Maximum total quotes to include (default 20)
        max_per_source: Maximum quotes from any single source (default 3)

    Returns:
        Formatted string of verified quotes for the prompt
    """
    verified = [s for s in snippets if s.get("status") == "PASS"]

    if not verified:
        return ""

    # Group by source URL for diversity
    from collections import defaultdict
    by_source = defaultdict(list)
    for snippet in verified:
        url = snippet.get("url", snippet.get("source_id", "unknown"))
        by_source[url].append(snippet)

    # Round-robin selection: pick one from each source, then second from each, etc.
    # This ensures diversity across sources
    diverse_quotes = []
    source_urls = list(by_source.keys())
    round_num = 0

    while len(diverse_quotes) < max_quotes and round_num < max_per_source:
        added_this_round = False
        for url in source_urls:
            if round_num < len(by_source[url]) and len(diverse_quotes) < max_quotes:
                diverse_quotes.append(by_source[url][round_num])
                added_this_round = True
        if not added_this_round:
            break
        round_num += 1

    # Format for the prompt
    formatted_quotes = []
    for i, snippet in enumerate(diverse_quotes, 1):
        quote = snippet.get("quote", "")
        title = snippet.get("source_title", "Unknown Source")
        url = snippet.get("url", snippet.get("source_id", ""))
        formatted_quotes.append(f'{i}. "{quote}" - [{title}]({url})')

    # Log diversity stats
    unique_sources = len(set(s.get("url", s.get("source_id", "")) for s in diverse_quotes))
    print(f"[VERIFIED] Selected {len(diverse_quotes)} quotes from {unique_sources} unique sources")

    return "\n".join(formatted_quotes)


async def generate_verified_findings(
    verified_quotes_str: str,
    model_config: dict,
) -> str:
    """Generate Verified Findings section via dedicated selector-only LLM call.

    This is a separate call to ensure the LLM only selects from verified quotes
    and cannot hallucinate additional content.

    Args:
        verified_quotes_str: Formatted string of verified quotes
        model_config: Model configuration dict

    Returns:
        Generated Verified Findings markdown section
    """
    if not verified_quotes_str:
        return VERIFIED_NO_QUOTES_MESSAGE

    prompt = SELECTOR_ONLY_PROMPT.format(verified_quotes=verified_quotes_str)

    try:
        response = await configurable_model.with_config(model_config).ainvoke([
            HumanMessage(content=prompt)
        ])
        verified_md = response.content.strip()

        # Ensure it starts with the correct heading
        if not verified_md.startswith("## Verified Findings"):
            verified_md = "## Verified Findings\n\n" + verified_md

        return verified_md

    except Exception as e:
        print(f"[VERIFIED] Error generating verified findings: {e}")
        return VERIFIED_NO_QUOTES_MESSAGE


def enforce_verified_section(report: str, verified_md: str) -> str:
    """Post-check: Ensure the Verified Findings section matches exactly.

    If the LLM modified the Verified Findings section, replace it with the
    original verified_md to prevent hallucination.

    Args:
        report: Generated report that may contain modified Verified Findings
        verified_md: The original verified findings markdown (immutable)

    Returns:
        Report with Verified Findings section guaranteed to match verified_md
    """
    # Pattern to find Verified Findings section (## Verified Findings ... until next ## or end)
    pattern = r'(## Verified Findings\s*\n[\s\S]*?)(?=\n## |\Z)'

    match = re.search(pattern, report)

    if not match:
        # No Verified Findings section found - append it
        print("[VERIFIED] Warning: Verified Findings section missing from report. Appending.")
        return report + "\n\n" + verified_md

    existing_section = match.group(1).strip()
    expected_section = verified_md.strip()

    # Check if they match (normalize whitespace for comparison)
    existing_normalized = re.sub(r'\s+', ' ', existing_section)
    expected_normalized = re.sub(r'\s+', ' ', expected_section)

    if existing_normalized != expected_normalized:
        # LLM modified the section - replace it
        print("[VERIFIED] Warning: LLM modified Verified Findings section. Replacing with original.")
        report = re.sub(pattern, verified_md + "\n", report)

    return report


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with optional human review.

    This function uses a two-call approach to eliminate hallucination risk:
    1. First call: Generate Verified Findings via selector-only prompt
    2. Second call: Generate main report with Verified Findings injected as immutable

    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys

    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    configurable = Configuration.from_runnable_config(config)

    # Don't clear source_store if evaluation is enabled (eval needs sources)
    cleared_state = {
        "notes": {"type": "override", "value": []},
        "evidence_snippets": {"type": "override", "value": []},
    }
    if not configurable.run_evaluation:
        cleared_state["source_store"] = {"type": "override", "value": []}  # Clear sources after report

    findings = "\n".join(notes)

    # Step 2: Configure the final report generation model
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }

    # Step 3: Compute all prompt strings ONCE before retry loop
    messages_str = get_buffer_string(state.get("messages", []))
    findings_str = findings  # Already computed above
    verified_disabled = state.get("verified_disabled", False)
    evidence_snippets = state.get("evidence_snippets", [])
    verified_quotes_str = "" if verified_disabled else format_verified_quotes(evidence_snippets)

    # Step 4: Generate Verified Findings section FIRST (separate call)
    if verified_disabled:
        verified_md = VERIFIED_DISABLED_MESSAGE
        print("[VERIFIED] Verification disabled, using placeholder message.")
    elif verified_quotes_str:
        print(f"[VERIFIED] Generating Verified Findings from {len([s for s in evidence_snippets if s.get('status') == 'PASS'])} PASS snippets...")
        verified_md = await generate_verified_findings(verified_quotes_str, writer_model_config)
    else:
        verified_md = VERIFIED_NO_QUOTES_MESSAGE
        print("[VERIFIED] No PASS snippets available.")

    # Step 5: Generate main report with Verified Findings injected as immutable
    # Initialize truncation caps (characters)
    messages_cap = DEFAULT_MESSAGES_CAP_CHARS
    findings_cap = DEFAULT_FINDINGS_CAP_CHARS
    verified_cap = DEFAULT_VERIFIED_CAP_CHARS

    max_retries = 3
    current_retry = 0
    generated_report = None

    while current_retry <= max_retries:
        try:
            # Apply current caps to all strings
            messages_truncated = messages_str[:messages_cap]
            findings_truncated = findings_str[:findings_cap]
            verified_md_truncated = verified_md[:verified_cap]

            # Create prompt with truncated content
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=messages_truncated,
                findings=findings_truncated,
                date=get_today_str()
            )

            # Inject Verified Findings as immutable block
            immutable_block = f"""

=== VERIFIED FINDINGS (IMMUTABLE - DO NOT EDIT) ===
The following section has been pre-generated and verified. Include it EXACTLY as shown in your report. Do not modify, paraphrase, or omit any part of it.

{verified_md_truncated}

=== END VERIFIED FINDINGS ===

IMPORTANT: Copy the "## Verified Findings" section above into your report exactly as written. Do not alter it in any way.
"""
            final_report_prompt += immutable_block

            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            generated_report = final_report.content
            break  # Success - exit retry loop

        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry > max_retries:
                    break  # Exit loop, will return error below

                # Reduce all caps by 10%
                messages_cap = int(messages_cap * 0.9)
                findings_cap = int(findings_cap * 0.9)
                verified_cap = int(verified_cap * 0.9)

                print(f"[REPORT] Token limit exceeded, retry {current_retry}/{max_retries}. "
                      f"Caps: messages={messages_cap}, findings={findings_cap}, verified={verified_cap}")
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }

    # Check if report generation succeeded
    if generated_report is None:
        return {
            "final_report": "Error generating final report: Maximum retries exceeded",
            "messages": [AIMessage(content="Report generation failed after maximum retries")],
            **cleared_state
        }

    # Step 5: Post-check - enforce Verified Findings section integrity
    generated_report = enforce_verified_section(generated_report, verified_md)

    # Count citations in report for logging
    citation_count = len(re.findall(r'\[\d+\]', generated_report))
    print(f"[REPORT] ✓ Complete ({len(generated_report)} chars, {citation_count} citations)")

    # Step 6: Human review checkpoint (if review_mode == "full")
    if configurable.review_mode == "full":
        # Get any flagged issues from fact-check
        flagged_issues = state.get("flagged_issues", [])
        issues_section = ""
        if flagged_issues:
            issues_section = f"""
⚠️ FLAGGED ISSUES FROM FACT-CHECK:
{chr(10).join(f'  ⚠ {issue}' for issue in flagged_issues)}
"""

        # Format review request
        review_request = f"""
═══════════════════════════════════════════════════════════════
HUMAN REVIEW REQUIRED: Final Report
═══════════════════════════════════════════════════════════════
{issues_section}
GENERATED REPORT:
{generated_report[:5000]}{'... [truncated for review]' if len(generated_report) > 5000 else ''}

═══════════════════════════════════════════════════════════════
OPTIONS:
1. Reply with "approve" to accept this report
2. Reply with specific feedback to regenerate with changes
3. Reply with your own edited version of the report
═══════════════════════════════════════════════════════════════
"""
        # Interrupt for human review
        human_response = interrupt(review_request)

        # DEBUG: Log what we received from interrupt
        print(f"\n{'='*60}")
        print(f"[DEBUG REPORT] Raw interrupt response type: {type(human_response)}")
        print(f"[DEBUG REPORT] Raw interrupt response repr: {repr(human_response)[:200]}")
        print(f"{'='*60}\n")

        try:
            # Process human response with error handling
            response_str = str(human_response).strip().strip('"\'')  # Strip quotes for JSON/YAML
            response_lower = response_str.lower()

            print(f"[DEBUG REPORT] After processing: '{response_str[:100]}...' (len={len(response_str)})")

            # Known commands
            approve_commands = ["approve", "ok", "yes", "y", "proceed", "continue", "go", "accept", "done", "good"]

            if response_lower in approve_commands:
                # Human approved - return the report
                print(f"[REVIEW] Human approved final report with command: {response_lower}")
                return {
                    "final_report": generated_report,
                    "messages": [AIMessage(content=generated_report)],
                    **cleared_state
                }
            elif len(response_str) > 200:
                # Human provided their own edited version (substantial text)
                print(f"[REVIEW] Human provided edited report ({len(response_str)} chars).")
                return {
                    "final_report": response_str,
                    "messages": [AIMessage(content=response_str)],
                    **cleared_state
                }
            elif len(response_str) < 20:
                # Very short unknown response - default to approve with warning
                print(f"[REVIEW] Unknown short response '{response_str}'. Defaulting to approve. "
                      f"Use 'approve' to accept, or provide substantial edits (200+ chars).")
                return {
                    "final_report": generated_report,
                    "messages": [AIMessage(content=generated_report)],
                    **cleared_state
                }
            else:
                # Medium-length response - likely feedback, proceed with original
                # (regeneration would require another LLM call which is complex)
                print(f"[REVIEW] Human feedback noted: '{response_str[:50]}...'. Proceeding with original report. "
                      f"To edit, provide the full report text (200+ chars).")
                return {
                    "final_report": generated_report,
                    "messages": [AIMessage(content=generated_report)],
                    **cleared_state
                }
        except Exception as e:
            # If anything goes wrong, log it and default to approve
            print(f"[ERROR REPORT] Exception processing interrupt: {e}")
            print(f"[ERROR REPORT] Response was: {repr(human_response)[:200]}")
            import traceback
            traceback.print_exc()
            # Fallback: default to returning the report
            return {
                "final_report": generated_report,
                "messages": [AIMessage(content=generated_report)],
                **cleared_state
            }

    # No human review required - return the report
    return {
        "final_report": generated_report,
        "messages": [AIMessage(content=generated_report)],
        **cleared_state
    }
