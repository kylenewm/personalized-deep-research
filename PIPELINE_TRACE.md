# Pipeline Trace: Prompts & Code Locations

## How the Supervisor Gets Instructions

The supervisor doesn't have a standalone system prompt in `supervisor.py`. Instead:

1. **`brief.py:110-114`** creates supervisor_messages with:
   - `SystemMessage(content=lead_researcher_prompt)` ← THE SUPERVISOR'S INSTRUCTIONS
   - `HumanMessage(content=research_brief)` ← THE TASK

2. The supervisor model just reads `supervisor_messages` and decides what tools to call

---

## FULL PROMPT: Supervisor Instructions

**File:** `prompts.py:79-136` (`lead_researcher_prompt`)

```
You are a research supervisor. Your job is to conduct research by calling the "ConductResearch" tool. For context, today's date is {date}.

<Task>
Your focus is to call thePipeline Trace: Prompts & Code Locations
How the Supervisor Gets Instructions
The supervisor doesn't have a standalone system prompt in supervisor.py. Instead:

brief.py:110-114 creates supervisor_messages with:

SystemMessage(content=lead_researcher_prompt) ← THE SUPERVISOR'S INSTRUCTIONS
HumanMessage(content=research_brief) ← THE TASK
The supervisor model just reads supervisor_messages and decides what tools to call

FULL PROMPT: Supervisor Instructions
File: prompts.py:79-136 (lead_researcher_prompt)

You are a research supervisor. Your job is to conduct research by calling the "ConductResearch" tool. For context, today's date is {date}.

<Task>
Your focus is to call the "ConductResearch" tool to conduct research against the overall research question passed in by the user.
When you are completely satisfied with the research findings returned from the tool calls, then you should call the "ResearchComplete" tool to indicate that you are done with your research.
</Task>

<Available Tools>
You have access to three main tools:
1. **ConductResearch**: Delegate research tasks to specialized sub-agents
2. **ResearchComplete**: Indicate that research is complete
3. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool before calling ConductResearch to plan your approach, and after each ConductResearch to assess progress. Do not call think_tool with any other tools in parallel.**
</Available Tools>

<Instructions>
Think like a research manager with limited time and resources. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Decide how to delegate the research** - Carefully consider the question and decide how to delegate the research. Are there multiple independent directions that can be explored simultaneously?
3. **After each call to ConductResearch, pause and assess** - Do I have enough to answer? What's still missing?
</Instructions>

<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards single agent** - Use single agent for simplicity unless the user request has clear opportunity for parallelization
- **Stop when you can answer confidently** - Don't keep delegating research for perfection
- **Limit tool calls** - Always stop after {max_researcher_iterations} tool calls to ConductResearch and think_tool if you cannot find the right sources

**Maximum {max_concurrent_research_units} parallel agents per iteration**
</Hard Limits>

<Show Your Thinking>
Before you call ConductResearch tool call, use think_tool to plan your approach:
- Can the task be broken down into smaller sub-tasks?

After each ConductResearch tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I delegate more research or call ResearchComplete?
</Show Your Thinking>

<Scaling Rules>
**Simple fact-finding, lists, and rankings** can use a single sub-agent:
- *Example*: List the top 10 coffee shops in San Francisco → Use 1 sub-agent

**Comparisons presented in the user request** can use a sub-agent for each element of the comparison:
- *Example*: Compare OpenAI vs. Anthropic vs. DeepMind approaches to AI safety → Use 3 sub-agents
- Delegate clear, distinct, non-overlapping subtopics

**Important Reminders:**
- Each ConductResearch call spawns a dedicated research agent for that specific topic
- A separate agent will write the final report - you just need to gather information
- When calling ConductResearch, provide complete standalone instructions - sub-agents can't see other agents' work
- Do NOT use acronyms or abbreviations in your research questions, be very clear and specific
</Scaling Rules>
FULL PROMPT: Researcher Instructions
File: prompts.py:138-183 (research_system_prompt)

You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

<Task>
Your job is to use tools to gather information about the user's input topic.
You can use any of the tools provided to you to find resources that can help answer the research question. You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
</Task>

<Available Tools>
You have access to two main tools:
1. **tavily_search**: For conducting web searches to gather information
2. **think_tool**: For reflection and strategic planning during research
{mcp_prompt}

**CRITICAL: Use think_tool after each search to reflect on results and plan next steps. Do not call think_tool with the tavily_search or any other tools. It should be to reflect on the results of the search.**
</Available Tools>

<Instructions>
Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Start with broader searches** - Use broad, comprehensive queries first
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 2-3 search tool calls maximum
- **Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>
FULL PROMPT: Brief Generation
File: prompts.py:44-77 (transform_messages_into_research_topic_prompt)

You will be given a set of messages that have been exchanged so far between yourself and the user.
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.

The messages that have been exchanged so far between yourself and the user are:
<Messages>
{messages}
</Messages>

Today's date is {date}.

You will return a single research question that will be used to guide the research.

Guidelines:
1. Maximize Specificity and Detail
- Include all known user preferences and explicitly list key attributes or dimensions to consider.
- It is important that all details from the user are included in the instructions.

2. Fill in Unstated But Necessary Dimensions as Open-Ended
- If certain attributes are essential for a meaningful output but the user has not provided them, explicitly state that they are open-ended or default to no specific constraint.

3. Avoid Unwarranted Assumptions
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat it as flexible or accept all possible options.

4. Use the First Person
- Phrase the request from the perspective of the user.

5. Sources
- If specific sources should be prioritized, specify them in the research question.
- For product and travel research, prefer linking directly to official or primary websites
- For academic or scientific queries, prefer linking directly to the original paper
- For people, try linking directly to their LinkedIn profile
- If the query is in a specific language, prioritize sources published in that language.
Tool Definitions (What Supervisor/Researcher Can Call)
ConductResearch Tool
File: state.py:74-78

class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )
ResearchComplete Tool
File: state.py:80-81

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""
think_tool
File: utils.py:706-731

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    ...
    """
    return f"Reflection recorded: {reflection}"
tavily_search Tool
File: utils.py:173-381

@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API."""
Detailed Flow Diagram
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BRIEF GENERATION (brief.py:29-128)                                  │
│                                                                     │
│ 1. Get user messages                                                │
│ 2. Optional: Gather context from Tavily (brief_context_days=90)     │
│ 3. LLM call with transform_messages_into_research_topic_prompt      │
│    → Output: research_brief (string)                                │
│                                                                     │
│ 4. Create supervisor_messages:                                      │
│    [SystemMessage(lead_researcher_prompt),                          │
│     HumanMessage(research_brief)]                                   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SUPERVISOR LOOP (supervisor.py)                                     │
│                                                                     │
│ supervisor() - Lines 23-68:                                         │
│   1. Bind tools: [ConductResearch, ResearchComplete, think_tool]    │
│   2. Call LLM with supervisor_messages                              │
│   3. LLM decides: think? ConductResearch? ResearchComplete?         │
│                                                                     │
│ supervisor_tools() - Lines 71-274:                                  │
│   - think_tool → record reflection, loop back                       │
│   - ConductResearch → spawn researcher_subgraph (parallel)          │
│   - ResearchComplete → exit to next stage                           │
│   - max iterations (default 6, test 2) → force exit                 │
│                                                                     │
│ For each ConductResearch call:                                      │
│   └─→ researcher_subgraph.ainvoke({                                 │
│         researcher_messages: [HumanMessage(research_topic)],        │
│         research_topic: research_topic                              │
│       })                                                            │
└─────────────────────────────────────────────────────────────────────┘
    │
    │ (for each research topic)
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ RESEARCHER LOOP (researcher.py)                                     │
│                                                                     │
│ researcher() - Lines 46-105:                                        │
│   1. Add SystemMessage(research_system_prompt) to messages          │
│   2. Bind tools: [tavily_search, think_tool, ResearchComplete]      │
│   3. Call LLM                                                       │
│   4. LLM decides: search? think? done?                              │
│                                                                     │
│ researcher_tools() - Lines 108-188:                                 │
│   - tavily_search → execute search, return results                  │
│   - think_tool → record, loop back                                  │
│   - max iterations (default 10, test 3) → go to compress            │
│                                                                     │
│ compress_research() - Lines 245-342:                                │
│   1. Add HumanMessage(compress_research_simple_human_message)       │
│   2. LLM with compress_research_system_prompt                       │
│   3. Extract sources from tool messages                             │
│   4. Return: compressed_research, raw_notes, source_store           │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TAVILY SEARCH DETAIL (utils.py:173-381)                             │
│                                                                     │
│ 1. Execute queries async (tavily_search_async)                      │
│ 2. Dedupe results by URL                                            │
│ 3. For each result:                                                 │
│    - Summarize webpage (summarize_webpage_prompt)                   │
│    - If irrelevant → mark as SKIP                                   │
│ 4. Try Extract API for cleaner content (if enabled)                 │
│ 5. Store sources in source_store                                    │
│ 6. Format output:                                                   │
│    "--- SOURCE 1: {title} ---\nURL: {url}\n\nSUMMARY:\n{content}"   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ (back to supervisor, loop until done)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SAFEGUARDED REPORT (pipeline_v2.py) - DEFAULT PATH                  │
│                                                                     │
│ Stage 1: POINTER EXTRACTION (lines 110-175)                         │
│   - Batch sources (10 per batch)                                    │
│   - Prompt: POINTER_PROMPT (pointer_extract.py:383-413)             │
│   - LLM outputs: {source_id, keywords[], context, relevance}        │
│   - Code: find_best_match() matches keywords to actual text         │
│   - Quality filter: is_quality_extraction() rejects garbage         │
│                                                                     │
│ DEDUPLICATION (lines 212-264)                                       │
│   - Pass 1: Max 1 extraction per source URL                         │
│   - Pass 2: Jaccard similarity > 0.4 = duplicate                    │
│                                                                     │
│ CLEANUP (lines 271-333)                                             │
│   - Prompt: CLEANUP_PROMPT (pointer_extract.py:317-341)             │
│   - LLM outputs cleaned text                                        │
│   - Code verifies: cleaned in original → use, else keep original    │
│                                                                     │
│ Stage 2: ARRANGER (lines 431-454)                                   │
│   - Prompt: ARRANGER_PROMPT (lines 340-379)                         │
│   - LLM groups facts by theme (3-5 themes)                          │
│   - LLM curates: drops ~30-50% as irrelevant                        │
│                                                                     │
│ Stage 3: SYNTHESIS (lines 504-565)                                  │
│   - For each theme:                                                 │
│     - Prompt: THEME_SYNTHESIS_PROMPT (lines 461-488)                │
│     - LLM writes intro + transitions                                │
│     - Facts themselves are LOCKED (code-extracted)                  │
│                                                                     │
│ ASSEMBLY (lines 661-698)                                            │
│   - Generate exec summary, analysis, conclusion                     │
│   - Combine with themed sections                                    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
FINAL REPORT
Quick Reference: What Controls What
Behavior	Controlled By	File:Line
How supervisor breaks down research	lead_researcher_prompt	prompts.py:79-136
How researcher searches	research_system_prompt	prompts.py:138-183
How brief is generated	transform_messages_into_research_topic_prompt	prompts.py:44-77
How webpages are summarized	summarize_webpage_prompt	prompts.py:402-472
How findings are compressed	compress_research_system_prompt	prompts.py:186-245
How pointers are extracted	POINTER_PROMPT	pointer_extract.py:383-413
How garbage is cleaned	CLEANUP_PROMPT	pointer_extract.py:317-341
How facts are grouped	ARRANGER_PROMPT	pipeline_v2.py:340-379
How themes are synthesized	THEME_SYNTHESIS_PROMPT	pipeline_v2.py:461-488
How final report is written (legacy)	final_report_generation_prompt	prompts.py:251-368
Config That Affects Behavior
Config	Default	Test Mode	Effect
max_researcher_iterations	6	2	Supervisor loop limit
max_react_tool_calls	10	3	Researcher search limit
max_concurrent_research_units	5	2	Parallel researchers
use_safeguarded_generation	true	true	Pipeline v2 vs legacy
safeguarded_batch_size	10	10	Sources per pointer batch
safeguarded_min_score	Pipeline Trace: Prompts & Code Locations
How the Supervisor Gets Instructions
The supervisor doesn't have a standalone system prompt in supervisor.py. Instead:

brief.py:110-114 creates supervisor_messages with:

SystemMessage(content=lead_researcher_prompt) ← THE SUPERVISOR'S INSTRUCTIONS
HumanMessage(content=research_brief) ← THE TASK
The supervisor model just reads supervisor_messages and decides what tools to call

FULL PROMPT: Supervisor Instructions
File: prompts.py:79-136 (lead_researcher_prompt)

You are a research supervisor. Your job is to conduct research by calling the "ConductResearch" tool. For context, today's date is {date}.

<Task>
Your focus is to call the "ConductResearch" tool to conduct research against the overall research question passed in by the user.
When you are completely satisfied with the research findings returned from the tool calls, then you should call the "ResearchComplete" tool to indicate that you are done with your research.
</Task>

<Available Tools>
You have access to three main tools:
1. **ConductResearch**: Delegate research tasks to specialized sub-agents
2. **ResearchComplete**: Indicate that research is complete
3. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool before calling ConductResearch to plan your approach, and after each ConductResearch to assess progress. Do not call think_tool with any other tools in parallel.**
</Available Tools>

<Instructions>
Think like a research manager with limited time and resources. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Decide how to delegate the research** - Carefully consider the question and decide how to delegate the research. Are there multiple independent directions that can be explored simultaneously?
3. **After each call to ConductResearch, pause and assess** - Do I have enough to answer? What's still missing?
</Instructions>

<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards single agent** - Use single agent for simplicity unless the user request has clear opportunity for parallelization
- **Stop when you can answer confidently** - Don't keep delegating research for perfection
- **Limit tool calls** - Always stop after {max_researcher_iterations} tool calls to ConductResearch and think_tool if you cannot find the right sources

**Maximum {max_concurrent_research_units} parallel agents per iteration**
</Hard Limits>

<Show Your Thinking>
Before you call ConductResearch tool call, use think_tool to plan your approach:
- Can the task be broken down into smaller sub-tasks?

After each ConductResearch tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I delegate more research or call ResearchComplete?
</Show Your Thinking>

<Scaling Rules>
**Simple fact-finding, lists, and rankings** can use a single sub-agent:
- *Example*: List the top 10 coffee shops in San Francisco → Use 1 sub-agent

**Comparisons presented in the user request** can use a sub-agent for each element of the comparison:
- *Example*: Compare OpenAI vs. Anthropic vs. DeepMind approaches to AI safety → Use 3 sub-agents
- Delegate clear, distinct, non-overlapping subtopics

**Important Reminders:**
- Each ConductResearch call spawns a dedicated research agent for that specific topic
- A separate agent will write the final report - you just need to gather information
- When calling ConductResearch, provide complete standalone instructions - sub-agents can't see other agents' work
- Do NOT use acronyms or abbreviations in your research questions, be very clear and specific
</Scaling Rules>
FULL PROMPT: Researcher Instructions
File: prompts.py:138-183 (research_system_prompt)

You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

<Task>
Your job is to use tools to gather information about the user's input topic.
You can use any of the tools provided to you to find resources that can help answer the research question. You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
</Task>

<Available Tools>
You have access to two main tools:
1. **tavily_search**: For conducting web searches to gather information
2. **think_tool**: For reflection and strategic planning during research
{mcp_prompt}

**CRITICAL: Use think_tool after each search to reflect on results and plan next steps. Do not call think_tool with the tavily_search or any other tools. It should be to reflect on the results of the search.**
</Available Tools>

<Instructions>
Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Start with broader searches** - Use broad, comprehensive queries first
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 2-3 search tool calls maximum
- **Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>
FULL PROMPT: Brief Generation
File: prompts.py:44-77 (transform_messages_into_research_topic_prompt)

You will be given a set of messages that have been exchanged so far between yourself and the user.
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.

The messages that have been exchanged so far between yourself and the user are:
<Messages>
{messages}
</Messages>

Today's date is {date}.

You will return a single research question that will be used to guide the research.

Guidelines:
1. Maximize Specificity and Detail
- Include all known user preferences and explicitly list key attributes or dimensions to consider.
- It is important that all details from the user are included in the instructions.

2. Fill in Unstated But Necessary Dimensions as Open-Ended
- If certain attributes are essential for a meaningful output but the user has not provided them, explicitly state that they are open-ended or default to no specific constraint.

3. Avoid Unwarranted Assumptions
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat it as flexible or accept all possible options.

4. Use the First Person
- Phrase the request from the perspective of the user.

5. Sources
- If specific sources should be prioritized, specify them in the research question.
- For product and travel research, prefer linking directly to official or primary websites
- For academic or scientific queries, prefer linking directly to the original paper
- For people, try linking directly to their LinkedIn profile
- If the query is in a specific language, prioritize sources published in that language.
Tool Definitions (What Supervisor/Researcher Can Call)
ConductResearch Tool
File: state.py:74-78

class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )
ResearchComplete Tool
File: state.py:80-81

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""
think_tool
File: utils.py:706-731

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    ...
    """
    return f"Reflection recorded: {reflection}"
tavily_search Tool
File: utils.py:173-381

@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API."""
Detailed Flow Diagram
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BRIEF GENERATION (brief.py:29-128)                                  │
│                                                                     │
│ 1. Get user messages                                                │
│ 2. Optional: Gather context from Tavily (brief_context_days=90)     │
│ 3. LLM call with transform_messages_into_research_topic_prompt      │
│    → Output: research_brief (string)                                │
│                                                                     │
│ 4. Create supervisor_messages:                                      │
│    [SystemMessage(lead_researcher_prompt),                          │
│     HumanMessage(research_brief)]                                   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SUPERVISOR LOOP (supervisor.py)                                     │
│                                                                     │
│ supervisor() - Lines 23-68:                                         │
│   1. Bind tools: [ConductResearch, ResearchComplete, think_tool]    │
│   2. Call LLM with supervisor_messages                              │
│   3. LLM decides: think? ConductResearch? ResearchComplete?         │
│                                                                     │
│ supervisor_tools() - Lines 71-274:                                  │
│   - think_tool → record reflection, loop back                       │
│   - ConductResearch → spawn researcher_subgraph (parallel)          │
│   - ResearchComplete → exit to next stage                           │
│   - max iterations (default 6, test 2) → force exit                 │
│                                                                     │
│ For each ConductResearch call:                                      │
│   └─→ researcher_subgraph.ainvoke({                                 │
│         researcher_messages: [HumanMessage(research_topic)],        │
│         research_topic: research_topic                              │
│       })                                                            │
└─────────────────────────────────────────────────────────────────────┘
    │
    │ (for each research topic)
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ RESEARCHER LOOP (researcher.py)                                     │
│                                                                     │
│ researcher() - Lines 46-105:                                        │
│   1. Add SystemMessage(research_system_prompt) to messages          │
│   2. Bind tools: [tavily_search, think_tool, ResearchComplete]      │
│   3. Call LLM                                                       │
│   4. LLM decides: search? think? done?                              │
│                                                                     │
│ researcher_tools() - Lines 108-188:                                 │
│   - tavily_search → execute search, return results                  │
│   - think_tool → record, loop back                                  │
│   - max iterations (default 10, test 3) → go to compress            │
│                                                                     │
│ compress_research() - Lines 245-342:                                │
│   1. Add HumanMessage(compress_research_simple_human_message)       │
│   2. LLM with compress_research_system_prompt                       │
│   3. Extract sources from tool messages                             │
│   4. Return: compressed_research, raw_notes, source_store           │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TAVILY SEARCH DETAIL (utils.py:173-381)                             │
│                                                                     │
│ 1. Execute queries async (tavily_search_async)                      │
│ 2. Dedupe results by URL                                            │
│ 3. For each result:                                                 │
│    - Summarize webpage (summarize_webpage_prompt)                   │
│    - If irrelevant → mark as SKIP                                   │
│ 4. Try Extract API for cleaner content (if enabled)                 │
│ 5. Store sources in source_store                                    │
│ 6. Format output:                                                   │
│    "--- SOURCE 1: {title} ---\nURL: {url}\n\nSUMMARY:\n{content}"   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ (back to supervisor, loop until done)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SAFEGUARDED REPORT (pipeline_v2.py) - DEFAULT PATH                  │
│                                                                     │
│ Stage 1: POINTER EXTRACTION (lines 110-175)                         │
│   - Batch sources (10 per batch)                                    │
│   - Prompt: POINTER_PROMPT (pointer_extract.py:383-413)             │
│   - LLM outputs: {source_id, keywords[], context, relevance}        │
│   - Code: find_best_match() matches keywords to actual text         │
│   - Quality filter: is_quality_extraction() rejects garbage         │
│                                                                     │
│ DEDUPLICATION (lines 212-264)                                       │
│   - Pass 1: Max 1 extraction per source URL                         │
│   - Pass 2: Jaccard similarity > 0.4 = duplicate                    │
│                                                                     │
│ CLEANUP (lines 271-333)                                             │
│   - Prompt: CLEANUP_PROMPT (pointer_extract.py:317-341)             │
│   - LLM outputs cleaned text                                        │
│   - Code verifies: cleaned in original → use, else keep original    │
│                                                                     │
│ Stage 2: ARRANGER (lines 431-454)                                   │
│   - Prompt: ARRANGER_PROMPT (lines 340-379)                         │
│   - LLM groups facts by theme (3-5 themes)                          │
│   - LLM curates: drops ~30-50% as irrelevant                        │
│                                                                     │
│ Stage 3: SYNTHESIS (lines 504-565)                                  │
│   - For each theme:                                                 │
│     - Prompt: THEME_SYNTHESIS_PROMPT (lines 461-488)                │
│     - LLM writes intro + transitions                                │
│     - Facts themselves are LOCKED (code-extracted)                  │
│                                                                     │
│ ASSEMBLY (lines 661-698)                                            │
│   - Generate exec summary, analysis, conclusion                     │
│   - Combine with themed sections                                    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
FINAL REPORT
Quick Reference: What Controls What
Behavior	Controlled By	File:Line
How supervisor breaks down research	lead_researcher_prompt	prompts.py:79-136
How researcher searches	research_system_prompt	prompts.py:138-183
How brief is generated	transform_messages_into_research_topic_prompt	prompts.py:44-77
How webpages are summarized	summarize_webpage_prompt	prompts.py:402-472
How findings are compressed	compress_research_system_prompt	prompts.py:186-245
How pointers are extracted	POINTER_PROMPT	pointer_extract.py:383-413
How garbage is cleaned	CLEANUP_PROMPT	pointer_extract.py:317-341
How facts are grouped	ARRANGER_PROMPT	pipeline_v2.py:340-379
How themes are synthesized	THEME_SYNTHESIS_PROMPT	pipeline_v2.py:461-488
How final report is written (legacy)	final_report_generation_prompt	prompts.py:251-368
Config That Affects Behavior
Config	Default	Test Mode	Effect
max_researcher_iterations	6	2	Supervisor loop limit
max_react_tool_calls	10	3	Researcher search limit
max_concurrent_research_units	5	2	Parallel researchers
use_safeguarded_generation	true	true	Pipeline v2 vs legacy
safeguarded_batch_size	10	10	Sources per pointer batch
safeguarded_min_score	0.3	0.3	Keyword match threshold
enable_brief_context	true	-	Pre-search for context
brief_context_days	90	-	How recent for context
State Mutations (What Gets Written Where)
brief.py:write_research_brief
  └─→ state.research_brief
  └─→ state.supervisor_messages (SystemMessage + HumanMessage)

supervisor.py:supervisor_tools
  └─→ state.notes (compressed research from all researchers)
  └─→ state.raw_notes (raw tool outputs)
  └─→ state.source_store (all sources with content)

researcher.py:compress_research
  └─→ returns compressed_research, raw_notes, source_store
  └─→ (aggregated by supervisor_tools)

pipeline_v2.py:run_pipeline_v2
  └─→ state.final_report (via safeguarded_report node)0.3	0.3	Keyword match threshold
enable_brief_context	true	-	Pre-search for context
brief_context_days	90	-	How recent for context
State Mutations (What Gets Written Where)
brief.py:write_research_brief
  └─→ state.research_brief
  └─→ state.supervisor_messages (SystemMessage + HumanMessage)

supervisor.py:supervisor_tools
  └─→ state.notes (compressed research from all researchers)
  └─→ state.raw_notes (raw tool outputs)
  └─→ state.source_store (all sources with content)

researcher.py:compress_research
  └─→ returns compressed_research, raw_notes, source_store
  └─→ (aggregated by supervisor_tools)

pipeline_v2.py:run_pipeline_v2
  └─→ state.final_report (via safeguarded_report node) "ConductResearch" tool to conduct research against the overall research question passed in by the user.
When you are completely satisfied with the research findings returned from the tool calls, then you should call the "ResearchComplete" tool to indicate that you are done with your research.
</Task>

<Available Tools>
You have access to three main tools:
1. **ConductResearch**: Delegate research tasks to specialized sub-agents
2. **ResearchComplete**: Indicate that research is complete
3. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool before calling ConductResearch to plan your approach, and after each ConductResearch to assess progress. Do not call think_tool with any other tools in parallel.**
</Available Tools>

<Instructions>
Think like a research manager with limited time and resources. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Decide how to delegate the research** - Carefully consider the question and decide how to delegate the research. Are there multiple independent directions that can be explored simultaneously?
3. **After each call to ConductResearch, pause and assess** - Do I have enough to answer? What's still missing?
</Instructions>

<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards single agent** - Use single agent for simplicity unless the user request has clear opportunity for parallelization
- **Stop when you can answer confidently** - Don't keep delegating research for perfection
- **Limit tool calls** - Always stop after {max_researcher_iterations} tool calls to ConductResearch and think_tool if you cannot find the right sources

**Maximum {max_concurrent_research_units} parallel agents per iteration**
</Hard Limits>

<Show Your Thinking>
Before you call ConductResearch tool call, use think_tool to plan your approach:
- Can the task be broken down into smaller sub-tasks?

After each ConductResearch tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I delegate more research or call ResearchComplete?
</Show Your Thinking>

<Scaling Rules>
**Simple fact-finding, lists, and rankings** can use a single sub-agent:
- *Example*: List the top 10 coffee shops in San Francisco → Use 1 sub-agent

**Comparisons presented in the user request** can use a sub-agent for each element of the comparison:
- *Example*: Compare OpenAI vs. Anthropic vs. DeepMind approaches to AI safety → Use 3 sub-agents
- Delegate clear, distinct, non-overlapping subtopics

**Important Reminders:**
- Each ConductResearch call spawns a dedicated research agent for that specific topic
- A separate agent will write the final report - you just need to gather information
- When calling ConductResearch, provide complete standalone instructions - sub-agents can't see other agents' work
- Do NOT use acronyms or abbreviations in your research questions, be very clear and specific
</Scaling Rules>
```

---

## FULL PROMPT: Researcher Instructions

**File:** `prompts.py:138-183` (`research_system_prompt`)

```
You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

<Task>
Your job is to use tools to gather information about the user's input topic.
You can use any of the tools provided to you to find resources that can help answer the research question. You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
</Task>

<Available Tools>
You have access to two main tools:
1. **tavily_search**: For conducting web searches to gather information
2. **think_tool**: For reflection and strategic planning during research
{mcp_prompt}

**CRITICAL: Use think_tool after each search to reflect on results and plan next steps. Do not call think_tool with the tavily_search or any other tools. It should be to reflect on the results of the search.**
</Available Tools>

<Instructions>
Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Start with broader searches** - Use broad, comprehensive queries first
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 2-3 search tool calls maximum
- **Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>
```

---

## FULL PROMPT: Brief Generation

**File:** `prompts.py:44-77` (`transform_messages_into_research_topic_prompt`)

```
You will be given a set of messages that have been exchanged so far between yourself and the user.
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.

The messages that have been exchanged so far between yourself and the user are:
<Messages>
{messages}
</Messages>

Today's date is {date}.

You will return a single research question that will be used to guide the research.

Guidelines:
1. Maximize Specificity and Detail
- Include all known user preferences and explicitly list key attributes or dimensions to consider.
- It is important that all details from the user are included in the instructions.

2. Fill in Unstated But Necessary Dimensions as Open-Ended
- If certain attributes are essential for a meaningful output but the user has not provided them, explicitly state that they are open-ended or default to no specific constraint.

3. Avoid Unwarranted Assumptions
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat it as flexible or accept all possible options.

4. Use the First Person
- Phrase the request from the perspective of the user.

5. Sources
- If specific sources should be prioritized, specify them in the research question.
- For product and travel research, prefer linking directly to official or primary websites
- For academic or scientific queries, prefer linking directly to the original paper
- For people, try linking directly to their LinkedIn profile
- If the query is in a specific language, prioritize sources published in that language.
```

---

## Tool Definitions (What Supervisor/Researcher Can Call)

### ConductResearch Tool

**File:** `state.py:74-78`

```python
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )
```

### ResearchComplete Tool

**File:** `state.py:80-81`

```python
class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""
```

### think_tool

**File:** `utils.py:706-731`

```python
@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    ...
    """
    return f"Reflection recorded: {reflection}"
```

### tavily_search Tool

**File:** `utils.py:173-381`

```python
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API."""
```

---

## Detailed Flow Diagram

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BRIEF GENERATION (brief.py:29-128)                                  │
│                                                                     │
│ 1. Get user messages                                                │
│ 2. Optional: Gather context from Tavily (brief_context_days=90)     │
│ 3. LLM call with transform_messages_into_research_topic_prompt      │
│    → Output: research_brief (string)                                │
│                                                                     │
│ 4. Create supervisor_messages:                                      │
│    [SystemMessage(lead_researcher_prompt),                          │
│     HumanMessage(research_brief)]                                   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SUPERVISOR LOOP (supervisor.py)                                     │
│                                                                     │
│ supervisor() - Lines 23-68:                                         │
│   1. Bind tools: [ConductResearch, ResearchComplete, think_tool]    │
│   2. Call LLM with supervisor_messages                              │
│   3. LLM decides: think? ConductResearch? ResearchComplete?         │
│                                                                     │
│ supervisor_tools() - Lines 71-274:                                  │
│   - think_tool → record reflection, loop back                       │
│   - ConductResearch → spawn researcher_subgraph (parallel)          │
│   - ResearchComplete → exit to next stage                           │
│   - max iterations (default 6, test 2) → force exit                 │
│                                                                     │
│ For each ConductResearch call:                                      │
│   └─→ researcher_subgraph.ainvoke({                                 │
│         researcher_messages: [HumanMessage(research_topic)],        │
│         research_topic: research_topic                              │
│       })                                                            │
└─────────────────────────────────────────────────────────────────────┘
    │
    │ (for each research topic)
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ RESEARCHER LOOP (researcher.py)                                     │
│                                                                     │
│ researcher() - Lines 46-105:                                        │
│   1. Add SystemMessage(research_system_prompt) to messages          │
│   2. Bind tools: [tavily_search, think_tool, ResearchComplete]      │
│   3. Call LLM                                                       │
│   4. LLM decides: search? think? done?                              │
│                                                                     │
│ researcher_tools() - Lines 108-188:                                 │
│   - tavily_search → execute search, return results                  │
│   - think_tool → record, loop back                                  │
│   - max iterations (default 10, test 3) → go to compress            │
│                                                                     │
│ compress_research() - Lines 245-342:                                │
│   1. Add HumanMessage(compress_research_simple_human_message)       │
│   2. LLM with compress_research_system_prompt                       │
│   3. Extract sources from tool messages                             │
│   4. Return: compressed_research, raw_notes, source_store           │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TAVILY SEARCH DETAIL (utils.py:173-381)                             │
│                                                                     │
│ 1. Execute queries async (tavily_search_async)                      │
│ 2. Dedupe results by URL                                            │
│ 3. For each result:                                                 │
│    - Summarize webpage (summarize_webpage_prompt)                   │
│    - If irrelevant → mark as SKIP                                   │
│ 4. Try Extract API for cleaner content (if enabled)                 │
│ 5. Store sources in source_store                                    │
│ 6. Format output:                                                   │
│    "--- SOURCE 1: {title} ---\nURL: {url}\n\nSUMMARY:\n{content}"   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ (back to supervisor, loop until done)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SAFEGUARDED REPORT (pipeline_v2.py) - DEFAULT PATH                  │
│                                                                     │
│ Stage 1: POINTER EXTRACTION (lines 110-175)                         │
│   - Batch sources (10 per batch)                                    │
│   - Prompt: POINTER_PROMPT (pointer_extract.py:383-413)             │
│   - LLM outputs: {source_id, keywords[], context, relevance}        │
│   - Code: find_best_match() matches keywords to actual text         │
│   - Quality filter: is_quality_extraction() rejects garbage         │
│                                                                     │
│ DEDUPLICATION (lines 212-264)                                       │
│   - Pass 1: Max 1 extraction per source URL                         │
│   - Pass 2: Jaccard similarity > 0.4 = duplicate                    │
│                                                                     │
│ CLEANUP (lines 271-333)                                             │
│   - Prompt: CLEANUP_PROMPT (pointer_extract.py:317-341)             │
│   - LLM outputs cleaned text                                        │
│   - Code verifies: cleaned in original → use, else keep original    │
│                                                                     │
│ Stage 2: ARRANGER (lines 431-454)                                   │
│   - Prompt: ARRANGER_PROMPT (lines 340-379)                         │
│   - LLM groups facts by theme (3-5 themes)                          │
│   - LLM curates: drops ~30-50% as irrelevant                        │
│                                                                     │
│ Stage 3: SYNTHESIS (lines 504-565)                                  │
│   - For each theme:                                                 │
│     - Prompt: THEME_SYNTHESIS_PROMPT (lines 461-488)                │
│     - LLM writes intro + transitions                                │
│     - Facts themselves are LOCKED (code-extracted)                  │
│                                                                     │
│ ASSEMBLY (lines 661-698)                                            │
│   - Generate exec summary, analysis, conclusion                     │
│   - Combine with themed sections                                    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
FINAL REPORT
```

---

## Quick Reference: What Controls What

| Behavior | Controlled By | File:Line |
|----------|---------------|-----------|
| How supervisor breaks down research | `lead_researcher_prompt` | prompts.py:79-136 |
| How researcher searches | `research_system_prompt` | prompts.py:138-183 |
| How brief is generated | `transform_messages_into_research_topic_prompt` | prompts.py:44-77 |
| How webpages are summarized | `summarize_webpage_prompt` | prompts.py:402-472 |
| How findings are compressed | `compress_research_system_prompt` | prompts.py:186-245 |
| How pointers are extracted | `POINTER_PROMPT` | pointer_extract.py:383-413 |
| How garbage is cleaned | `CLEANUP_PROMPT` | pointer_extract.py:317-341 |
| How facts are grouped | `ARRANGER_PROMPT` | pipeline_v2.py:340-379 |
| How themes are synthesized | `THEME_SYNTHESIS_PROMPT` | pipeline_v2.py:461-488 |
| How final report is written (legacy) | `final_report_generation_prompt` | prompts.py:251-368 |

---

## Config That Affects Behavior

| Config | Default | Test Mode | Effect |
|--------|---------|-----------|--------|
| `max_researcher_iterations` | 6 | 2 | Supervisor loop limit |
| `max_react_tool_calls` | 10 | 3 | Researcher search limit |
| `max_concurrent_research_units` | 5 | 2 | Parallel researchers |
| `use_safeguarded_generation` | true | true | Pipeline v2 vs legacy |
| `safeguarded_batch_size` | 10 | 10 | Sources per pointer batch |
| `safeguarded_min_score` | 0.3 | 0.3 | Keyword match threshold |
| `enable_brief_context` | true | - | Pre-search for context |
| `brief_context_days` | 90 | - | How recent for context |

---

## State Mutations (What Gets Written Where)

```
brief.py:write_research_brief
  └─→ state.research_brief
  └─→ state.supervisor_messages (SystemMessage + HumanMessage)

supervisor.py:supervisor_tools
  └─→ state.notes (compressed research from all researchers)
  └─→ state.raw_notes (raw tool outputs)
  └─→ state.source_store (all sources with content)

researcher.py:compress_research
  └─→ returns compressed_research, raw_notes, source_store
  └─→ (aggregated by supervisor_tools)

pipeline_v2.py:run_pipeline_v2
  └─→ state.final_report (via safeguarded_report node)
```
