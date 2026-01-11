# Execution Trace: Literal Step-by-Step

This is the actual order of operations when you run a research query. Every step, every input, every output.

**Related docs:**
- `ARCHITECTURE.md` — High-level overview and system design
- `STATE.md` — Current status, decisions, and development sandbox
- `scripts/sandbox_pipeline.py` — Fast iteration on Pipeline v2 without re-running research

**Sandbox:** To test changes to Pipeline v2 (extraction, arranger, synthesis) without re-running research:
```bash
# Capture gold queries once
python scripts/sandbox_pipeline.py --capture "query" --name my_query --review

# Iterate on report generation (free, fast)
python scripts/sandbox_pipeline.py --run my_query
```

---

## PHASE 1: INITIALIZATION

### Step 1.1: check_store
**File:** `nodes/store.py`

```
INPUT:  (nothing from state)
ACTION: Call langgraph.config.get_store()
OUTPUT:
  - verified_disabled = True if store unavailable
  - verified_disabled_reason = "Store unavailable" or ""
NEXT:   → clarify_with_user
```

---

## PHASE 2: CLARIFICATION

### Step 2.1: clarify_with_user
**File:** `nodes/clarify.py`

```
INPUT:  state.messages (user's original query)

ACTION:
  1. Convert messages to string: get_buffer_string(messages)
  2. Build prompt:
     clarify_with_user_instructions.format(
       messages=messages_str,
       date=get_today_str()
     )
  3. LLM call with ClarifyWithUser structured output

LLM PROMPT (prompts.py:3-41):
  "These are the messages that have been exchanged so far...
   Assess whether you need to ask a clarifying question...
   Respond in valid JSON format with these exact keys:
   'need_clarification': boolean,
   'question': '...',
   'verification': '...'"

OUTPUT:
  - If need_clarification=true: interrupt(question), wait for user
  - If need_clarification=false: continue

NEXT:   → write_research_brief
```

---

## PHASE 3: BRIEF GENERATION

### Step 3.1: gather_brief_context (OPTIONAL)
**File:** `utils.py:499-638`
**Condition:** `enable_brief_context=true AND revision_count=0`

```
INPUT:  user_messages string

ACTION 1 - Generate search queries:
  PROMPT (prompts.py:479-499):
    "Generate {num_queries} broad, exploratory search queries...
     User request: {user_messages}
     Return ONLY a valid JSON array of strings."

  LLM OUTPUT: ["query1", "query2", "query3"]

ACTION 2 - Execute Tavily searches:
  tavily_search_async(
    queries,
    max_results=5,
    topic="general",
    include_raw_content=False,  # Just summaries
    days=90                      # Last 3 months
  )

  Also searches news if include_news=true:
  tavily_search_async(queries[:2], topic="news", days=30)

ACTION 3 - Extract context:
  PROMPT (prompts.py:502-526):
    "Extract key context from these search results...
     1. key_entities: Companies, people, products...
     2. recent_events: News from last 3-6 months...
     3. key_metrics: Numbers, percentages, dates...
     4. context_summary: 2-3 sentences..."

  LLM OUTPUT: BriefContext object

OUTPUT: context_block string (formatted for injection)
```

### Step 3.2: write_research_brief
**File:** `nodes/brief.py:29-128`

```
INPUT:
  - state.messages
  - context_block (from 3.1, if gathered)
  - state.feedback_on_brief (if revision)

ACTION:
  1. Get user messages: get_buffer_string(state.messages)

  2. Build prompt:
     transform_messages_into_research_topic_prompt.format(
       messages=user_messages,
       date=get_today_str()
     )
     + context_block (if available)
     + "PREVIOUS FEEDBACK TO ADDRESS:\n{feedback}" (if revision)

  3. LLM call with ResearchQuestion structured output

LLM PROMPT (prompts.py:44-77):
  "You will be given a set of messages...
   Your job is to translate these messages into a more detailed
   and concrete research question...

   Guidelines:
   1. Maximize Specificity and Detail
   2. Fill in Unstated But Necessary Dimensions as Open-Ended
   3. Avoid Unwarranted Assumptions
   4. Use the First Person
   5. Sources - prefer official/primary websites..."

LLM OUTPUT: ResearchQuestion(research_brief="detailed research question...")

ACTION 4 - Create supervisor messages:
  supervisor_messages = [
    SystemMessage(content=lead_researcher_prompt.format(
      date=get_today_str(),
      max_concurrent_research_units=5,  # or 2 in test mode
      max_researcher_iterations=6       # or 2 in test mode
    )),
    HumanMessage(content=research_brief)
  ]

OUTPUT:
  - state.research_brief = "detailed research question..."
  - state.supervisor_messages = [SystemMessage, HumanMessage]

NEXT:   → validate_brief
```

### Step 3.3: validate_brief
**File:** `nodes/brief.py:131-312`
**Condition:** `use_council=true OR review_mode != "none"`

```
INPUT:
  - state.research_brief
  - state.human_approved_brief (if resuming after interrupt)

ACTION (if use_council=true):
  council_vote_on_brief(brief, council_config, config)

  COUNCIL PROCESS (council.py):
    - Run multiple models in parallel (e.g., gpt-4.1, claude-sonnet)
    - Each model returns: decision (approve/revise/reject), confidence, feedback
    - Synthesize feedback from all models
    - Calculate consensus score

ACTION (if review_mode != "none"):
  interrupt(review_request)  # Pause for human review

  Human can respond:
  - "approve" → continue with current brief
  - "ignore" → continue ignoring council feedback
  - <edited brief> → use human's version

OUTPUT:
  - state.human_approved_brief (if human edited)
  - state.council_brief_feedback

ROUTING:
  - If approved → research_supervisor
  - If revise needed → write_research_brief (loop back)

NEXT:   → research_supervisor
```

---

## PHASE 4: RESEARCH SUPERVISOR

### Step 4.1: supervisor (entry)
**File:** `nodes/supervisor.py:23-68`

```
INPUT:
  - state.supervisor_messages = [
      SystemMessage(lead_researcher_prompt),  ← SUPERVISOR'S BRAIN
      HumanMessage(research_brief),           ← THE TASK
      ...previous tool messages if looping...
    ]

ACTION:
  1. Bind tools to model:
     lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

     research_model = configurable_model
       .bind_tools(lead_researcher_tools)
       .with_retry(stop_after_attempt=3)
       .with_config({model: "openai:gpt-4.1", max_tokens: 16000})

  2. LLM call:
     response = await research_model.ainvoke(supervisor_messages)

LLM SEES:
  SystemMessage: lead_researcher_prompt (the full prompt from prompts.py:79-136)
  HumanMessage: research_brief
  [Previous ToolMessages if looping]

LLM DECIDES (via tool_calls):
  Option A: think_tool(reflection="Let me plan my approach...")
  Option B: ConductResearch(research_topic="Detailed topic to research...")
  Option C: ResearchComplete()

OUTPUT:
  - Append response to supervisor_messages
  - research_iterations += 1

NEXT:   → supervisor_tools
```

### Step 4.2: supervisor_tools
**File:** `nodes/supervisor.py:71-274`

```
INPUT:
  - state.supervisor_messages (with latest LLM response)
  - state.research_iterations

EXIT CONDITIONS CHECK (line 106-122):
  - research_iterations > max_researcher_iterations (6 or 2) → END
  - No tool_calls in response → END
  - ResearchComplete in tool_calls → END

IF think_tool CALLED:
  Create ToolMessage(content="Reflection recorded: {reflection}")
  Append to supervisor_messages
  → Loop back to supervisor

IF ConductResearch CALLED (lines 143-268):
  For each ConductResearch call (up to max_concurrent_research_units):

    SPAWN RESEARCHER SUBGRAPH:
      researcher_subgraph.ainvoke({
        "researcher_messages": [HumanMessage(content=research_topic)],
        "research_topic": research_topic
      }, config)

    This runs PHASE 5 (see below)

  Wait for all: asyncio.gather(*research_tasks, return_exceptions=True)

  AGGREGATE RESULTS:
    For each researcher result:
      - compressed_research → ToolMessage content
      - raw_notes → append to state.raw_notes
      - source_store → merge into state.source_store (dedupe by URL)

  QUALITY FILTER (lines 213-246):
    - Skip sources with content < 500 chars
    - Enforce max_total_sources (200)
    - Log: "[SUPERVISOR] Sources: X existing + Y new = Z total"

OUTPUT:
  - state.supervisor_messages += [ToolMessage for each result]
  - state.raw_notes += raw_notes from all researchers
  - state.source_store += sources from all researchers

NEXT:
  - If EXIT CONDITIONS → END (goto validate_findings)
  - Else → supervisor (loop)
```

---

## PHASE 5: RESEARCHER SUBGRAPH (runs for each ConductResearch)

### Step 5.1: researcher
**File:** `nodes/researcher.py:46-105`

```
INPUT:
  - researcher_messages = [HumanMessage(research_topic)]

ACTION:
  1. Get tools:
     tools = [tavily_search, think_tool, ResearchComplete, ...MCP tools]

  2. Build messages:
     messages = [
       SystemMessage(content=research_system_prompt.format(
         date=get_today_str(),
         mcp_prompt=configurable.mcp_prompt or ""
       )),
       ...researcher_messages
     ]

  3. Bind tools and call LLM:
     research_model = configurable_model
       .bind_tools(tools)
       .with_retry(stop_after_attempt=3)

     response = await research_model.ainvoke(messages)

LLM SEES:
  SystemMessage: research_system_prompt (prompts.py:138-183)
    "You are a research assistant...
     <Available Tools>
     1. tavily_search: For conducting web searches
     2. think_tool: For reflection and strategic planning

     <Hard Limits>
     - Simple queries: Use 2-3 search tool calls maximum
     - Complex queries: Use up to 5 search tool calls maximum
     - Always stop: After 5 search tool calls..."

  HumanMessage: research_topic (detailed paragraph from supervisor)

LLM DECIDES (via tool_calls):
  Option A: tavily_search(queries=["query1", "query2"])
  Option B: think_tool(reflection="Analyzing results...")
  Option C: ResearchComplete()

OUTPUT:
  - Append response to researcher_messages
  - tool_call_iterations += 1

NEXT:   → researcher_tools
```

### Step 5.2: researcher_tools
**File:** `nodes/researcher.py:108-188`

```
INPUT:
  - researcher_messages (with latest LLM response)
  - tool_call_iterations

EXIT CONDITIONS CHECK (line 171-177):
  - tool_call_iterations >= max_react_tool_calls (10 or 3) → compress
  - ResearchComplete in tool_calls → compress
  - No tool_calls → compress

IF tavily_search CALLED:
  Execute Step 5.3 (see below)

IF think_tool CALLED:
  Create ToolMessage(content="Reflection recorded: {reflection}")
  Append to researcher_messages
  → Loop back to researcher

OUTPUT:
  - researcher_messages += [ToolMessage for each tool result]

NEXT:
  - If EXIT CONDITIONS → compress_research
  - Else → researcher (loop)
```

### Step 5.3: tavily_search execution
**File:** `utils.py:173-381`

```
INPUT:
  - queries: List[str] (from LLM tool call)
  - max_results: 5
  - topic: "general"

ACTION 1 - Execute searches (lines 192-198):
  tavily_search_async(queries, max_results=5, include_raw_content=True)

  TAVILY API RETURNS:
    [{
      "url": "https://...",
      "title": "...",
      "content": "snippet...",
      "raw_content": "full page content..."
    }, ...]

ACTION 2 - Dedupe by URL (lines 201-206):
  unique_results = {url: result for each result}

ACTION 3 - Summarize each page (lines 233-265):
  For each unique result:
    IF raw_content exists:
      summary = await summarize_webpage(
        model,
        raw_content[:max_char_to_include],  # Default 20000
        research_topic
      )

      SUMMARIZATION PROMPT (prompts.py:402-472):
        "You are tasked with summarizing the raw content of a webpage...
         RESEARCH TOPIC: {research_topic}

         RELEVANCE CHECK (do this first):
         Return 'SKIP' if:
         - Content is mostly navigation, ads, boilerplate
         - Content is paywall/error page
         - Content has no meaningful information related to topic

         If relevant, create summary with:
         1. Main topic/purpose
         2. Key facts, statistics, data points
         3. Important quotes
         ..."

      LLM OUTPUT: Summary(summary="...", key_excerpts="...")

      IF summary == "SKIP":
        Mark URL as skipped (irrelevant)
      ELSE:
        formatted_summary = "<summary>...</summary><key_excerpts>...</key_excerpts>"

ACTION 4 - Store sources (lines 281-368):
  IF use_tavily_extract=true:
    Try tavily_extract(urls) for cleaner content

    EXTRACT API:
      - Handles JavaScript rendering
      - Cleaner content than raw_content
      - Batches of 20 URLs max

  Build source_records:
    [{
      "url": "...",
      "title": "...",
      "content": extracted_content or raw_content,
      "query": original_query,
      "extraction_method": "extract_api" or "search_raw",
      "timestamp": "..."
    }, ...]

  store_source_records(source_records, config)
    → Caches locally AND stores in LangGraph Store if available

ACTION 5 - Format output (lines 374-381):
  OUTPUT STRING:
    "Search results: \n\n
     --- SOURCE 1: {title} ---
     URL: {url}

     SUMMARY:
     {formatted_summary}

     --------------------------------------------------------------------------------

     --- SOURCE 2: {title} ---
     ..."

OUTPUT: Formatted string returned as ToolMessage content
```

### Step 5.4: compress_research
**File:** `nodes/researcher.py:245-342`

```
INPUT:
  - researcher_messages (all messages from research loop)

ACTION:
  1. Add instruction message:
     researcher_messages.append(HumanMessage(
       content=compress_research_simple_human_message
     ))

     HUMAN MESSAGE (prompts.py:247-249):
       "All above messages are about research conducted by an AI Researcher.
        Please clean up these findings.
        DO NOT summarize the information. I want the raw information returned,
        just in a cleaner format. Make sure all relevant information is preserved."

  2. Build messages with system prompt:
     messages = [
       SystemMessage(content=compress_research_system_prompt.format(
         date=get_today_str()
       )),
       ...researcher_messages
     ]

     SYSTEM PROMPT (prompts.py:186-245):
       "You are a research assistant that has conducted research...
        Your job is now to clean up the findings, but preserve all of the
        relevant statements and information...

        <CRITICAL GROUNDING RULES - PREVENT HALLUCINATION>
        You may ONLY include information that appears EXPLICITLY in the search results.

        VERIFICATION CHECKLIST:
        □ Does this exact fact appear in a source? If NO → DO NOT INCLUDE
        □ Can I point to the specific source and quote? If NO → DO NOT INCLUDE
        ...

        <Citation Rules>
        - Assign each unique URL a single citation number [N]
        - End with ### Sources that lists each source..."

  3. LLM call:
     response = await synthesizer_model.ainvoke(messages)

ACTION 2 - Extract sources from tool messages (lines 294-303):
  First try: get_cached_sources(config)  # From Extract API cache

  Fallback: extract_sources_from_tool_messages(tool_messages)

    REGEX PARSING (lines 191-242):
      Pattern: '--- SOURCE \d+: (.+?) ---\s*\nURL: (.+?)\n\n(?:SUMMARY:\n)?(.+?)(?=\n\n---|$)'

      Extracts: title, url, summary content

      Creates: [{
        "url": "...",
        "title": "...",
        "content": "summary content...",
        "extraction_method": "search_parsed",
        "timestamp": "..."
      }, ...]

OUTPUT:
  {
    "compressed_research": "Cleaned findings with citations...",
    "raw_notes": ["All raw tool outputs concatenated..."],
    "source_store": [source_records...]
  }

RETURNS TO: supervisor_tools (Step 4.2)
```

---

## PHASE 6: FINDINGS VALIDATION

### Step 6.1: validate_findings
**File:** `nodes/findings.py`

```
INPUT:
  - state.notes (compressed research from all researchers)

ACTION (if use_findings_council=true):
  PROMPT (prompts.py:371-400):
    "You are a fact-checker reviewing research findings...

     <What to Check>
     1. Fabricated Names: Flag product/model/company names that seem invented
     2. Impossible Dates: Flag dates in the future
     3. Uncited Claims: Flag statistics without citations
     4. Suspicious Statistics: Flag overly precise numbers
     5. Missing Sources: Flag claims referencing sources without URLs
     6. Contradictions: Flag claims that contradict each other

     <Output Format>
     - decision: 'approve', 'revise', or 'reject'
     - confidence: 0.0 to 1.0
     - issues_found: List of specific issues
     - suggested_fixes: Recommendations
     - reasoning: Overall assessment"

  LLM OUTPUT: FindingsReview

OUTPUT:
  - state.flagged_issues = issues_found (for visibility, non-blocking)

NOTE: This is ADVISORY ONLY - does not block pipeline

NEXT:   → safeguarded_report OR extract_evidence (based on config)
```

---

## PHASE 7A: SAFEGUARDED REPORT (Pipeline v2) - DEFAULT PATH

### Step 7A.1: safeguarded_report_generation (entry)
**File:** `nodes/safeguarded_report.py`

```
INPUT:
  - state.source_store (all sources from research)
  - state.research_brief
  - state.notes

ACTION:
  1. Convert sources to dict format:
     sources = {f"src_{i:03d}": source for i, source in enumerate(state.source_store)}

  2. Call pipeline:
     report = await run_pipeline_v2(
       sources=sources,
       topic=state.research_brief,
       title="Research Report",
       llm_call=llm_call_wrapper,
       batch_size=10,
       min_score=0.3
     )

  3. Render:
     final_report = render_hybrid_report(report) or render_html(report)

OUTPUT:
  - state.final_report = rendered report

NOW ENTERING pipeline_v2.py:run_pipeline_v2 (line 936)...
```

### Step 7A.2: Stage 1 - Batched Pointer Extraction
**File:** `pipeline_v2.py:144-175`

```
INPUT:
  - sources: Dict[str, {content, url, title}] (e.g., 141 sources)
  - topic: research question
  - batch_size: 10

ACTION 1 - Batch sources (line 165):
  batches = batch_sources(sources, batch_size=10)
  # Result: 15 batches of 10 sources each (for 141 sources)

FOR EACH BATCH (lines 168-173):

  ACTION 2 - Format sources for prompt:
    formatted = format_sources_for_prompt(batch, max_chars=5000)

    FORMAT (pointer_extract.py:452-465):
      "[src_001] Title of Article
       Content of article truncated to 5000 chars...

       ---

       [src_002] Another Title
       Content...

       ---
       ..."

  ACTION 3 - Build prompt:
    prompt = POINTER_PROMPT.format(sources=formatted, topic=topic)

    POINTER_PROMPT (pointer_extract.py:383-413):
      "Extract facts from these sources that DIRECTLY answer: {topic}

       RELEVANCE CHECK (critical):
       - Only extract facts that help answer the specific question
       - Skip sources that don't contain relevant information
       - Skip generic/promotional content

       For each RELEVANT fact, output:
       - source_id: Match exactly (e.g., 'src_001')
       - keywords: 3-5 SINGLE words that appear in that source
       - context: What this fact is about (3-5 words)
       - relevance: 1-5 score (5=directly answers, 3=somewhat relevant)

       ONLY include facts with relevance >= 3.

       CRITICAL: Use single distinctive words, not phrases.
       - Good: ['Biden', 'October', '2023', 'Executive', 'Order']
       - Bad: ['Executive Order', 'October 2023']

       Sources:
       {sources}

       Output JSON array:
       [{{'source_id': 'src_001', 'keywords': [...], 'context': '...', 'relevance': 5}}]"

  ACTION 4 - LLM call:
    response = await llm_call(prompt)

    LLM OUTPUT (example):
      [
        {"source_id": "src_001", "keywords": ["ElevenLabs", "latency", "200ms"], "context": "Voice model performance", "relevance": 5},
        {"source_id": "src_003", "keywords": ["OpenAI", "Whisper", "transcription"], "context": "Speech recognition", "relevance": 4}
      ]

  ACTION 5 - Parse response:
    pointers = parse_pointer_response(response, min_relevance=3)

    PARSING (pointer_extract.py:416-449):
      - Extract JSON array from response
      - Filter out relevance < 3
      - Create Pointer objects: {source_id, keywords, context}

  ACTION 6 - Extract text for each pointer:
    For each pointer:
      extraction = extract_from_pointer(pointer, batch, min_score=0.3)

      EXTRACTION PROCESS (pointer_extract.py:249-314):
        1. Get source content from batch[pointer.source_id]

        2. find_best_match(keywords, content, min_score=0.3):

           a. Check keyword presence:
              keywords_found = [kw for kw in keywords if kw.lower() in content.lower()]
              match_ratio = len(keywords_found) / len(keywords)

              If match_ratio < 0.3 → return (None, match_ratio)

           b. Split content into sentences:
              sentences = re.split(r'(?<=[.!?])\s+', content)

           c. Score each sentence:
              For each sentence:
                sent_keywords = count keywords in sentence
                score = sent_keywords / len(keywords)

              Track best_sentence, best_score

           d. If best_score < min_score, try pairs of sentences:
              passage = sentences[i] + " " + sentences[i+1]
              Score passage, update best if better

           e. If still < min_score, try triplets

           f. Return (best_sentence, best_score)

        3. clean_extracted_text(text, max_length=300):
           - Strip HTML tags
           - Remove separators (---, ===)
           - Normalize whitespace
           - Truncate at sentence boundary if > 300 chars

        4. is_quality_extraction(text) - QUALITY FILTER:

           REJECTS (pointer_extract.py:90-158):
           - len(text) < 50 chars
           - text.count('|') > 3 (table fragments)
           - "Metadata" and "License" in text
           - Navigation patterns: "[skip to", "[read more]", "log in["
           - Multiple bracket links (>= 3)
           - alpha_ratio < 0.5 (mostly punctuation)
           - Ends with "*", "...", ":" (truncated)
           - Starts with "##", "**", "| " (markdown artifacts)

        5. Return Extraction:
           - status: "verified" (score >= min_score AND quality pass)
                     "partial" (score > 0 but < min_score)
                     "not_found" (no keywords found OR quality fail)
           - extracted_text: the matched text
           - match_score: the score
           - source_url: from source

BATCH OUTPUT:
  extractions = [Extraction, Extraction, ...]

AFTER ALL BATCHES:
  all_extractions = [all Extractions from all batches]
  verified = [e for e in all_extractions if e.status == "verified"]

  LOG: "Total: {len(verified)} verified out of {len(all_extractions)} extractions"
```

### Step 7A.3: Deduplication
**File:** `pipeline_v2.py:212-264`

```
INPUT:
  - verified: List[Extraction] (e.g., 68 verified extractions)

ACTION - deduplicate_extractions():

  PASS 1 - Per-source deduplication (lines 236-244):
    "Keep only best extraction per source URL"

    Sort by match_score descending
    seen_sources = set()
    source_deduped = []

    For each extraction:
      If source_id not in seen_sources:
        source_deduped.append(extraction)
        seen_sources.add(source_id)

    # Result: If same source had 3 pointers, only best one kept

  PASS 2 - Cross-source semantic deduplication (lines 246-264):
    "Remove extractions that say the same thing from different sources"

    kept = []
    For each extraction in source_deduped:

      Normalize text for comparison:
        - Strip markdown: **bold** → bold
        - Strip links: [text](url) → text
        - Normalize whitespace, lowercase

      Check against all kept extractions:
        similarity = compute_text_similarity(ext_normalized, kept_normalized)

        JACCARD SIMILARITY (lines 182-197):
          words1 = set(text1.split())
          words2 = set(text2.split())
          intersection = len(words1 & words2)
          union = len(words1 | words2)
          return intersection / union

        If similarity >= 0.4:
          is_duplicate = True
          break

      If not is_duplicate:
        kept.append(extraction)

OUTPUT:
  deduped = kept  # e.g., 68 → 37 facts

  LOG: "Deduplicated: 68 → 37 facts (31 duplicates removed)"
```

### Step 7A.4: Cleanup
**File:** `pipeline_v2.py:271-333`

```
INPUT:
  - verified: List[Extraction] (deduplicated, e.g., 37)

ACTION - cleanup_extractions():

  FOR EACH BATCH of 20 extractions:

    ACTION 1 - Format for prompt:
      facts_text = format_facts_for_cleanup(batch)

      FORMAT (pointer_extract.py:374-380):
        "[0] First extracted text up to 500 chars...

         [1] Second extracted text...

         [2] Third extracted text..."

    ACTION 2 - Build prompt:
      prompt = CLEANUP_PROMPT.format(facts=facts_text)

      CLEANUP_PROMPT (pointer_extract.py:317-341):
        "For each text, output ONLY the meaningful content with navigation garbage removed.

         Rules:
         - Remove navigation links: [Skip to...], [Read more], [Contact us], Log in, Sign up
         - Remove UI artifacts: Search K, menu items, keyboard shortcuts
         - Remove image markdown: ![](...)
         - Remove header artifacts: # Title, [Site Name](/), page titles with |
         - Remove formatting artifacts: * **Date** ###, changelog prefixes
         - Remove unrelated content: FAQ questions in brackets, promotional text
         - Keep the actual informative content about the topic
         - If there's no meaningful content, output 'NO_CONTENT'
         - CRITICAL: Output must be an EXACT substring of the original (copy-paste, don't rephrase!)

         Texts:
         {facts}

         Output JSON array:
         [{'index': 0, 'cleaned': 'the exact meaningful content here'},
          {'index': 1, 'cleaned': 'NO_CONTENT'},
          ...]"

    ACTION 3 - LLM call:
      response = await llm_call(prompt)

      LLM OUTPUT (example):
        [
          {"index": 0, "cleaned": "ElevenLabs achieves 200ms latency on voice synthesis"},
          {"index": 1, "cleaned": "NO_CONTENT"},
          {"index": 2, "cleaned": "OpenAI's Whisper model supports 97 languages"}
        ]

    ACTION 4 - Parse and verify:
      cleanup_results = parse_cleanup_response(response)

      For each extraction:
        cleaned = results_map.get(index)

        result = verify_and_apply_cleanup(original, cleaned)

        VERIFICATION (pointer_extract.py:355-371):
          If cleaned is None or "NO_CONTENT":
            return None  # REJECT - no meaningful content

          If cleaned in original:  # EXACT SUBSTRING CHECK
            If len(cleaned) >= 50:
              return cleaned  # USE CLEANED VERSION
            Else:
              return None  # TOO SHORT after cleaning
          Else:
            return original  # LLM MODIFIED TEXT - keep original

        If result is None:
          Skip this extraction (don't add to cleaned list)
        Else:
          extraction.extracted_text = result
          cleaned_extractions.append(extraction)

OUTPUT:
  cleaned = [extractions that passed cleanup]  # e.g., 37 → 35

  LOG: "Cleaned: 37 → 35 facts (2 removed as too short)"
```

### Step 7A.5: Stage 2 - Arranger
**File:** `pipeline_v2.py:431-454`

```
INPUT:
  - verified: List[Extraction] (cleaned, e.g., 35)
  - topic: research question

ACTION 1 - Format facts:
  facts_text = format_facts_for_arranger(verified)

  FORMAT (lines 382-392):
    "[1] First fact text truncated to 300 chars...
        Source: Context from pointer

     [2] Second fact text...
        Source: Context

     ..."

ACTION 2 - Build prompt:
  prompt = ARRANGER_PROMPT.format(
    topic=topic,
    num_facts=len(verified),
    facts=facts_text
  )

  ARRANGER_PROMPT (lines 340-379):
    "You are organizing research findings to answer a specific question.

     QUESTION: {topic}

     You have {num_facts} verified facts. Your tasks:

     1. AGGRESSIVE QUALITY FILTER (drop liberally):
        - DROP tutorial/promo content: 'We'll show you...', 'In this guide...'
        - DROP vague claims: 'is a versatile tool', 'offers many features'
        - DROP facts that don't contain specific information
        - DROP anything that reads like marketing copy
        - KEEP only facts with concrete details

     2. RELEVANCE FILTER:
        - Does this fact DIRECTLY help answer the question?
        - If you have to stretch to make it relevant, drop it

     3. GROUP remaining facts by theme (3-5 themes)
        - Themes should map to parts of the answer
        - For 'best X' questions: 'Top Models', 'Performance Metrics', 'Selection Criteria'

     4. FINAL CHECK per fact:
        - Would you cite this in a professional research report?
        - If embarrassing to include, drop it

     VERIFIED FACTS:
     {facts}

     Output JSON:
     {{'groups': [{{'theme': 'Theme Name', 'fact_ids': [1, 2, 5]}}],
       'excluded': [{{'id': 3, 'reason': 'tutorial intro - no specific info'}}]}}

     CRITICAL:
     - Be ruthless. Better 5 strong facts than 15 weak ones.
     - Each fact_id in exactly ONE group OR excluded."

ACTION 3 - LLM call:
  response = await llm_call(prompt)

  LLM OUTPUT (example):
    {
      "groups": [
        {"theme": "Voice Synthesis Performance", "fact_ids": [1, 5, 12, 18]},
        {"theme": "Model Comparisons", "fact_ids": [3, 7, 15, 22]},
        {"theme": "Use Cases & Applications", "fact_ids": [8, 11, 25]},
        {"theme": "Pricing & Availability", "fact_ids": [2, 19, 28]}
      ],
      "excluded": [
        {"id": 4, "reason": "generic marketing - no specifics"},
        {"id": 6, "reason": "tutorial intro"},
        {"id": 9, "reason": "tangential to question"}
      ]
    }

ACTION 4 - Parse response:
  curated = parse_arranger_response(response, verified)

  RESULT: CuratedFacts(
    groups=[ThemeGroup(theme, fact_ids), ...],
    excluded_ids=[4, 6, 9, ...]
  )

OUTPUT:
  curated.groups = [ThemeGroup, ...]  # 4-6 themes
  curated.excluded_ids = [...]  # ~30-50% dropped

  LOG: "Created 4 themes, 25 facts kept, 10 excluded"
```

### Step 7A.6: Stage 3 - Per-Theme Synthesis
**File:** `pipeline_v2.py:504-565`

```
INPUT:
  - curated.groups: List[ThemeGroup]
  - verified: List[Extraction] (all verified facts for lookup)
  - topic: research question

FOR EACH THEME GROUP:

  ACTION 1 - Format facts for this theme:
    facts_text = format_theme_facts([], group.fact_ids, verified)

    FORMAT (lines 491-501):
      "FACT 1 (ID 5): ElevenLabs achieves 200ms latency on voice synthesis...
         Source: Voice model performance

       FACT 2 (ID 12): PlayHT offers real-time streaming with 150ms latency...
         Source: Streaming capabilities

       ..."

  ACTION 2 - Build prompt:
    prompt = THEME_SYNTHESIS_PROMPT.format(
      theme=group.theme,
      topic=topic,
      facts=facts_text
    )

    THEME_SYNTHESIS_PROMPT (lines 461-488):
      "You are writing a section of a research report.

       Theme: {theme}
       Research Topic: {topic}

       VERIFIED FACTS for this section (do NOT modify these):
       {facts}

       Write:
       1. INTRO: 2-3 sentences introducing this theme
       2. TRANSITIONS: One short transition sentence before each fact

       You may DROP 1-2 facts if they don't fit the flow (list IDs in 'dropped').

       CRITICAL:
       - Do NOT rewrite the facts themselves
       - Transitions only CONNECT, they don't add new information
       - Keep transitions to 1 sentence each
       - If a fact contains marketing superlatives ('best', 'most', '#1'),
         your transition should attribute it: 'The source describes...'
         Do NOT present vendor marketing as objective fact.

       Output JSON:
       {{'intro': 'Your theme introduction...',
         'transitions': ['Transition before fact 1', 'Transition before fact 2', ...],
         'dropped': []  // max 2}}"

  ACTION 3 - LLM call:
    response = await llm_call(prompt)

    LLM OUTPUT (example):
      {
        "intro": "Voice synthesis latency is a critical factor for real-time applications. Several providers have achieved sub-second response times, making conversational AI more natural.",
        "transitions": [
          "Leading the pack in speed,",
          "Similarly competitive,",
          "For enterprise deployments,"
        ],
        "dropped": []
      }

  ACTION 4 - Build section:
    kept_ids = [id for id in fact_ids if id not in dropped]
    facts = [verified[id-1] for id in kept_ids]

    section = ThemedSection(
      theme=group.theme,
      intro=intro,
      facts=facts,  # LOCKED - code-extracted text
      transitions=transitions
    )

OUTPUT:
  sections = [ThemedSection, ThemedSection, ...]
```

### Step 7A.7: Assembly
**File:** `pipeline_v2.py:661-698`

```
INPUT:
  - sections: List[ThemedSection]
  - topic, title

ACTION 1 - Generate executive summary:
  overview = "\n".join([f"- {s.theme}: {len(s.facts)} verified findings" for s in sections])

  PROMPT (lines 572-582):
    "Write an executive summary for this research report.
     Topic: {topic}
     The report has these themed sections:
     {sections_overview}
     Write 3-4 sentences summarizing the key findings.
     Do NOT make up facts - only summarize what's in the sections."

  exec_summary = await llm_call(prompt)

ACTION 2 - Generate analysis:
  PROMPT (lines 585-602):
    "Write an analysis section for this research report.
     Topic: {topic}
     Key findings by theme:
     {findings_summary}

     Write 2-3 paragraphs analyzing:
     1. Key patterns or trends
     2. Implications of these findings
     3. Notable gaps or areas needing more research

     This is YOUR interpretation (will be styled as AI analysis)"

  analysis = await llm_call(prompt)

ACTION 3 - Generate conclusion:
  PROMPT (lines 605-614):
    "Write a conclusion for this research report.
     Topic: {topic}
     Key themes covered: {themes}
     Write 2-3 sentences with key takeaways."

  conclusion = await llm_call(prompt)

ACTION 4 - Build report:
  report = HybridReport(
    title=title,
    executive_summary=exec_summary,
    sections=sections,
    analysis=analysis,
    conclusion=conclusion,
    total_extracted=len(all_extractions),
    total_verified=len(verified),
    total_used=sum(len(s.facts) for s in sections)
  )

OUTPUT:
  HybridReport ready for rendering
```

### Step 7A.8: Rendering
**File:** `pipeline_v2.py:705-929`

```
INPUT:
  - report: HybridReport

ACTION - render_hybrid_report() or render_html():

  STRUCTURE:
    <h1>Title</h1>

    <h2>Executive Summary</h2>
    <p class="synthesis">{exec_summary}</p>  ← GRAY (AI-written)

    <h2>Verified Findings</h2>

    FOR EACH SECTION:
      <h3>{theme}</h3>
      <p class="synthesis">{intro}</p>  ← GRAY (AI-written)

      FOR EACH FACT:
        <p class="synthesis">{transition}</p>  ← GRAY (AI-written)
        <div class="verified-fact">  ← GREEN (code-extracted)
          <p>{fact.extracted_text}</p>
          <a href="{fact.source_url}">{fact.pointer.context}</a>
        </div>

    <h2>Analysis & Implications</h2>
    <p class="synthesis">{analysis}</p>  ← GRAY (AI-written)

    <h2>Conclusion</h2>
    <p class="synthesis">{conclusion}</p>  ← GRAY (AI-written)

    <div class="stats">
      Sources: {total_extracted} · Verified: {total_verified} · In report: {total_used}
    </div>

OUTPUT:
  state.final_report = rendered HTML/markdown
```

---

## PHASE 7B: LEGACY PATH (if use_safeguarded_generation=false)

### Step 7B.1: extract_evidence
**File:** `nodes/extract.py`

```
INPUT:
  - state.source_store (all sources)

ACTION:
  For each source:
    If extraction_method == "extract_api":
      Use spacy sentence chunking
    Else:
      Use regex + paragraph extraction

    Create EvidenceSnippet for each passage (15-60 words):
      {
        snippet_id: hash(source_id + quote),
        source_id: source_id,
        url: source.url,
        source_title: source.title,
        quote: extracted_text,
        status: "PENDING"
      }

OUTPUT:
  state.evidence_snippets = [EvidenceSnippet, ...]

NOTE: This is DETERMINISTIC - no LLM calls
```

### Step 7B.2: verify_evidence
**File:** `nodes/verify.py`

```
INPUT:
  - state.evidence_snippets

ACTION:
  For each snippet:
    Check 1 - Strict substring:
      If quote in source_content:
        status = "PASS"

    Check 2 - Fuzzy Jaccard (if strict fails):
      For sliding windows in source:
        similarity = jaccard(quote_words, window_words)
        If similarity > 0.8:
          status = "PASS"

    If neither:
      status = "FAIL"

OUTPUT:
  state.evidence_snippets = [updated with PASS/FAIL status]

NOTE: This is DETERMINISTIC - no LLM calls
```

### Step 7B.3: claim_pre_check (optional)
**File:** `nodes/claim_gate.py`

```
INPUT:
  - state.notes

ACTION:
  Extract claims from notes
  Check each against sources
  Log warnings for unverifiable claims

OUTPUT:
  state.claim_warnings = ["Warning: claim X not found in sources", ...]
```

### Step 7B.4: final_report_generation
**File:** `nodes/report.py`

```
INPUT:
  - state.research_brief
  - state.notes
  - state.evidence_snippets (PASS only)
  - state.messages

ACTION 1 - Generate verified findings section:
  PROMPT (prompts.py:555-578):
    "## VERIFIED FINDINGS SECTION (SELECTOR MODE)

     You MUST include a 'Verified Findings' section.

     <AVAILABLE_VERIFIED_QUOTES>
     {verified_quotes}  ← Only PASS snippets
     </AVAILABLE_VERIFIED_QUOTES>

     SELECTOR MODE RULES:
     1. Select 3-5 of the MOST RELEVANT quotes
     2. Copy each quote EXACTLY - do NOT paraphrase
     3. Format: * '[Exact Quote]' - [Source Title](URL)

     NEVER:
     - Invent or fabricate quotes
     - Modify the wording
     - Use quotes not in the list"

  verified_section = await llm_call(prompt)

ACTION 2 - Generate main report:
  PROMPT (prompts.py:251-368):
    "Based on all the research conducted, create a comprehensive answer...

     <Research Brief>
     {research_brief}
     </Research Brief>

     <Messages>
     {messages}
     </Messages>

     <Findings>
     {findings}
     </Findings>

     <CRITICAL GROUNDING RULES - PREVENT HALLUCINATION>
     You have NO prior knowledge. You may ONLY use information from Findings.

     BEFORE WRITING ANY CLAIM:
     1. Does this EXACT information appear in Findings? If NO, don't write it
     2. Can I cite the specific source? If NO, don't write it
     ..."

  Inject verified_section as IMMUTABLE block

  report = await llm_call(prompt)

ACTION 3 - Post-check:
  enforce_verified_section(report, verified_section)
  # Ensures LLM didn't modify the verified section

OUTPUT:
  state.final_report = report
```

---

## PHASE 8: EVALUATION (optional)

### Step 8.1: run_evaluation
**File:** `nodes/eval.py`
**Condition:** `run_evaluation=true`

```
INPUT:
  - state.final_report
  - state.source_store

ACTION:
  Run citation-first evaluation:
  1. Extract all citations from report
  2. Check each citation against sources
  3. Calculate metrics:
     - Citation accuracy
     - Coverage
     - Hallucination rate

OUTPUT:
  state.eval_result = {metrics}
```

---

## END

Final state contains:
- `final_report`: The generated report
- `source_store`: All sources used
- `evidence_snippets`: Verified quotes (legacy path)
- `eval_result`: Evaluation metrics (if enabled)
