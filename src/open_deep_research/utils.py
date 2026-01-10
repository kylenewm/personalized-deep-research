"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import (
    summarize_webpage_prompt,
    generate_context_queries_prompt,
    extract_brief_context_prompt,
)
from open_deep_research.state import ResearchComplete, Summary, SourceRecord, BriefContext

##########################
# Domain Filtering
##########################

def is_blocked_domain(url: str, blocked_domains: List[str]) -> bool:
    """Check if a URL belongs to a blocked domain.

    Args:
        url: The URL to check
        blocked_domains: List of domain patterns to block (e.g., ["youtube.com", "reddit.com"])

    Returns:
        True if the URL should be blocked, False otherwise
    """
    if not blocked_domains:
        return False

    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www. prefix for matching
        if domain.startswith("www."):
            domain = domain[4:]

        # Check against blocked domains
        for blocked in blocked_domains:
            blocked = blocked.lower().strip()
            # Match exact domain or subdomain (e.g., "m.youtube.com" matches "youtube.com")
            if domain == blocked or domain.endswith("." + blocked):
                return True
        return False
    except Exception:
        return False


##########################
# Source Storage Utils
##########################

# Module-level cache for sources when external Store is unavailable
# This allows Extract API sources to propagate through state locally
_source_cache: Dict[str, List[dict]] = {}

# Truncation marker for visibility when content is cut off
# Used across the pipeline to indicate data loss
TRUNCATION_MARKER = "\n\n[...CONTENT TRUNCATED...]"


def get_source_store_key(config: RunnableConfig) -> str:
    """Generate unique key for source storage per thread."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    return f"verification_sources_{thread_id}"


def cache_sources(sources: List[dict], config: RunnableConfig) -> None:
    """Cache sources locally for retrieval by researcher nodes.

    This is used when LangGraph Store is unavailable (local development).
    """
    key = get_source_store_key(config)
    if key not in _source_cache:
        _source_cache[key] = []

    # Dedupe by URL
    existing_urls = {s.get("url") for s in _source_cache[key]}
    for source in sources:
        if source.get("url") not in existing_urls:
            _source_cache[key].append(source)
            existing_urls.add(source.get("url"))

    logging.info(f"[SOURCES] Cached {len(sources)} sources locally (total: {len(_source_cache[key])})")


def get_cached_sources(config: RunnableConfig) -> List[dict]:
    """Retrieve cached sources for this thread."""
    key = get_source_store_key(config)
    return _source_cache.get(key, [])


async def store_source_records(
    records: List[dict],
    config: RunnableConfig,
    max_content_length: int = 50000
) -> None:
    """Store source records in LangGraph store for later verification.

    Args:
        records: List of source record dicts with url, title, content, query
        config: Runtime config containing thread_id
        max_content_length: Max chars to store per source (default 50k)
    """
    # Truncate content before caching (with marker if truncated)
    # Uses module-level TRUNCATION_MARKER constant
    for record in records:
        content = record.get("content", "")
        if len(content) > max_content_length:
            record["content"] = content[:max_content_length - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
            record["was_truncated"] = True
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

    # Always cache locally (for local development without external Store)
    cache_sources(records, config)

    store = get_store()
    if store is None:
        logging.warning("[SOURCES] No store available, using local cache only")
        return
    
    key = get_source_store_key(config)
    
    # Get existing sources
    try:
        existing_item = await store.aget(("verification",), key)
        existing = existing_item.value if existing_item else []
    except Exception:
        existing = []
    
    # Add new sources (dedupe by URL)
    # Note: Content already truncated with marker in cache step above
    existing_urls = {s.get("url") for s in existing}
    new_records = []
    for record in records:
        if record.get("url") not in existing_urls:
            record["timestamp"] = datetime.now().isoformat()
            new_records.append(record)
            existing_urls.add(record.get("url"))
    
    if new_records:
        all_records = existing + new_records
        await store.aput(("verification",), key, all_records)
        logging.info(f"[SOURCES] Stored {len(new_records)} new sources (total: {len(all_records)})")


async def get_stored_sources(config: RunnableConfig) -> List[dict]:
    """Retrieve all stored source records for this thread.

    Checks local cache first (for local development), then falls back to external Store.
    """
    # Try local cache first (for local development without external Store)
    cached = get_cached_sources(config)
    if cached:
        logging.info(f"[SOURCES] Using {len(cached)} cached sources")
        return cached

    store = get_store()
    if store is None:
        logging.info("[SOURCES] No store available, returning empty list")
        return []
    
    key = get_source_store_key(config)
    try:
        item = await store.aget(("verification",), key)
        sources = item.value if item else []
        logging.info(f"[SOURCES] Retrieved {len(sources)} sources for verification")
        return sources
    except Exception as e:
        logging.warning(f"[SOURCES] Failed to retrieve sources: {e}")
        return []


##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    
    # Step 2: Deduplicate results by URL to avoid processing the same content multiple times
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    
    # Step 3: Set up the summarization model with configuration
    configurable = Configuration.from_runnable_config(config)

    # Step 3.5: Filter out blocked domains (e.g., YouTube, Reddit, Quora)
    blocked_domains = getattr(configurable, 'blocked_domains', [])
    if blocked_domains:
        pre_filter_count = len(unique_results)
        unique_results = {
            url: data for url, data in unique_results.items()
            if not is_blocked_domain(url, blocked_domains)
        }
        blocked_count = pre_filter_count - len(unique_results)
        if blocked_count > 0:
            print(f"[SEARCH] Filtered {blocked_count} results from blocked domains")

    # Character limit to stay within model token limits (configurable)
    max_char_to_include = configurable.max_content_length
    
    # Initialize summarization model with retry logic
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # Step 4: Create summarization tasks (skip empty content)
    # Use queries as research topic context for relevance filtering
    research_topic = " | ".join(queries[:3])  # Combine first 3 queries as topic context

    async def noop():
        """No-op function for results without raw content."""
        return None

    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include],
            research_topic
        )
        for result in unique_results.values()
    ]
    
    # Step 5: Execute all summarization tasks in parallel
    # BUG FIX: Protect gather with return_exceptions=True
    summaries_raw = await asyncio.gather(*summarization_tasks, return_exceptions=True)

    # Handle exceptions - use original content as fallback
    # Track skipped URLs (irrelevant content marked as SKIP)
    summaries = []
    skipped_urls = set()
    url_list = list(unique_results.keys())

    for i, summary in enumerate(summaries_raw):
        if isinstance(summary, Exception):
            print(f"[SEARCH] Summarization failed for result {i}: {summary}")
            summaries.append(None)  # Will use original content
        elif summary is None:
            # Content was marked irrelevant (SKIP) - track for filtering
            skipped_urls.add(url_list[i])
            summaries.append(None)
        else:
            summaries.append(summary)

    if skipped_urls:
        print(f"[SEARCH] Filtered {len(skipped_urls)} irrelevant sources")

    # Step 6: Combine results with their summaries (exclude skipped)
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
        if url not in skipped_urls  # Exclude irrelevant content
    }
    
    # Step 6.5: Store source records for verification (SIDE EFFECT)
    # Try Tavily Extract API for cleaner content if enabled, fall back to raw search content
    source_records_to_store = []

    # Check if Extract API is enabled
    use_extract = configurable.use_tavily_extract if hasattr(configurable, 'use_tavily_extract') else False

    if use_extract and config:
        # Try Extract API for cleaner content
        urls_to_extract = [url for url in unique_results.keys()]
        try:
            extracted_content = await tavily_extract(urls_to_extract, config)
            extracted_by_url = {item["url"]: item for item in extracted_content}
            logging.info(f"[SOURCES] Extracted {len(extracted_content)}/{len(urls_to_extract)} URLs via Extract API")
        except Exception as e:
            logging.warning(f"[SOURCES] Extract API failed, falling back to raw content: {e}")
            extracted_by_url = {}
    else:
        extracted_by_url = {}

    # Build source records, preferring Extract API content when available
    # Skip irrelevant sources that were marked as SKIP during summarization
    MIN_CONTENT_LENGTH = 500  # Flag sources with less content as potentially truncated
    truncated_urls = []

    for url, result in unique_results.items():
        if url in skipped_urls:
            continue  # Skip irrelevant content

        if not (result.get("raw_content") or result.get("content")):
            continue

        if url in extracted_by_url and extracted_by_url[url].get("content"):
            # Use Extract API content (cleaner)
            content = extracted_by_url[url]["content"]
            source_records_to_store.append({
                "url": url,
                "title": extracted_by_url[url].get("title") or result.get("title", ""),
                "content": content,
                "query": result.get("query", ""),
                "extraction_method": "extract_api"
            })
            if len(content) < MIN_CONTENT_LENGTH:
                truncated_urls.append(url)
        else:
            # Fall back to raw search content
            content = result.get("raw_content") or result.get("content", "")
            source_records_to_store.append({
                "url": url,
                "title": result.get("title", ""),
                "content": content,
                "query": result.get("query", ""),
                "extraction_method": "search_raw"
            })
            if len(content) < MIN_CONTENT_LENGTH:
                truncated_urls.append(url)

    # Attempt to refetch truncated sources via Extract API if not already tried
    if truncated_urls and config and not use_extract:
        logging.info(f"[SOURCES] {len(truncated_urls)} sources have <{MIN_CONTENT_LENGTH} chars, attempting Extract API refetch")
        try:
            refetched = await tavily_extract(truncated_urls, config)
            refetched_by_url = {item["url"]: item for item in refetched}

            # Update source records with refetched content
            for i, record in enumerate(source_records_to_store):
                if record["url"] in refetched_by_url:
                    new_content = refetched_by_url[record["url"]].get("content", "")
                    if len(new_content) > len(record["content"]):
                        source_records_to_store[i]["content"] = new_content
                        source_records_to_store[i]["extraction_method"] = "extract_api_refetch"
                        logging.info(f"[SOURCES] Refetched {record['url'][:50]}... ({len(record['content'])} -> {len(new_content)} chars)")
        except Exception as e:
            logging.warning(f"[SOURCES] Extract API refetch failed: {e}")

    # Log warning for any remaining truncated sources
    still_truncated = [r for r in source_records_to_store if len(r.get("content", "")) < MIN_CONTENT_LENGTH]
    if still_truncated:
        logging.warning(f"[SOURCES] {len(still_truncated)} sources still have <{MIN_CONTENT_LENGTH} chars (may be paywalled/JS-heavy):")
        for r in still_truncated[:5]:  # Log first 5
            logging.warning(f"[SOURCES]   - {r['url'][:60]}... ({len(r.get('content', ''))} chars)")

    # Store asynchronously (fire and forget, don't block tool response)
    if source_records_to_store and config:
        try:
            await store_source_records(source_records_to_store, config)
        except Exception as e:
            logging.warning(f"[SOURCES] Failed to store sources: {e}")
    
    # Step 7: Format the final output
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."
    
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    days: Optional[int] = None,
    config: RunnableConfig = None
):
    """Execute multiple Tavily search queries asynchronously.
    
    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        days: Optional limit to last N days (e.g., 90 for 3 months)
        config: Runtime configuration for API key access
        
    Returns:
        List of search result dictionaries from Tavily API
    """
    # Initialize the Tavily client with API key from config
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # Build search parameters
    search_params = {
        "max_results": max_results,
        "include_raw_content": include_raw_content,
        "topic": topic
    }
    if days is not None:
        search_params["days"] = days
    
    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(query, **search_params)
        for query in search_queries
    ]

    # BUG FIX: Protect gather with return_exceptions=True
    search_results_raw = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Filter out failed searches and log them
    search_results = []
    for i, result in enumerate(search_results_raw):
        if isinstance(result, Exception):
            print(f"[SEARCH] Search query {i} failed: {result}")
            search_results.append({"results": []})  # Empty result
        else:
            search_results.append(result)

    return search_results


async def tavily_extract(
    urls: List[str],
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    """Extract clean content from URLs using Tavily Extract API.

    This provides cleaner content than parsing raw_content from search results,
    as Tavily's extraction handles JavaScript rendering and content extraction.

    Args:
        urls: List of URLs to extract content from
        config: Runtime configuration for API key access

    Returns:
        List of extraction results with 'url', 'title', 'content' fields.
        Returns empty list if extraction fails.
    """
    if not urls:
        return []

    try:
        from tavily import TavilyClient

        api_key = get_tavily_api_key(config)
        if not api_key:
            logging.warning("[EXTRACT] No Tavily API key found, skipping extraction")
            return []

        client = TavilyClient(api_key=api_key)

        # Batch URLs to stay under Tavily's 20 URL limit
        BATCH_SIZE = 20
        extracted = []

        for i in range(0, len(urls), BATCH_SIZE):
            batch_urls = urls[i:i + BATCH_SIZE]
            try:
                result = client.extract(urls=batch_urls)
                for item in result.get("results", []):
                    content = item.get("raw_content") or item.get("content") or ""
                    extracted.append({
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "content": content,
                        "extraction_method": "extract_api"
                    })
            except Exception as batch_error:
                logging.warning(f"[EXTRACT] Batch {i//BATCH_SIZE + 1} failed: {batch_error}")
                continue

        logging.info(f"[EXTRACT] Successfully extracted {len(extracted)}/{len(urls)} URLs")
        return extracted

    except Exception as e:
        logging.warning(f"[EXTRACT] Tavily Extract API failed: {e}")
        return []


##########################
# Brief Context Injection
##########################

async def gather_brief_context(
    user_messages: str,
    config: RunnableConfig,
    max_queries: int = 3,
    max_results: int = 5,
    days: int = 90,
    include_news: bool = True
) -> BriefContext:
    """Gather recent context from Tavily to inform brief generation.
    
    This pre-searches for recent context before brief generation to make
    research briefs more specific and grounded in current events.
    
    Args:
        user_messages: The user's research request/messages
        config: Runtime configuration for API keys
        max_queries: Number of exploratory queries to generate
        max_results: Maximum results per query
        days: Only search last N days (default 90 = 3 months)
        include_news: Also search news sources for recent developments
        
    Returns:
        BriefContext with entities, events, metrics, summary, and sources
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: Generate open-ended search queries using LLM
    query_model = init_chat_model(
        model=configurable.summarization_model,  # Use fast model
        api_key=get_api_key_for_model(configurable.summarization_model, config),
        tags=["langsmith:brief_context_queries"]
    )
    
    query_prompt = generate_context_queries_prompt.format(
        num_queries=max_queries,
        user_messages=user_messages,
        date=get_today_str()
    )
    
    try:
        query_response = await query_model.ainvoke([HumanMessage(content=query_prompt)])
        # Parse JSON array from response
        import json
        queries = json.loads(query_response.content.strip())
        if not isinstance(queries, list):
            queries = [user_messages]  # Fallback to raw user input
    except Exception as e:
        logging.warning(f"Failed to generate context queries: {e}, using raw user input")
        queries = [user_messages]
    
    # Step 2: Execute Tavily searches with recency filter
    all_results = []
    sources_used = []
    
    # Search general topic
    try:
        general_results = await tavily_search_async(
            search_queries=queries,
            max_results=max_results,
            topic="general",
            include_raw_content=False,  # Just summaries for speed
            days=days,
            config=config
        )
        for result_set in general_results:
            if "results" in result_set:
                all_results.extend(result_set["results"])
    except Exception as e:
        logging.warning(f"General search failed: {e}")
    
    # Optionally search news
    if include_news:
        try:
            news_results = await tavily_search_async(
                search_queries=queries[:2],  # Limit news queries
                max_results=3,  # Fewer news results
                topic="news",
                include_raw_content=False,
                days=min(days, 30),  # News should be more recent
                config=config
            )
            for result_set in news_results:
                if "results" in result_set:
                    all_results.extend(result_set["results"])
        except Exception as e:
            logging.warning(f"News search failed: {e}")
    
    # Collect source URLs
    sources_used = list(set(r.get("url", "") for r in all_results if r.get("url")))
    
    # Step 3: Extract structured context from results
    if not all_results:
        logging.warning("No search results for brief context, returning empty context")
        return BriefContext()
    
    # Format results for extraction
    formatted_results = "\n\n".join([
        f"Title: {r.get('title', 'Unknown')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')[:500]}"
        for r in all_results[:15]  # Limit to top 15 results
    ])
    
    extract_model = init_chat_model(
        model=configurable.summarization_model,
        api_key=get_api_key_for_model(configurable.summarization_model, config),
        tags=["langsmith:brief_context_extract"]
    )
    
    extract_prompt = extract_brief_context_prompt.format(
        days=days,
        search_results=formatted_results,
        user_query=user_messages
    )
    
    try:
        extract_response = await extract_model.ainvoke([HumanMessage(content=extract_prompt)])
        import json
        # Try to parse JSON from response
        content = extract_response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        extracted = json.loads(content)
        
        return BriefContext(
            key_entities=extracted.get("key_entities", [])[:10],  # Limit entities
            recent_events=extracted.get("recent_events", [])[:5],  # Limit events
            key_metrics=extracted.get("key_metrics", [])[:8],  # Limit metrics
            context_summary=extracted.get("context_summary", ""),
            sources_used=sources_used[:10]  # Limit sources logged
        )
    except Exception as e:
        logging.warning(f"Failed to extract brief context: {e}")
        # Return partial context with just sources
        return BriefContext(
            context_summary="Context gathering completed but extraction failed.",
            sources_used=sources_used[:10]
        )


def format_brief_context(context: BriefContext, days: int = 90) -> str:
    """Format BriefContext into a string for injection into brief generation prompt."""
    from open_deep_research.prompts import brief_context_injection_instructions
    
    return brief_context_injection_instructions.format(
        days=days,
        entities=", ".join(context.key_entities) if context.key_entities else "None discovered",
        events="; ".join(context.recent_events) if context.recent_events else "None discovered",
        metrics="; ".join(context.key_metrics) if context.key_metrics else "None discovered",
        summary=context.context_summary or "No summary available",
        sources=", ".join(context.sources_used[:5]) if context.sources_used else "None"
    )


async def summarize_webpage(model: BaseChatModel, webpage_content: str, research_topic: str = "") -> str | None:
    """Summarize webpage content using AI model with timeout protection.

    Args:
        model: The chat model configured for summarization
        webpage_content: Raw webpage content to be summarized
        research_topic: The research topic for relevance filtering

    Returns:
        Formatted summary with key excerpts, None if irrelevant/skipped, or original content if summarization fails
    """
    try:
        # Create prompt with current date context and research topic
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str(),
            research_topic=research_topic or "General research"
        )

        # Execute summarization with timeout to prevent hanging
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 60 second timeout for summarization
        )

        # Check for SKIP response (irrelevant content)
        if summary.summary.strip().upper() == "SKIP" or summary.key_excerpts.strip().upper() == "SKIP":
            logging.info("Content marked as irrelevant/SKIP by summarization model")
            return None

        # Format the summary with structured sections
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except asyncio.TimeoutError:
        # Timeout during summarization - return original content
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # Other errors during summarization - log and return original content
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP Utils
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """Exchange Supabase token for MCP access token using OAuth token exchange.
    
    Args:
        supabase_token: Valid Supabase authentication token
        base_mcp_url: Base URL of the MCP server
        
    Returns:
        Token data dictionary if successful, None if failed
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
                    
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """Retrieve stored authentication tokens with expiration validation.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        
    Returns:
        Token dictionary if valid and not expired, None otherwise
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # Retrieve stored tokens
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # Check token expiration
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # Token expired, clean up and return None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """Store authentication tokens in the configuration store.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        tokens: Token dictionary to store
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # Store the tokens
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """Fetch and refresh MCP tokens, obtaining new ones if needed.
    
    Args:
        config: Runtime configuration with authentication details
        
    Returns:
        Valid token dictionary, or None if unable to obtain tokens
    """
    # Try to get existing valid tokens first
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # Extract Supabase token for new token exchange
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # Extract MCP configuration
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # Exchange Supabase token for MCP tokens
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # Store the new tokens and return them
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap MCP tool with comprehensive authentication and error handling.
    
    Args:
        tool: The MCP structured tool to wrap
        
    Returns:
        Enhanced tool with authentication error handling
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """Enhanced coroutine with MCP error handling and user-friendly messages."""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """Recursively search for MCP errors in exception chains."""
            if isinstance(exc, McpError):
                return exc
            
            # Handle ExceptionGroup (Python 3.11+) by checking attributes
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # Execute the original tool functionality
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # Search for MCP-specific errors in the exception chain
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # Not an MCP error, re-raise the original exception
                raise original_error
            
            # Handle MCP-specific error cases
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # Check for authentication/interaction required error
            if error_code == -32003:  # Interaction required error code
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # Extract user-friendly message if available
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # Append URL if provided for user reference
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # For other MCP errors, re-raise the original
            raise original_error
    
    # Replace the tool's coroutine with our enhanced version
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load and configure MCP (Model Context Protocol) tools with authentication.
    
    Args:
        config: Runtime configuration containing MCP server details
        existing_tool_names: Set of tool names already in use to avoid conflicts
        
    Returns:
        List of configured MCP tools ready for use
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: Handle authentication if required
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # Step 2: Validate configuration requirements
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # Step 3: Set up MCP server connection
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # Configure authentication headers if tokens are available
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: When Multi-MCP Server support is merged in OAP, update this code
    
    # Step 4: Load tools from MCP server
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # If MCP server connection fails, return empty list
        return []
    
    # Step 5: Filter and configure tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # Skip tools with conflicting names
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # Only include tools specified in configuration
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # Wrap tool with authentication handling and add to list
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


##########################
# Tool Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.
    
    Args:
        search_api: The search API provider to use (Anthropic, OpenAI, Tavily, or None)
        
    Returns:
        List of configured search tool objects for the specified provider
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic's native web search with usage limits
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI's web search preview functionality
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # Configure Tavily search tool with metadata
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]
        
    elif search_api == SearchAPI.NONE:
        # No search functionality configured
        return []
        
    # Default fallback for unknown search API types
    return []
    
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research, search, and MCP tools.
    
    Args:
        config: Runtime configuration specifying search API and MCP settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Add configured search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # Add MCP tools if configured
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# Model Provider Native Websearch Utils
##########################

def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used in the response.
    
    Args:
        response: The response object from Anthropic's API
        
    Returns:
        True if web search was called, False otherwise
    """
    try:
        # Navigate through the response metadata structure
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # Check for server-side tool usage information
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # Look for web search request count
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # Return True if any web search requests were made
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # Handle cases where response structure is unexpected
        return False

def openai_websearch_called(response):
    """Detect if OpenAI's web search functionality was used in the response.
    
    Args:
        response: The response object from OpenAI's API
        
    Returns:
        True if web search was called, False otherwise
    """
    # Check for tool outputs in the response metadata
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # Look for web search calls in the tool outputs
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token Limit Exceeded Utils
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.
    
    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection
        
    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()
    
    # Step 1: Determine provider from model name if available
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # Step 2: Check provider-specific token limit patterns
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    # Step 3: If provider unknown, check all providers
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an OpenAI exception
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # Check for typical OpenAI token limit error types
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # Look for token-related keywords in error message
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # Check for specific OpenAI error codes
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an Anthropic exception
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # Check for Anthropic-specific error patterns
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic uses specific error messages for token limits
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is a Google/Gemini exception
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # Check for Google-specific resource exhaustion errors
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # Check for specific Google API resource exhaustion patterns
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """Look up the token limit for a specific model.
    
    Args:
        model_string: The model identifier string to look up
        
    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # Model not found in lookup table
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.
    
    This is useful for handling token limit exceeded errors by removing recent context.
    
    Args:
        messages: List of message objects to truncate
        
    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]
    
    # No AI messages found, return original list
    return messages

##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """Get Tavily API key from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
