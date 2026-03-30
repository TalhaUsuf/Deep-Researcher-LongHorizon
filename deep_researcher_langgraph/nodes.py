"""LangGraph node functions for the deep research workflow.

Each node takes a DeepResearchState and returns a partial state update dict.
All LLM calls go through the shared LLMService instance stored in the graph config.

Tree recursion model:
  The original implementation recurses PER-RESULT at each depth level — each
  result's follow-up questions spawn a new set of queries at the next depth.

  This LangGraph version uses a stack-based approach (`branch_stack`) to
  faithfully reproduce the call-stack recursion:
    - `fan_out_branches` pushes one {query, depth} item per result onto the stack
    - `pick_next_branch` pops the top item and sets combined_query + current_depth
    - When a sub-branch fans out, its children are pushed ON TOP of remaining
      sibling branches, ensuring DFS ordering identical to the original
    - When the stack empties, all branches at all depths have been processed
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from gpt_researcher import GPTResearcher
from gpt_researcher.actions.query_processing import get_search_results
from gpt_researcher.actions.retriever import get_retrievers
from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from gpt_researcher.utils.enum import ReportType, ReportSource

from .llm_service import LLMService
from .prompts import (
    GENERATE_RESEARCH_PLAN_PROMPT,
    GENERATE_REPORT_PROMPT,
    GENERATE_SEARCH_QUERIES_PROMPT,
    PROCESS_RESEARCH_RESULTS_PROMPT,
)
from .schemas import (
    FollowUpQuestionsResponse,
    ResearchAnalysis,
    SearchQueriesResponse,
)
from .state import BranchItem, DeepResearchState, ResearchProgress

logger = logging.getLogger(__name__)

MAX_CONTEXT_WORDS = 25000


def _get_llm_service(config: RunnableConfig) -> LLMService:
    """Extract the LLMService from the LangGraph runnable config."""
    try:
        return config["configurable"]["llm_service"]
    except KeyError:
        raise ValueError(
            "LLMService not found in graph config. "
            "Pass it via config={'configurable': {'llm_service': llm_service_instance}}."
        )


def _get_config(config: RunnableConfig) -> "Config":
    """Extract the shared Config from the LangGraph runnable config."""
    try:
        return config["configurable"]["config"]
    except KeyError:
        raise ValueError(
            "Config not found in graph config. "
            "Pass it via config={'configurable': {'config': config_instance}}."
        )


def _get_progress(config: RunnableConfig) -> Optional[ResearchProgress]:
    return config["configurable"].get("progress")


def _count_words(text: Any) -> int:
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    return len(str(text).split())


def _trim_context(context_list: List[str], max_words: int = MAX_CONTEXT_WORDS) -> List[str]:
    """Trim context list to stay within word limit, keeping most recent items."""
    total_words = 0
    trimmed = []
    for item in reversed(context_list):
        words = _count_words(item)
        if total_words + words <= max_words:
            trimmed.insert(0, item)
            total_words += words
        else:
            break
    return trimmed


def _notify_progress(state: DeepResearchState, config: dict) -> None:
    """Fire the on_progress callback if provided."""
    on_progress = state.get("on_progress")
    progress = _get_progress(config)
    if on_progress and progress:
        on_progress(progress)


# ---------------------------------------------------------------------------
# Node: generate_research_plan
# ---------------------------------------------------------------------------

async def generate_research_plan(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate follow-up questions from initial search results to guide deep research.

    Mirrors original DeepResearchSkill.generate_research_plan():
      - Runs initial search across all configured retrievers
      - Uses strategic LLM with ReasoningEfforts.High and temperature=0.4
      - Original does not set max_tokens explicitly (inherits default 4000 from
        create_chat_completion)
    """
    llm_service = _get_llm_service(config)
    cfg = _get_config(config)
    query = state["query"]
    breadth = state.get("breadth", 4)

    # Run initial search across all configured retrievers (matches original)
    retrievers = get_retrievers(state.get("headers", {}), cfg)
    all_search_results: List[dict] = []
    for retriever_class in retrievers:
        try:
            results = await get_search_results(query, retriever_class)
            all_search_results.extend(results)
        except Exception as e:
            logger.warning(f"Error with retriever {retriever_class.__name__}: {e}")

    if not all_search_results and retrievers:
        raise RuntimeError(
            f"All {len(retrievers)} retrievers failed during initial search for query: {query!r}. "
            "Check retriever configuration and API keys."
        )

    logger.info(f"Initial web knowledge obtained: {len(all_search_results)} results")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    messages = GENERATE_RESEARCH_PLAN_PROMPT.format_messages(
        query=query,
        current_time=current_time,
        search_results=str(all_search_results),
        num_questions=breadth,
    )

    # Original: strategic LLM, ReasoningEfforts.High, temperature=0.4,
    # max_tokens not set (defaults to 4000 from create_chat_completion)
    strategic_llm = llm_service.get_strategic_llm(
        temperature=0.4,
        max_tokens=4000,
        reasoning_effort=ReasoningEfforts.High.value,
    )
    response: FollowUpQuestionsResponse = await llm_service.invoke_structured(
        strategic_llm, FollowUpQuestionsResponse, messages
    )

    questions = response.questions[:breadth]

    # Build combined query exactly like the original
    qa_pairs = [f"Q: {q}\nA: Automatically proceeding with research" for q in questions]
    combined_query = (
        f"Initial Query: {query}\nFollow-up Questions and Answers:\n"
        + "\n".join(qa_pairs)
    )

    # Initialize progress tracker
    progress = _get_progress(config)
    if progress:
        progress.total_depth = state["depth"]
        progress.total_breadth = breadth

    return {
        "initial_search_results": all_search_results,
        "follow_up_questions": questions,
        "combined_query": combined_query,
        "current_depth": state["depth"],
        "current_breadth": 0,
        "all_learnings": [],
        "all_citations": {},
        "all_visited_urls": [],
        "all_context": [],
        "all_sources": [],
        "branch_stack": [],
        "total_queries": 0,
        "completed_queries": 0,
        "messages": [HumanMessage(content=query)],
    }


# ---------------------------------------------------------------------------
# Node: generate_search_queries
# ---------------------------------------------------------------------------

async def generate_search_queries(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate diverse search queries for the current research level.

    Mirrors original DeepResearchSkill.generate_search_queries():
      - Uses strategic LLM with temperature=0.4
      - Original does not set max_tokens (defaults to 4000)
      - Breadth halves at each depth: max(2, breadth // 2)
    """
    llm_service = _get_llm_service(config)
    current_depth = state["current_depth"]
    original_breadth = state["breadth"]

    # Calculate breadth for current level (halves each level, min 2)
    levels_deep = state["depth"] - current_depth
    breadth = original_breadth
    for _ in range(levels_deep):
        breadth = max(2, breadth // 2)

    query = state.get("combined_query") or state["query"]

    messages = GENERATE_SEARCH_QUERIES_PROMPT.format_messages(
        query=query,
        num_queries=breadth,
    )

    # Original: strategic LLM, temperature=0.4, no explicit max_tokens (defaults to 4000)
    strategic_llm = llm_service.get_strategic_llm(temperature=0.4, max_tokens=4000)
    response: SearchQueriesResponse = await llm_service.invoke_structured(
        strategic_llm, SearchQueriesResponse, messages
    )

    search_queries = [
        {"query": q.query, "research_goal": q.research_goal}
        for q in response.queries[:breadth]
    ]

    logger.info(
        f"Generated {len(search_queries)} queries at depth {current_depth}: "
        f"{[q['query'] for q in search_queries]}"
    )

    # Update progress
    progress = _get_progress(config)
    if progress:
        progress.total_queries += len(search_queries)
    _notify_progress(state, config)

    return {
        "search_queries": search_queries,
        "total_queries": state.get("total_queries", 0) + len(search_queries),
    }


# ---------------------------------------------------------------------------
# Node: execute_research
# ---------------------------------------------------------------------------

async def execute_research(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute research for all search queries at the current level.

    Mirrors original DeepResearchSkill.deep_research() inner loop:
      - Uses asyncio.Semaphore with concurrency_limit (original default 2)
      - For each query, spawns a GPTResearcher instance
      - Propagates MCP config, visited URLs, tone, headers
      - Extracts learnings via strategic LLM (temperature=0.4, max_tokens=1000,
        ReasoningEfforts.High — matching original process_research_results)
      - Handles per-query errors gracefully (continues with successful results)
    """
    llm_service = _get_llm_service(config)
    search_queries = state["search_queries"]
    concurrency_limit = state.get("concurrency_limit", 2)
    visited_urls: Set[str] = set(state.get("all_visited_urls", []))

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def _process_single_query(serp_query: dict) -> dict | None:
        async with semaphore:
            try:
                # Update progress
                progress = _get_progress(config)
                if progress:
                    progress.current_query = serp_query["query"]
                _notify_progress(state, config)

                # Spawn a GPTResearcher per query (matches original exactly)
                researcher = GPTResearcher(
                    query=serp_query["query"],
                    report_type=ReportType.ResearchReport.value,
                    report_source=ReportSource.Web.value,
                    tone=state.get("tone", "Objective"),
                    websocket=state.get("websocket"),
                    config_path=state.get("config_path"),
                    headers=state.get("headers", {}),
                    visited_urls=visited_urls,
                    mcp_configs=state.get("mcp_configs"),
                    mcp_strategy=state.get("mcp_strategy"),
                )

                context = await researcher.conduct_research()
                query_visited = researcher.visited_urls
                sources = researcher.research_sources

                context_str = "\n".join(context) if isinstance(context, list) else (context or "")

                # Analyze results: original uses strategic LLM, temperature=0.4,
                # max_tokens=1000, ReasoningEfforts.High
                messages = PROCESS_RESEARCH_RESULTS_PROMPT.format_messages(
                    query=serp_query["query"],
                    context=context_str,
                )

                strategic_llm = llm_service.get_strategic_llm(
                    temperature=0.4,
                    max_tokens=1000,
                    reasoning_effort=ReasoningEfforts.High.value,
                )
                analysis: ResearchAnalysis = await llm_service.invoke_structured(
                    strategic_llm, ResearchAnalysis, messages
                )

                learnings = [item.insight for item in analysis.learnings]
                citations = {
                    item.insight: item.source_url
                    for item in analysis.learnings
                    if item.source_url
                }
                follow_ups = analysis.follow_up_questions

                # Update progress
                if progress:
                    progress.completed_queries += 1
                    progress.current_breadth += 1
                _notify_progress(state, config)

                return {
                    "query": serp_query["query"],
                    "learnings": learnings,
                    "follow_up_questions": follow_ups,
                    "citations": citations,
                    "context": context_str,
                    "sources": sources or [],
                    "visited_urls": list(query_visited),
                    "research_goal": serp_query.get("research_goal", ""),
                }
            except asyncio.CancelledError:
                raise
            except (TypeError, AttributeError, KeyError, ImportError, ValueError) as e:
                raise RuntimeError(
                    f"Configuration or programming error during query '{serp_query['query']}': {e}"
                ) from e
            except Exception as e:
                import traceback
                logger.error(
                    f"Error processing query '{serp_query['query']}': {e}\n"
                    f"{traceback.format_exc()}"
                )
                return None

    tasks = [_process_single_query(q) for q in search_queries]
    raw_results = await asyncio.gather(*tasks)
    results = [r for r in raw_results if r is not None]

    if not results and search_queries:
        raise RuntimeError(
            f"All {len(search_queries)} research queries failed. "
            "Check logs for individual query errors."
        )

    # Aggregate results
    new_learnings = []
    new_citations = {}
    new_visited = []
    new_context = []
    new_sources = []

    for r in results:
        new_learnings.extend(r["learnings"])
        new_citations.update(r["citations"])
        new_visited.extend(r["visited_urls"])
        if r["context"]:
            new_context.append(r["context"])
        new_sources.extend(r["sources"])

    return {
        "research_results": results,
        "all_learnings": new_learnings,
        "all_citations": new_citations,
        "all_visited_urls": new_visited,
        "all_context": new_context,
        "all_sources": new_sources,
        "completed_queries": state.get("completed_queries", 0) + len(results),
        "current_breadth": len(results),
    }


# ---------------------------------------------------------------------------
# Node: fan_out_branches
# ---------------------------------------------------------------------------

async def fan_out_branches(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Push per-result branch queries onto the stack for tree recursion.

    The original recurses PER-RESULT: each result's research_goal + follow-up
    questions become the query for a new recursive call at depth-1 with
    breadth = max(2, breadth // 2).

    This node pushes new branch items ON TOP of the existing stack, ensuring
    DFS ordering: sub-branches are fully explored before sibling branches
    resume, exactly matching the original's call-stack recursion.
    """
    results = state.get("research_results", [])
    current_depth = state["current_depth"]
    new_depth = current_depth - 1

    # Build branch items for each result in order (first result at front = processed first)
    new_items = []
    for r in results:
        goal = r.get("research_goal", "")
        follow_ups = " ".join(r.get("follow_up_questions", []))
        branch_query = (
            f"Previous research goal: {goal}\n"
            f"Follow-up questions: {follow_ups}"
        )
        new_items.append(BranchItem(query=branch_query, depth=new_depth))

    # Push new items on top of existing stack (DFS order)
    existing_stack = list(state.get("branch_stack", []))
    updated_stack = new_items + existing_stack

    # Update progress for depth transition
    progress = _get_progress(config)
    if progress:
        progress.current_depth = state["depth"] - new_depth

    return {
        "branch_stack": updated_stack,
        "research_results": [],
    }


# ---------------------------------------------------------------------------
# Node: pick_next_branch
# ---------------------------------------------------------------------------

async def pick_next_branch(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Pop the top branch from the stack and set it as the current query/depth."""
    stack = list(state.get("branch_stack", []))

    if stack:
        item = stack.pop(0)  # Pop from top (front)
        return {
            "combined_query": item["query"],
            "current_depth": item["depth"],
            "branch_stack": stack,
        }

    return {"combined_query": "", "current_depth": 0, "branch_stack": []}


# ---------------------------------------------------------------------------
# Node: assemble_final_context
# ---------------------------------------------------------------------------

async def assemble_final_context(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Assemble and trim the final research context from all accumulated data.

    Mirrors original DeepResearchSkill.run() final context assembly:
      - Deduplicate learnings
      - Attach citations as "[Source: url]" suffixes
      - Extend with full research context
      - Trim to MAX_CONTEXT_WORDS (25,000 words)
    """
    all_learnings = list(set(state.get("all_learnings", [])))
    all_citations = state.get("all_citations", {})
    all_context = state.get("all_context", [])

    context_with_citations = []
    for learning in all_learnings:
        citation = all_citations.get(learning, "")
        if citation:
            context_with_citations.append(f"{learning} [Source: {citation}]")
        else:
            context_with_citations.append(learning)

    context_with_citations.extend(all_context)

    final_context_list = _trim_context(context_with_citations)
    final_context = "\n".join(final_context_list)

    logger.info(
        f"Assembled final context: {len(final_context_list)} items, "
        f"{_count_words(final_context)} words"
    )

    return {
        "final_context": final_context,
    }


# ---------------------------------------------------------------------------
# Node: generate_report
# ---------------------------------------------------------------------------

async def generate_report(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate the final research report from assembled context."""
    final_context = state.get("final_context", "")
    if not final_context.strip():
        raise RuntimeError(
            "Cannot generate report: no research context was assembled. "
            "This likely means all research queries produced no usable results."
        )

    llm_service = _get_llm_service(config)
    smart_llm = llm_service.get_smart_llm(temperature=0.4, max_tokens=4000)

    messages = GENERATE_REPORT_PROMPT.format_messages(
        query=state["query"],
        context=final_context,
        tone=state.get("tone", "Objective"),
    )

    report = await llm_service.invoke(smart_llm, messages)

    if not report or not report.strip():
        raise RuntimeError("Report generation returned empty content from the LLM.")

    return {"report": report, "messages": [AIMessage(content=report)]}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_continue_deeper(state: DeepResearchState) -> str:
    """After execute_research: decide whether to branch deeper or route to stack check."""
    current_depth = state.get("current_depth", 0)
    if current_depth > 1:
        return "go_deeper"
    return "check_stack"


def has_more_work(state: DeepResearchState) -> str:
    """After fan_out or when depth is exhausted: check if branches remain on the stack."""
    stack = state.get("branch_stack", [])
    if stack:
        return "next_branch"
    return "done"
