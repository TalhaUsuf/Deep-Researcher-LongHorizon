"""LangGraph node functions for the deep research workflow.

Each node takes a DeepResearchState and returns a partial state update dict.
All LLM calls go through the shared LLMService instance stored in the graph config.

Level-by-level parallel execution model:
  All branches at the same depth level are researched concurrently.
  fan_out_branches collects all next-level branches into pending_branches.
  generate_search_queries expands all pending branches in parallel.
  execute_research runs all queries at the current level concurrently.
  Results are stored as TreeNode entries in research_tree for hierarchical rendering.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

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
from .state import BranchItem, DeepResearchState, ResearchProgress, TreeNode

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
    """Trim context list to stay within word limit, keeping earliest items first.

    Learnings with citations appear at the front of the list (from
    assemble_final_context), so forward iteration preserves curated
    insights over verbose raw scrape text.
    """
    total_words = 0
    trimmed = []
    for item in context_list:
        words = _count_words(item)
        if total_words + words <= max_words:
            trimmed.append(item)
            total_words += words
        else:
            break
    return trimmed


def _notify_progress(state: DeepResearchState, config: dict) -> None:
    """Fire the on_progress callback if provided."""
    on_progress = config.get("configurable", {}).get("on_progress")
    progress = _get_progress(config)
    if on_progress and progress:
        try:
            on_progress(progress)
        except Exception:
            logger.warning("Progress callback raised an exception", exc_info=True)


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

    # Truncate query for retriever search (Tavily has a 400 char limit).
    # The full query is still used for LLM prompts.
    search_query = query[:400] if len(query) > 400 else query

    # Run initial search across all configured retrievers (matches original)
    retrievers = get_retrievers(state.get("headers", {}), cfg)
    all_search_results: List[dict] = []
    for retriever_class in retrievers:
        try:
            results = await get_search_results(search_query, retriever_class)
            all_search_results.extend(results)
        except Exception as e:
            logger.warning(f"Error with retriever {retriever_class.__name__}: {e}")

    if not all_search_results and retrievers:
        logger.warning(
            f"All {len(retrievers)} retrievers failed during initial search for query: {query!r}. "
            "Proceeding with query-only research plan (graceful degradation)."
        )

    logger.info(f"Initial web knowledge obtained: {len(all_search_results)} results")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    messages = GENERATE_RESEARCH_PLAN_PROMPT.format_messages(
        query=query,
        current_time=current_time,
        search_results=str(all_search_results),
        num_questions=breadth,
    )

    # Use smart LLM for structured output — reasoning models (e.g. qwen3-235b)
    # consume thinking tokens that exhaust max_tokens before producing JSON.
    smart_llm = llm_service.get_smart_llm(temperature=0.4, max_tokens=4000)
    response: FollowUpQuestionsResponse = await llm_service.invoke_structured(
        smart_llm, FollowUpQuestionsResponse, messages
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
        "all_context": [],
        "all_sources": [],
        "pending_branches": [],
        "research_tree": [],
        "total_queries": 0,
        "completed_queries": 0,
        "messages": [HumanMessage(content=query)],
    }


# ---------------------------------------------------------------------------
# Node: generate_search_queries
# ---------------------------------------------------------------------------

async def generate_search_queries(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate diverse search queries for the current research level.

    Two cases:
    - First call (pending_branches empty): generate from combined_query, assign root
      paths "0", "1", "2", ...
    - Subsequent calls (pending_branches non-empty): generate queries for EACH branch
      concurrently using a local semaphore, assigning child paths "0.0", "0.1", "1.0", ...
    """
    llm_service = _get_llm_service(config)
    current_depth = state["current_depth"]
    original_breadth = state["breadth"]

    # Calculate breadth for current level (halves each level, min 2)
    levels_deep = state["depth"] - current_depth
    breadth = original_breadth
    for _ in range(levels_deep):
        breadth = max(2, breadth // 2)

    pending = state.get("pending_branches", [])
    # Use smart LLM for structured output (reasoning models exhaust tokens on thinking)
    smart_llm = llm_service.get_smart_llm(temperature=0.4, max_tokens=4000)

    if not pending and current_depth == state["depth"]:
        # First call (root level only): generate from combined_query with root paths
        query = state.get("combined_query") or state["query"]
        messages = GENERATE_SEARCH_QUERIES_PROMPT.format_messages(
            query=query,
            num_queries=breadth,
        )
        response: SearchQueriesResponse = await llm_service.invoke_structured(
            smart_llm, SearchQueriesResponse, messages
        )
        search_queries = []
        for i, q in enumerate(response.queries[:breadth]):
            search_queries.append({
                "query": q.query,
                "research_goal": q.research_goal,
                "path": str(i),
                "parent_topic": state["query"],
            })
    else:
        # Subsequent calls: generate for each pending branch concurrently
        concurrency_limit = state.get("concurrency_limit", 2)
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def gen_for_branch(branch: BranchItem) -> List[dict]:
            async with semaphore:
                branch_messages = GENERATE_SEARCH_QUERIES_PROMPT.format_messages(
                    query=branch["query"],
                    num_queries=breadth,
                )
                branch_response = await llm_service.invoke_structured(
                    smart_llm, SearchQueriesResponse, branch_messages
                )
                queries = []
                for j, q in enumerate(branch_response.queries[:breadth]):
                    queries.append({
                        "query": q.query,
                        "research_goal": q.research_goal,
                        "path": f"{branch['path']}.{j}",
                        "parent_topic": branch["parent_topic"],
                    })
                return queries

        tasks = [gen_for_branch(b) for b in pending]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        search_queries = [
            q
            for batch in batch_results
            if not isinstance(batch, Exception)
            for q in batch
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
      - For each query, spawns a GPTResearcher instance with its own visited_urls copy
      - Extracts learnings via strategic LLM (temperature=0.4, ReasoningEfforts.High)
      - Builds TreeNode entries for hierarchical context assembly
      - Handles per-query errors gracefully (continues with successful results)
    """
    llm_service = _get_llm_service(config)
    cfg = _get_config(config)
    search_queries = state["search_queries"]
    semaphore = asyncio.Semaphore(state.get("concurrency_limit", 2))

    async def _process_single_query(serp_query: dict) -> dict | None:
        async with semaphore:
            try:
                # Update progress
                progress = _get_progress(config)
                if progress:
                    progress.current_query = serp_query["query"]
                _notify_progress(state, config)

                # Each researcher gets its own copy of visited_urls (not shared ref)
                researcher = GPTResearcher(
                    query=serp_query["query"],
                    report_type=ReportType.ResearchReport.value,
                    report_source=ReportSource.Web.value,
                    tone=state.get("tone", "Objective"),
                    websocket=config.get("configurable", {}).get("websocket"),
                    config_path=state.get("config_path"),
                    headers=state.get("headers", {}),
                    visited_urls=set(state.get("all_visited_urls", [])),
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

                # Use smart LLM for structured output (reasoning models exhaust
                # tokens on thinking before producing JSON)
                analysis_llm = llm_service.get_smart_llm(
                    temperature=0.4,
                    max_tokens=cfg.strategic_token_limit,
                )
                analysis: ResearchAnalysis = await llm_service.invoke_structured(
                    analysis_llm, ResearchAnalysis, messages
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
                    "path": serp_query.get("path", ""),
                    "parent_topic": serp_query.get("parent_topic", ""),
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
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in raw_results if r is not None and not isinstance(r, Exception)]

    if not results and search_queries:
        raise RuntimeError(
            f"All {len(search_queries)} research queries failed. "
            "Check logs for individual query errors."
        )

    # Aggregate results and build tree nodes
    new_learnings = []
    new_citations = {}
    new_visited = []
    new_context = []
    new_sources = []
    tree_nodes: List[TreeNode] = []

    for r in results:
        new_learnings.extend(r["learnings"])
        new_citations.update(r["citations"])
        new_visited.extend(r["visited_urls"])
        if r["context"]:
            new_context.append(r["context"])
        new_sources.extend(r["sources"])
        tree_nodes.append(TreeNode(
            path=r.get("path", ""),
            depth_level=state["current_depth"],
            topic=r.get("research_goal", ""),
            learnings=r["learnings"],
            citations=r["citations"],
            context=r["context"],
        ))

    return {
        "research_results": results,
        "all_learnings": new_learnings,
        "all_citations": new_citations,
        "all_visited_urls": new_visited,
        "all_context": new_context,
        "all_sources": new_sources,
        "research_tree": tree_nodes,
        "completed_queries": state.get("completed_queries", 0) + len(results),
        "current_breadth": len(results),
    }


# ---------------------------------------------------------------------------
# Node: fan_out_branches
# ---------------------------------------------------------------------------

async def fan_out_branches(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Collect next-level branch queries from research results for parallel processing.

    Children inherit the path and research_goal of their parent result so that
    generate_search_queries can assign child paths like "0.0", "0.1", "1.0", etc.
    """
    results = state.get("research_results", [])
    current_depth = state["current_depth"]
    new_depth = current_depth - 1

    new_branches: List[BranchItem] = []
    for r in results:
        goal = r.get("research_goal", "")
        follow_ups = " ".join(r.get("follow_up_questions", []))
        branch_query = (
            f"Previous research goal: {goal}\n"
            f"Follow-up questions: {follow_ups}"
        )
        new_branches.append(BranchItem(
            query=branch_query,
            depth=new_depth,
            path=r["path"],
            parent_topic=r["research_goal"],
        ))

    # Update progress for depth transition
    progress = _get_progress(config)
    if progress:
        progress.current_depth = state["depth"] - new_depth

    return {
        "pending_branches": new_branches,
        "current_depth": new_depth,
        "research_results": [],
    }


# ---------------------------------------------------------------------------
# Node: assemble_final_context
# ---------------------------------------------------------------------------

async def assemble_final_context(state: DeepResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Assemble and trim the final research context from the research tree.

    Renders hierarchical sections using heading levels derived from tree depth,
    deduplicates with stable ordering, and trims to MAX_CONTEXT_WORDS.
    """
    tree = state.get("research_tree", [])
    all_context = state.get("all_context", [])
    max_depth = state.get("depth", 1)

    # Sort tree nodes by path for hierarchical rendering
    sorted_tree = sorted(
        tree,
        key=lambda n: tuple(int(x) for x in n["path"].split(".")) if n["path"] else (),
    )

    context_parts: List[str] = []
    for node in sorted_tree:
        depth_level = node.get("depth_level", 1)
        heading_level = max_depth - depth_level + 1
        heading_prefix = "#" * (heading_level + 1)
        topic = node.get("topic", "")
        if topic:
            context_parts.append(f"{heading_prefix} {topic}")

        for learning in node.get("learnings", []):
            citation = node["citations"].get(learning, "")
            if citation:
                context_parts.append(f"- {learning} [Source: {citation}]")
            else:
                context_parts.append(f"- {learning}")

        if node.get("context"):
            context_parts.append(node["context"])

    # Add any raw context not captured in tree nodes
    context_parts.extend(all_context)

    # Stable deduplication preserving insertion order
    deduped = list(dict.fromkeys(context_parts))

    final_context_list = _trim_context(deduped)
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
    cfg = _get_config(config)
    # Use strategic LLM for report generation (more capable, avoids empty responses)
    report_llm = llm_service.get_strategic_llm(
        temperature=0.4, max_tokens=cfg.strategic_token_limit,
    )

    messages = GENERATE_REPORT_PROMPT.format_messages(
        query=state["query"],
        context=final_context,
        tone=state.get("tone", "Objective"),
        current_date=datetime.now().strftime("%B %d, %Y"),
    )

    # Retry up to 2 times if LLM returns empty content
    report = ""
    for attempt in range(3):
        report = await llm_service.invoke(report_llm, messages)
        if report and report.strip():
            break
        logger.warning(
            "Report generation attempt %d returned empty content; retrying...",
            attempt + 1,
        )

    if not report or not report.strip():
        raise RuntimeError("Report generation returned empty content from the LLM after 3 attempts.")

    return {"report": report, "messages": [AIMessage(content=report)]}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_continue_deeper(state: DeepResearchState) -> str:
    """After execute_research: decide whether to branch deeper or finish."""
    current_depth = state.get("current_depth", 0)
    results = state.get("research_results", [])
    if current_depth > 1 and results:
        return "go_deeper"
    return "done"
