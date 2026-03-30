"""Entry point for running the LangGraph deep research workflow.

Usage:
    from deep_researcher_langgraph import run_deep_research

    result = await run_deep_research(
        query="Your research question",
        breadth=4,
        depth=2,
    )
    print(result["report"])
"""

import asyncio
import logging
import time
import uuid
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Set

from langgraph.checkpoint.base import BaseCheckpointSaver

from gpt_researcher.config.config import Config

from .callbacks import TokenUsageCallbackHandler
from .graph import build_deep_research_graph
from .llm_service import LLMService
from .state import DeepResearchState, ResearchProgress

logger = logging.getLogger(__name__)


async def run_deep_research(
    query: str,
    breadth: Optional[int] = None,
    depth: Optional[int] = None,
    concurrency_limit: Optional[int] = None,
    tone: str = "Objective",
    config_path: Optional[str] = None,
    headers: Optional[dict] = None,
    websocket: Any = None,
    visited_urls: Optional[Set[str]] = None,
    mcp_configs: Optional[list] = None,
    mcp_strategy: Optional[str] = None,
    on_progress: Optional[Callable] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> Dict[str, Any]:
    """Run the full deep research pipeline using LangGraph.

    Args:
        query: The research question to investigate.
        breadth: Number of search queries per level (default from config, original default 4).
        depth: Number of recursive depth levels (default from config, original default 2).
        concurrency_limit: Max concurrent research tasks (default from config, original default 2).
        tone: Report tone (e.g. "Objective", "Analytical").
        config_path: Path to JSON config file override.
        headers: HTTP headers for retriever selection.
        websocket: WebSocket for streaming progress.
        visited_urls: Set of URLs already visited (for deduplication).
        mcp_configs: MCP server configurations.
        mcp_strategy: MCP strategy ("fast", "deep", "disabled").
        on_progress: Callback receiving ResearchProgress updates (mirrors original).
        thread_id: Unique thread identifier for checkpointer persistence.
            Auto-generated if not provided.
        checkpointer: Custom checkpoint saver. Passed through to
            ``build_deep_research_graph``. Defaults to ``MemorySaver`` when
            ``None``.

    Returns:
        Dict with keys: "report", "final_context", "visited_urls", "sources",
        "learnings", "citations", "usage_summary", "thread_id".
    """
    start_time = time.time()

    cfg = Config(config_path)
    # Match original defaults: breadth=4, depth=2, concurrency=2
    breadth = breadth or getattr(cfg, "deep_research_breadth", 4)
    depth = depth or getattr(cfg, "deep_research_depth", 2)
    concurrency_limit = concurrency_limit or getattr(cfg, "deep_research_concurrency", 2)

    logger.info(
        f"Starting LangGraph deep research: breadth={breadth}, depth={depth}, "
        f"concurrency={concurrency_limit}"
    )

    # Build shared LLM service with token tracking
    callback_handler = TokenUsageCallbackHandler()
    llm_service = LLMService(cfg, callback_handler)

    # Build progress tracker (mirrors original ResearchProgress)
    progress = ResearchProgress(total_depth=depth, total_breadth=breadth)

    # Build the graph (with checkpointer for persistence)
    graph = build_deep_research_graph(checkpointer=checkpointer)

    # Ensure a thread_id for checkpointer tracking
    thread_id = thread_id or str(uuid.uuid4())

    # Prepare initial state
    initial_state: DeepResearchState = {
        "query": query,
        "breadth": breadth,
        "depth": depth,
        "concurrency_limit": concurrency_limit,
        "tone": tone,
        "config_path": config_path,
        "headers": headers or {},
        "websocket": websocket,
        "mcp_configs": mcp_configs,
        "mcp_strategy": mcp_strategy,
        "on_progress": on_progress,
        # Initialised by nodes
        "initial_search_results": [],
        "follow_up_questions": [],
        "combined_query": "",
        "current_depth": depth,
        "current_breadth": 0,
        "search_queries": [],
        "research_results": [],
        "branch_stack": [],
        "all_learnings": [],
        "all_citations": {},
        "all_visited_urls": list(visited_urls) if visited_urls else [],
        "all_context": [],
        "all_sources": [],
        "total_queries": 0,
        "completed_queries": 0,
        "final_context": "",
        "report": "",
        "messages": [],
    }

    # Run the graph with thread_id for checkpointer persistence
    try:
        result = await graph.ainvoke(
            initial_state,
            config={
                "configurable": {
                    "llm_service": llm_service,
                    "progress": progress,
                    "config": cfg,
                    "thread_id": thread_id,
                }
            },
        )
    except Exception as e:
        logger.error(f"Deep research workflow failed: {e}", exc_info=True)
        raise RuntimeError(
            f"Deep research failed during execution: {e}. "
            f"Completed {progress.completed_queries}/{progress.total_queries} queries."
        ) from e

    elapsed = timedelta(seconds=time.time() - start_time)
    usage_summary = llm_service.get_usage_summary()

    logger.info(f"Deep research completed in {elapsed}")
    logger.info(f"Token usage: {usage_summary}")

    return {
        "report": result.get("report", ""),
        "final_context": result.get("final_context", ""),
        "visited_urls": result.get("all_visited_urls", []),
        "sources": result.get("all_sources", []),
        "learnings": list(set(result.get("all_learnings", []))),
        "citations": result.get("all_citations", {}),
        "usage_summary": usage_summary,
        "thread_id": thread_id,
    }


def main():
    """CLI entry point for standalone deep research."""
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph Deep Researcher")
    parser.add_argument("query", help="Research question")
    parser.add_argument("--breadth", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--tone", default="Objective")
    parser.add_argument("--config-path", default=None)
    args = parser.parse_args()

    import sys
    logging.basicConfig(level=logging.INFO)

    try:
        result = asyncio.run(
            run_deep_research(
                query=args.query,
                breadth=args.breadth,
                depth=args.depth,
                concurrency_limit=args.concurrency,
                tone=args.tone,
                config_path=args.config_path,
            )
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 80)
    print("RESEARCH REPORT")
    print("=" * 80)
    print(result["report"])
    print("\n" + "=" * 80)
    print("TOKEN USAGE")
    print("=" * 80)
    print(result["usage_summary"])


if __name__ == "__main__":
    main()
