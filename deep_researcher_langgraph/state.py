"""LangGraph state definitions for deep research workflow."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated, Callable
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _merge_dicts(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    return {**a, **b}


class SearchQuery(TypedDict):
    """A single search query with its research goal."""
    query: str
    research_goal: str


class ResearchResult(TypedDict):
    """Result from processing a single research query."""
    query: str
    learnings: List[str]
    follow_up_questions: List[str]
    citations: Dict[str, str]
    context: str
    sources: List[dict]
    visited_urls: List[str]
    research_goal: str


class BranchItem(TypedDict):
    """A single branch on the DFS stack: a query to research at a given depth."""
    query: str
    depth: int


class ResearchProgress:
    """Tracks progress of the deep research process."""

    def __init__(self, total_depth: int, total_breadth: int):
        self.current_depth: int = 1
        self.total_depth: int = total_depth
        self.current_breadth: int = 0
        self.total_breadth: int = total_breadth
        self.current_query: Optional[str] = None
        self.total_queries: int = 0
        self.completed_queries: int = 0


class DeepResearchState(TypedDict):
    # --- Input ---
    query: str
    breadth: int
    depth: int
    concurrency_limit: int
    tone: str
    config_path: Optional[str]
    headers: dict
    websocket: Any
    mcp_configs: Optional[list]
    mcp_strategy: Optional[str]
    on_progress: Optional[Callable]

    # --- Research plan ---
    initial_search_results: List[dict]
    follow_up_questions: List[str]
    combined_query: str

    # --- Current level ---
    current_depth: int
    current_breadth: int
    search_queries: List[SearchQuery]
    research_results: List[ResearchResult]

    # --- Branch stack for tree recursion ---
    # Stack of branch items. Each item holds a query and the depth at which
    # it should be researched.  fan_out_branches pushes new items;
    # pick_next_branch pops from the top.  This stack-based approach
    # correctly models the original's recursive call-stack semantics so that
    # sub-branches at depth N-1 are fully processed before sibling branches
    # at depth N resume.
    branch_stack: List[BranchItem]

    # --- Accumulated across levels ---
    all_learnings: Annotated[List[str], operator.add]
    all_citations: Annotated[Dict[str, str], _merge_dicts]
    all_visited_urls: Annotated[List[str], operator.add]
    all_context: Annotated[List[str], operator.add]
    all_sources: Annotated[List[dict], operator.add]

    # --- Progress ---
    total_queries: int
    completed_queries: int

    # --- Final output ---
    final_context: str
    report: str

    # --- Conversation history (checkpointer persistence) ---
    messages: Annotated[List[BaseMessage], add_messages]
