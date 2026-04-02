"""LangGraph state definitions for deep research workflow."""

from typing import TypedDict, List, Dict, Optional, Annotated
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _merge_dicts(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    return {**a, **b}


def _merge_ordered_unique(a: List[str], b: List[str]) -> List[str]:
    """Merge two lists preserving insertion order and removing duplicates."""
    return list(dict.fromkeys([*a, *b]))


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


class TreeNode(TypedDict):
    """A single node in the research tree, carrying its position and findings."""
    path: str              # "0", "0.1", "1.0.2" etc.
    depth_level: int       # which depth this was researched at (3=shallowest, 1=deepest)
    topic: str             # research_goal — becomes section heading
    learnings: List[str]
    citations: Dict[str, str]
    context: str


class BranchItem(TypedDict):
    """A single branch item: a query to research at a given depth with tree position."""
    query: str
    depth: int
    path: str            # tree path e.g. "0", "0.1", "1.0.2"
    parent_topic: str    # research_goal of parent, becomes section heading


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
    mcp_configs: Optional[list]
    mcp_strategy: Optional[str]

    # --- Research plan ---
    initial_search_results: List[dict]
    follow_up_questions: List[str]
    combined_query: str

    # --- Current level ---
    current_depth: int
    current_breadth: int
    search_queries: List[SearchQuery]
    research_results: List[ResearchResult]

    # --- Pending branches for parallel execution ---
    pending_branches: List[BranchItem]

    # --- Accumulated across levels ---
    research_tree: Annotated[List[TreeNode], operator.add]
    all_learnings: Annotated[List[str], operator.add]
    all_citations: Annotated[Dict[str, str], _merge_dicts]
    all_visited_urls: Annotated[List[str], _merge_ordered_unique]
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
