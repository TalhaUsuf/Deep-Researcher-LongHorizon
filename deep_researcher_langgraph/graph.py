"""LangGraph StateGraph definition for the deep research workflow.

The graph models the original's tree recursion using a branch stack:

                generate_research_plan
                         |
                         v
          +---> generate_search_queries
          |              |
          |              v
          |      execute_research
          |              |
          |              v
          |    should_continue_deeper?
          |       /              \
          |  "go_deeper"      "check_stack"
          |      |                 \
          |      v                  \
          |  fan_out_branches        |
          |      |                   |
          |      v                   |
          |  has_more_work?          |
          |    /       \             |
          | "next"   "done"          |
          |   |         \            |
          |   |          +------+    |
          |   |                 |    |
          |   |                 v    v
          |   |        assemble_final_context
          |   |                 |
          |   |                 v
          |   |           has_more_work?
          |   |            /         \
          |   |        "next"      "done"
          |   |          |            |
          |   v          v            v
          | pick_next_branch    generate_report
          |      |                    |
          +------+                    v
                                     END

The branch_stack ensures DFS ordering: when a sub-branch fans out,
its children are pushed ON TOP of remaining sibling branches, so they
are processed first — exactly matching the original's recursive call stack.
"""

from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from .state import DeepResearchState
from .nodes import (
    generate_research_plan,
    generate_search_queries,
    execute_research,
    fan_out_branches,
    pick_next_branch,
    assemble_final_context,
    generate_report,
    should_continue_deeper,
    has_more_work,
)


def build_deep_research_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> StateGraph:
    """Construct and return the compiled deep research LangGraph workflow.

    Args:
        checkpointer: Checkpoint saver for state persistence. Pass ``None``
            to use the default in-memory checkpointer (``MemorySaver``), or
            supply a custom backend (SQLite, Postgres, etc.).
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    workflow = StateGraph(DeepResearchState)

    # --- Register nodes ---
    workflow.add_node("generate_research_plan", generate_research_plan)
    workflow.add_node("generate_search_queries", generate_search_queries)
    workflow.add_node("execute_research", execute_research)
    workflow.add_node("fan_out_branches", fan_out_branches)
    workflow.add_node("pick_next_branch", pick_next_branch)
    workflow.add_node("assemble_final_context", assemble_final_context)
    workflow.add_node("generate_report", generate_report)

    # --- Entry ---
    workflow.set_entry_point("generate_research_plan")

    # --- Linear flow: plan → queries → research ---
    workflow.add_edge("generate_research_plan", "generate_search_queries")
    workflow.add_edge("generate_search_queries", "execute_research")

    # --- After research: deeper or check stack? ---
    workflow.add_conditional_edges(
        "execute_research",
        should_continue_deeper,
        {
            "go_deeper": "fan_out_branches",
            "check_stack": "assemble_final_context",
        },
    )

    # --- fan_out pushes branches, then check if stack has work ---
    workflow.add_conditional_edges(
        "fan_out_branches",
        has_more_work,
        {
            "next_branch": "pick_next_branch",
            "done": "assemble_final_context",
        },
    )

    # --- pick_next_branch routes back to generate_search_queries ---
    workflow.add_edge("pick_next_branch", "generate_search_queries")

    # --- assemble checks stack: more branches or report ---
    workflow.add_conditional_edges(
        "assemble_final_context",
        has_more_work,
        {
            "next_branch": "pick_next_branch",
            "done": "generate_report",
        },
    )

    # --- Final ---
    workflow.add_edge("generate_report", END)

    return workflow.compile(checkpointer=checkpointer)
