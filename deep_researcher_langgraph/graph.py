"""LangGraph StateGraph definition for the deep research workflow.

Level-by-level parallel execution model:

                generate_research_plan
                         |
                         v
          +---> generate_search_queries
          |              |
          |              v
          |      execute_research
          |              |
          |    should_continue_deeper?
          |       /              \\
          |  "go_deeper"       "done"
          |      |                 |
          |      v                 v
          |  fan_out_branches  assemble_final_context
          |      |                 |
          +------+                 v
                             generate_report
                                   |
                                   v
                                  END

All branches at the same depth level are collected into pending_branches and
processed concurrently in generate_search_queries / execute_research before
moving to the next level.
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
    assemble_final_context,
    generate_report,
    should_continue_deeper,
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
    # Default: no checkpointer. MemorySaver uses msgpack serialization which
    # fails on non-serializable state values (e.g. functions from GPTResearcher).
    # Pass an explicit checkpointer only when persistence is needed.

    workflow = StateGraph(DeepResearchState)

    # --- Nodes (6) ---
    workflow.add_node("generate_research_plan", generate_research_plan)
    workflow.add_node("generate_search_queries", generate_search_queries)
    workflow.add_node("execute_research", execute_research)
    workflow.add_node("fan_out_branches", fan_out_branches)
    workflow.add_node("assemble_final_context", assemble_final_context)
    workflow.add_node("generate_report", generate_report)

    # --- Entry ---
    workflow.set_entry_point("generate_research_plan")

    # --- Linear: plan → queries → research ---
    workflow.add_edge("generate_research_plan", "generate_search_queries")
    workflow.add_edge("generate_search_queries", "execute_research")

    # --- Branch or finish ---
    workflow.add_conditional_edges(
        "execute_research",
        should_continue_deeper,
        {"go_deeper": "fan_out_branches", "done": "assemble_final_context"},
    )

    # --- Level loop: fan_out always returns to generate_search_queries ---
    workflow.add_edge("fan_out_branches", "generate_search_queries")

    # --- Final ---
    workflow.add_edge("assemble_final_context", "generate_report")
    workflow.add_edge("generate_report", END)

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    return workflow.compile(**compile_kwargs)
