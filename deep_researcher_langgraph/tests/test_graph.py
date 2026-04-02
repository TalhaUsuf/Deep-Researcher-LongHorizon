"""Tests for LangGraph deep research graph structure and flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver

from deep_researcher_langgraph.graph import build_deep_research_graph
from deep_researcher_langgraph.state import DeepResearchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> DeepResearchState:
    """Return a minimal valid DeepResearchState with optional overrides."""
    state = {
        "query": "test query",
        "breadth": 4,
        "depth": 2,
        "concurrency_limit": 2,
        "tone": "Objective",
        "config_path": None,
        "headers": {},
        "mcp_configs": None,
        "mcp_strategy": None,
        "initial_search_results": [],
        "follow_up_questions": [],
        "combined_query": "",
        "current_depth": 2,
        "current_breadth": 0,
        "search_queries": [],
        "research_results": [],
        "pending_branches": [],
        "research_tree": [],
        "all_learnings": [],
        "all_citations": {},
        "all_visited_urls": [],
        "all_context": [],
        "all_sources": [],
        "total_queries": 0,
        "completed_queries": 0,
        "final_context": "",
        "report": "",
        "messages": [],
    }
    state.update(overrides)
    return state


_THREAD_CONFIG = {"configurable": {"thread_id": "test-thread"}}


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestGraphStructure:
    """Tests for the compiled graph object returned by build_deep_research_graph."""

    def test_build_returns_compiled_graph(self):
        graph = build_deep_research_graph()
        assert graph is not None
        assert hasattr(graph, "ainvoke"), "Compiled graph must expose ainvoke"

    def test_build_accepts_custom_checkpointer(self):
        custom_checkpointer = MemorySaver()
        graph = build_deep_research_graph(checkpointer=custom_checkpointer)
        assert graph.checkpointer is custom_checkpointer

    def test_graph_has_correct_nodes(self):
        graph = build_deep_research_graph()
        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "generate_research_plan",
            "generate_search_queries",
            "execute_research",
            "fan_out_branches",
            "assemble_final_context",
            "generate_report",
        }
        # LangGraph also adds __start__ and __end__ pseudo-nodes
        assert expected_nodes.issubset(node_names), (
            f"Missing nodes: {expected_nodes - node_names}"
        )

    def test_graph_does_not_have_removed_nodes(self):
        graph = build_deep_research_graph()
        node_names = set(graph.nodes.keys())
        assert "pick_next_branch" not in node_names
        assert "has_more_work" not in node_names


# ---------------------------------------------------------------------------
# Flow tests (all node functions mocked)
# ---------------------------------------------------------------------------

_PATCH_BASE = "deep_researcher_langgraph.graph"


def _make_node_tracker(call_order: list, node_name: str, return_value: dict):
    """Return an async callable that records its call and returns return_value."""
    async def _node(state, config=None):
        call_order.append(node_name)
        return return_value

    return _node


class TestDepthOneFlow:
    """Depth=1: should_continue_deeper returns 'done' immediately.
    Expected order: plan → queries → research → assemble → report
    """

    @pytest.mark.asyncio
    async def test_depth1_node_order(self):
        call_order = []

        with (
            patch(f"{_PATCH_BASE}.generate_research_plan",
                  new=_make_node_tracker(call_order, "generate_research_plan", {
                      "initial_search_results": [],
                      "follow_up_questions": ["q1"],
                      "combined_query": "combined",
                      "current_depth": 1,
                      "current_breadth": 0,
                      "all_learnings": [],
                      "all_citations": {},
                      "all_visited_urls": [],
                      "all_context": [],
                      "all_sources": [],
                      "pending_branches": [],
                      "research_tree": [],
                      "total_queries": 0,
                      "completed_queries": 0,
                  })),
            patch(f"{_PATCH_BASE}.generate_search_queries",
                  new=_make_node_tracker(call_order, "generate_search_queries", {
                      "search_queries": [{"query": "sq1", "research_goal": "goal1"}],
                      "total_queries": 1,
                  })),
            patch(f"{_PATCH_BASE}.execute_research",
                  new=_make_node_tracker(call_order, "execute_research", {
                      "research_results": [{"query": "sq1", "learnings": ["l1"],
                                            "follow_up_questions": [], "citations": {},
                                            "context": "ctx", "sources": [],
                                            "visited_urls": [], "research_goal": "goal1",
                                            "path": "0", "parent_topic": "test query"}],
                      "all_learnings": ["l1"],
                      "all_citations": {},
                      "all_visited_urls": [],
                      "all_context": ["ctx"],
                      "all_sources": [],
                      "research_tree": [],
                      "completed_queries": 1,
                      "current_breadth": 1,
                  })),
            patch(f"{_PATCH_BASE}.fan_out_branches",
                  new=_make_node_tracker(call_order, "fan_out_branches", {})),
            patch(f"{_PATCH_BASE}.assemble_final_context",
                  new=_make_node_tracker(call_order, "assemble_final_context", {
                      "final_context": "assembled context",
                  })),
            patch(f"{_PATCH_BASE}.generate_report",
                  new=_make_node_tracker(call_order, "generate_report", {
                      "report": "Final report",
                  })),
            patch(f"{_PATCH_BASE}.should_continue_deeper", return_value="done"),
        ):
            graph = build_deep_research_graph()
            initial = _base_state(depth=1, current_depth=1)
            result = await graph.ainvoke(initial, config=_THREAD_CONFIG)

        assert call_order == [
            "generate_research_plan",
            "generate_search_queries",
            "execute_research",
            "assemble_final_context",
            "generate_report",
        ]
        assert result["report"] == "Final report"


class TestDepthTwoFlowWithBranching:
    """Depth=2: first execute_research triggers 'go_deeper', fan_out collects
    branches into pending_branches, then generate_search_queries/execute_research
    run again at depth=1, returning 'done', then assemble and report.

    Expected order:
        plan → queries → research(depth=2, go_deeper) → fan_out
        → queries → research(depth=1, done) → assemble → report
    """

    @pytest.mark.asyncio
    async def test_depth2_branching_flow(self):
        call_order = []
        deeper_calls = {"n": 0}

        def _should_continue_deeper(state):
            deeper_calls["n"] += 1
            return "go_deeper" if deeper_calls["n"] == 1 else "done"

        with (
            patch(f"{_PATCH_BASE}.generate_research_plan",
                  new=_make_node_tracker(call_order, "generate_research_plan", {
                      "initial_search_results": [],
                      "follow_up_questions": ["q1"],
                      "combined_query": "combined",
                      "current_depth": 2,
                      "current_breadth": 0,
                      "all_learnings": [],
                      "all_citations": {},
                      "all_visited_urls": [],
                      "all_context": [],
                      "all_sources": [],
                      "pending_branches": [],
                      "research_tree": [],
                      "total_queries": 0,
                      "completed_queries": 0,
                  })),
            patch(f"{_PATCH_BASE}.generate_search_queries",
                  new=_make_node_tracker(call_order, "generate_search_queries", {
                      "search_queries": [{"query": "sq", "research_goal": "g",
                                          "path": "0", "parent_topic": "test query"}],
                      "total_queries": 1,
                  })),
            patch(f"{_PATCH_BASE}.execute_research",
                  new=_make_node_tracker(call_order, "execute_research", {
                      "research_results": [{"query": "sq", "learnings": ["l1"],
                                            "follow_up_questions": ["fu"],
                                            "citations": {}, "context": "ctx",
                                            "sources": [], "visited_urls": [],
                                            "research_goal": "g",
                                            "path": "0", "parent_topic": "test query"}],
                      "all_learnings": ["l1"],
                      "all_citations": {},
                      "all_visited_urls": [],
                      "all_context": ["ctx"],
                      "all_sources": [],
                      "research_tree": [],
                      "completed_queries": 1,
                      "current_breadth": 1,
                  })),
            patch(f"{_PATCH_BASE}.fan_out_branches",
                  new=_make_node_tracker(call_order, "fan_out_branches", {
                      "pending_branches": [{"query": "branch1", "depth": 1,
                                            "path": "0", "parent_topic": "g"}],
                      "current_depth": 1,
                      "research_results": [],
                  })),
            patch(f"{_PATCH_BASE}.assemble_final_context",
                  new=_make_node_tracker(call_order, "assemble_final_context", {
                      "final_context": "assembled",
                  })),
            patch(f"{_PATCH_BASE}.generate_report",
                  new=_make_node_tracker(call_order, "generate_report", {
                      "report": "Final report depth2",
                  })),
            patch(f"{_PATCH_BASE}.should_continue_deeper", new=_should_continue_deeper),
        ):
            graph = build_deep_research_graph()
            initial = _base_state(depth=2, current_depth=2)
            result = await graph.ainvoke(initial, config=_THREAD_CONFIG)

        assert call_order == [
            "generate_research_plan",
            "generate_search_queries",
            "execute_research",       # depth 2 → go_deeper
            "fan_out_branches",
            "generate_search_queries",
            "execute_research",       # depth 1 → done
            "assemble_final_context",
            "generate_report",
        ]
        assert result["report"] == "Final report depth2"


class TestEmptyPendingBranches:
    """When depth=1, should_continue_deeper returns 'done', skipping fan_out entirely."""

    @pytest.mark.asyncio
    async def test_empty_pending_branches_leads_to_report(self):
        call_order = []

        with (
            patch(f"{_PATCH_BASE}.generate_research_plan",
                  new=_make_node_tracker(call_order, "generate_research_plan", {
                      "initial_search_results": [],
                      "follow_up_questions": [],
                      "combined_query": "combined",
                      "current_depth": 1,
                      "current_breadth": 0,
                      "all_learnings": [],
                      "all_citations": {},
                      "all_visited_urls": [],
                      "all_context": [],
                      "all_sources": [],
                      "pending_branches": [],
                      "research_tree": [],
                      "total_queries": 0,
                      "completed_queries": 0,
                  })),
            patch(f"{_PATCH_BASE}.generate_search_queries",
                  new=_make_node_tracker(call_order, "generate_search_queries", {
                      "search_queries": [{"query": "sq", "research_goal": "g",
                                          "path": "0", "parent_topic": "test query"}],
                      "total_queries": 1,
                  })),
            patch(f"{_PATCH_BASE}.execute_research",
                  new=_make_node_tracker(call_order, "execute_research", {
                      "research_results": [],
                      "all_learnings": [],
                      "all_citations": {},
                      "all_visited_urls": [],
                      "all_context": [],
                      "all_sources": [],
                      "research_tree": [],
                      "completed_queries": 0,
                      "current_breadth": 0,
                  })),
            patch(f"{_PATCH_BASE}.fan_out_branches",
                  new=_make_node_tracker(call_order, "fan_out_branches", {})),
            patch(f"{_PATCH_BASE}.assemble_final_context",
                  new=_make_node_tracker(call_order, "assemble_final_context", {
                      "final_context": "empty context",
                  })),
            patch(f"{_PATCH_BASE}.generate_report",
                  new=_make_node_tracker(call_order, "generate_report", {
                      "report": "Report from depth 1",
                  })),
            patch(f"{_PATCH_BASE}.should_continue_deeper", return_value="done"),
        ):
            graph = build_deep_research_graph()
            initial = _base_state(depth=1, current_depth=1, pending_branches=[])
            result = await graph.ainvoke(initial, config=_THREAD_CONFIG)

        assert "fan_out_branches" not in call_order
        assert call_order[-2:] == ["assemble_final_context", "generate_report"]
        assert result["report"] == "Report from depth 1"
