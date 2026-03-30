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
        "websocket": None,
        "mcp_configs": None,
        "mcp_strategy": None,
        "on_progress": None,
        "initial_search_results": [],
        "follow_up_questions": [],
        "combined_query": "",
        "current_depth": 2,
        "current_breadth": 0,
        "search_queries": [],
        "research_results": [],
        "branch_stack": [],
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
        # The compiled graph stores node names in its .nodes mapping
        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "generate_research_plan",
            "generate_search_queries",
            "execute_research",
            "fan_out_branches",
            "pick_next_branch",
            "assemble_final_context",
            "generate_report",
        }
        # LangGraph also adds __start__ and __end__ pseudo-nodes
        assert expected_nodes.issubset(node_names), (
            f"Missing nodes: {expected_nodes - node_names}"
        )


# ---------------------------------------------------------------------------
# Flow tests (all node functions mocked)
# ---------------------------------------------------------------------------

# Patch base path for all node functions imported in graph.py
_PATCH_BASE = "deep_researcher_langgraph.graph"


def _make_node_tracker(call_order: list, node_name: str, return_value: dict):
    """Return an AsyncMock that records its call in *call_order* and returns *return_value*."""
    async def _node(state, config=None):
        call_order.append(node_name)
        return return_value

    return _node


class TestDepthOneFlow:
    """Depth=1 means should_continue_deeper returns 'check_stack' immediately,
    so the expected order is:
        generate_research_plan -> generate_search_queries -> execute_research
        -> assemble_final_context -> generate_report
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
                      "branch_stack": [],
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
                                            "visited_urls": [], "research_goal": "goal1"}],
                      "all_learnings": ["l1"],
                      "all_citations": {},
                      "all_visited_urls": [],
                      "all_context": ["ctx"],
                      "all_sources": [],
                      "completed_queries": 1,
                      "current_breadth": 1,
                  })),
            patch(f"{_PATCH_BASE}.fan_out_branches",
                  new=_make_node_tracker(call_order, "fan_out_branches", {})),
            patch(f"{_PATCH_BASE}.pick_next_branch",
                  new=_make_node_tracker(call_order, "pick_next_branch", {})),
            patch(f"{_PATCH_BASE}.assemble_final_context",
                  new=_make_node_tracker(call_order, "assemble_final_context", {
                      "final_context": "assembled context",
              
                  })),
            patch(f"{_PATCH_BASE}.generate_report",
                  new=_make_node_tracker(call_order, "generate_report", {
                      "report": "Final report",
                  })),
            patch(f"{_PATCH_BASE}.should_continue_deeper", return_value="check_stack"),
            patch(f"{_PATCH_BASE}.has_more_work", return_value="done"),
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
    """Depth=2 means the first execute_research triggers 'go_deeper',
    fan_out_branches pushes branches, pick_next_branch pops one, then
    a second execute_research at depth=1 yields 'check_stack', assemble
    runs, has_more_work returns 'done', and report generates.
    """

    @pytest.mark.asyncio
    async def test_depth2_branching_flow(self):
        call_order = []
        exec_count = {"n": 0}

        async def _execute_research(state, config=None):
            call_order.append("execute_research")
            exec_count["n"] += 1
            return {
                "research_results": [{"query": "sq", "learnings": [f"l{exec_count['n']}"],
                                      "follow_up_questions": ["fu"],
                                      "citations": {}, "context": "ctx",
                                      "sources": [], "visited_urls": [],
                                      "research_goal": "g"}],
                "all_learnings": [f"l{exec_count['n']}"],
                "all_citations": {},
                "all_visited_urls": [],
                "all_context": ["ctx"],
                "all_sources": [],
                "completed_queries": exec_count["n"],
                "current_breadth": 1,
            }

        deeper_calls = {"n": 0}

        def _should_continue_deeper(state):
            deeper_calls["n"] += 1
            # First time at depth 2 -> go_deeper; subsequent at depth 1 -> check_stack
            if deeper_calls["n"] == 1:
                return "go_deeper"
            return "check_stack"

        more_work_calls = {"n": 0}

        def _has_more_work(state):
            more_work_calls["n"] += 1
            # After fan_out: next_branch (1st), after assemble: done (2nd)
            if more_work_calls["n"] == 1:
                return "next_branch"
            return "done"

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
                      "branch_stack": [],
                      "total_queries": 0,
                      "completed_queries": 0,
              
                  })),
            patch(f"{_PATCH_BASE}.generate_search_queries",
                  new=_make_node_tracker(call_order, "generate_search_queries", {
                      "search_queries": [{"query": "sq", "research_goal": "g"}],
                      "total_queries": 1,
                  })),
            patch(f"{_PATCH_BASE}.execute_research", new=_execute_research),
            patch(f"{_PATCH_BASE}.fan_out_branches",
                  new=_make_node_tracker(call_order, "fan_out_branches", {
                      "branch_stack": [{"query": "branch1", "depth": 1}],
                      "research_results": [],
                  })),
            patch(f"{_PATCH_BASE}.pick_next_branch",
                  new=_make_node_tracker(call_order, "pick_next_branch", {
                      "combined_query": "branch1",
                      "current_depth": 1,
                      "branch_stack": [],
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
            patch(f"{_PATCH_BASE}.has_more_work", new=_has_more_work),
        ):
            graph = build_deep_research_graph()
            initial = _base_state(depth=2, current_depth=2)
            result = await graph.ainvoke(initial, config=_THREAD_CONFIG)

        assert call_order == [
            "generate_research_plan",
            "generate_search_queries",
            "execute_research",       # depth 2
            "fan_out_branches",
            "pick_next_branch",
            "generate_search_queries",
            "execute_research",       # depth 1
            "assemble_final_context",
            "generate_report",
        ]
        assert result["report"] == "Final report depth2"


class TestDFSOrdering:
    """With 2 branches at depth 2, the first branch should be fully processed
    before the second branch begins.
    """

    @pytest.mark.asyncio
    async def test_dfs_two_branches(self):
        call_order = []
        exec_count = {"n": 0}

        async def _execute_research(state, config=None):
            call_order.append("execute_research")
            exec_count["n"] += 1
            return {
                "research_results": [{"query": f"sq{exec_count['n']}",
                                      "learnings": [f"l{exec_count['n']}"],
                                      "follow_up_questions": [],
                                      "citations": {}, "context": "ctx",
                                      "sources": [], "visited_urls": [],
                                      "research_goal": "g"}],
                "all_learnings": [f"l{exec_count['n']}"],
                "all_citations": {},
                "all_visited_urls": [],
                "all_context": ["ctx"],
                "all_sources": [],
                "completed_queries": exec_count["n"],
                "current_breadth": 1,
            }

        deeper_calls = {"n": 0}

        def _should_continue_deeper(state):
            deeper_calls["n"] += 1
            # Only first execution goes deeper
            if deeper_calls["n"] == 1:
                return "go_deeper"
            return "check_stack"

        more_work_calls = {"n": 0}
        pick_calls = {"n": 0}

        def _has_more_work(state):
            more_work_calls["n"] += 1
            # 1st call (after fan_out): next_branch (branch A)
            # 2nd call (after assemble for branch A): next_branch (branch B)
            # 3rd call (after assemble for branch B): done
            if more_work_calls["n"] <= 2:
                return "next_branch"
            return "done"

        async def _pick_next_branch(state, config=None):
            call_order.append("pick_next_branch")
            pick_calls["n"] += 1
            if pick_calls["n"] == 1:
                return {
                    "combined_query": "branch_A",
                    "current_depth": 1,
                    "branch_stack": [{"query": "branch_B", "depth": 1}],
                }
            else:
                return {
                    "combined_query": "branch_B",
                    "current_depth": 1,
                    "branch_stack": [],
                }

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
                      "branch_stack": [],
                      "total_queries": 0,
                      "completed_queries": 0,
              
                  })),
            patch(f"{_PATCH_BASE}.generate_search_queries",
                  new=_make_node_tracker(call_order, "generate_search_queries", {
                      "search_queries": [{"query": "sq", "research_goal": "g"}],
                      "total_queries": 1,
                  })),
            patch(f"{_PATCH_BASE}.execute_research", new=_execute_research),
            patch(f"{_PATCH_BASE}.fan_out_branches",
                  new=_make_node_tracker(call_order, "fan_out_branches", {
                      "branch_stack": [
                          {"query": "branch_A", "depth": 1},
                          {"query": "branch_B", "depth": 1},
                      ],
                      "research_results": [],
                  })),
            patch(f"{_PATCH_BASE}.pick_next_branch", new=_pick_next_branch),
            patch(f"{_PATCH_BASE}.assemble_final_context",
                  new=_make_node_tracker(call_order, "assemble_final_context", {
                      "final_context": "assembled",
              
                  })),
            patch(f"{_PATCH_BASE}.generate_report",
                  new=_make_node_tracker(call_order, "generate_report", {
                      "report": "DFS report",
                  })),
            patch(f"{_PATCH_BASE}.should_continue_deeper", new=_should_continue_deeper),
            patch(f"{_PATCH_BASE}.has_more_work", new=_has_more_work),
        ):
            graph = build_deep_research_graph()
            initial = _base_state(depth=2, current_depth=2)
            result = await graph.ainvoke(initial, config=_THREAD_CONFIG)

        # DFS: branch A is fully processed before branch B
        assert call_order == [
            "generate_research_plan",
            "generate_search_queries",
            "execute_research",        # depth 2
            "fan_out_branches",
            "pick_next_branch",        # pops branch A
            "generate_search_queries",
            "execute_research",        # branch A at depth 1
            "assemble_final_context",  # branch A done
            "pick_next_branch",        # pops branch B
            "generate_search_queries",
            "execute_research",        # branch B at depth 1
            "assemble_final_context",  # branch B done
            "generate_report",
        ]
        assert result["report"] == "DFS report"


class TestEmptyBranchStack:
    """When the branch stack is empty after execute_research at depth 1,
    the flow should go to assemble_final_context then generate_report.
    """

    @pytest.mark.asyncio
    async def test_empty_stack_leads_to_report(self):
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
                      "branch_stack": [],
                      "total_queries": 0,
                      "completed_queries": 0,
              
                  })),
            patch(f"{_PATCH_BASE}.generate_search_queries",
                  new=_make_node_tracker(call_order, "generate_search_queries", {
                      "search_queries": [{"query": "sq", "research_goal": "g"}],
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
                      "completed_queries": 0,
                      "current_breadth": 0,
                  })),
            patch(f"{_PATCH_BASE}.fan_out_branches",
                  new=_make_node_tracker(call_order, "fan_out_branches", {})),
            patch(f"{_PATCH_BASE}.pick_next_branch",
                  new=_make_node_tracker(call_order, "pick_next_branch", {})),
            patch(f"{_PATCH_BASE}.assemble_final_context",
                  new=_make_node_tracker(call_order, "assemble_final_context", {
                      "final_context": "empty context",
              
                  })),
            patch(f"{_PATCH_BASE}.generate_report",
                  new=_make_node_tracker(call_order, "generate_report", {
                      "report": "Report from empty stack",
                  })),
            # depth=1 -> check_stack
            patch(f"{_PATCH_BASE}.should_continue_deeper", return_value="check_stack"),
            # empty stack -> done
            patch(f"{_PATCH_BASE}.has_more_work", return_value="done"),
        ):
            graph = build_deep_research_graph()
            initial = _base_state(depth=1, current_depth=1, branch_stack=[])
            result = await graph.ainvoke(initial, config=_THREAD_CONFIG)

        # Should never call fan_out_branches or pick_next_branch
        assert "fan_out_branches" not in call_order
        assert "pick_next_branch" not in call_order
        assert call_order[-2:] == ["assemble_final_context", "generate_report"]
        assert result["report"] == "Report from empty stack"
