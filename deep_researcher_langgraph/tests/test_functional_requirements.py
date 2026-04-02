"""Functional requirement regression tests for deep_researcher_langgraph.

These tests encode the behavior required by the functional spec and the
follow-up code review. Some tests are expected to fail until the reviewed
issues are fixed.
"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.types import StateSnapshot

import deep_researcher_langgraph.main as main_module
from deep_researcher_langgraph.graph import build_deep_research_graph
from deep_researcher_langgraph.main import run_deep_research
from deep_researcher_langgraph.nodes import (
    MAX_CONTEXT_WORDS,
    _trim_context,
    assemble_final_context,
    execute_research,
    generate_research_plan,
    generate_search_queries,
    should_continue_deeper,
)
from deep_researcher_langgraph.prompts import GENERATE_REPORT_PROMPT
from deep_researcher_langgraph.schemas import (
    FollowUpQuestionsResponse,
    LearningItem,
    ResearchAnalysis,
    SearchQueriesResponse,
    SearchQueryItem,
)
from gpt_researcher.skills.researcher import ResearchConductor


GRAPH_MODULE = "deep_researcher_langgraph.graph"
NODES_MODULE = "deep_researcher_langgraph.nodes"
MAIN_MODULE = "deep_researcher_langgraph.main"


def _state(base_state, **overrides):
    state = deepcopy(base_state)
    state.update(overrides)
    return state


def _analysis_for(query: str, *, follow_ups=None, source_url=None) -> ResearchAnalysis:
    return ResearchAnalysis(
        learnings=[
            LearningItem(
                insight=f"learning for {query}",
                source_url=source_url or f"https://{query}.example.com",
            )
        ],
        follow_up_questions=follow_ups or [f"follow-up for {query}?"],
    )


def _thread_config():
    return {"configurable": {"thread_id": "functional-requirements-thread"}}


def _mock_config_cls(*, breadth=4, depth=2, concurrency=2):
    cfg = MagicMock()
    cfg.deep_research_breadth = breadth
    cfg.deep_research_depth = depth
    cfg.deep_research_concurrency = concurrency
    return cfg


class _ReorderingSet(set):
    """Deterministically exposes why set()-based dedupe is not stable."""

    def __init__(self, iterable=()):
        self._ordered = list(dict.fromkeys(iterable))
        super().__init__(self._ordered)

    def __iter__(self):
        return iter(reversed(self._ordered))


@pytest.mark.asyncio
async def test_depth_one_graph_stops_without_recursing(base_state):
    """AC4 edge case: depth=1 should never fan out to a deeper level."""

    call_order = []

    async def plan(state, config=None):
        call_order.append("plan")
        return {
            "combined_query": "combined root query",
            "current_depth": 1,
            "pending_branches": [],
            "research_tree": [],
            "all_learnings": [],
            "all_citations": {},
            "all_context": [],
            "all_sources": [],
            "total_queries": 0,
            "completed_queries": 0,
        }

    async def search_queries(state, config=None):
        call_order.append("queries")
        return {
            "search_queries": [
                {
                    "query": "root query",
                    "research_goal": "Root Goal",
                    "path": "0",
                    "parent_topic": state["query"],
                }
            ],
            "total_queries": 1,
        }

    async def execute(state, config=None):
        call_order.append("research")
        return {
            "research_results": [
                {
                    "query": "root query",
                    "path": "0",
                    "parent_topic": state["query"],
                    "learnings": ["root learning"],
                    "follow_up_questions": [],
                    "citations": {},
                    "context": "root context",
                    "sources": [],
                    "visited_urls": [],
                    "research_goal": "Root Goal",
                }
            ],
            "research_tree": [],
            "all_learnings": ["root learning"],
            "all_citations": {},
            "all_visited_urls": [],
            "all_context": ["root context"],
            "all_sources": [],
            "completed_queries": 1,
            "current_breadth": 1,
        }

    async def fail_fan_out(state, config=None):
        pytest.fail("fan_out_branches should not run when depth=1")

    async def assemble(state, config=None):
        call_order.append("assemble")
        return {"final_context": "assembled context"}

    async def report(state, config=None):
        call_order.append("report")
        return {"report": "final report"}

    with (
        patch(f"{GRAPH_MODULE}.generate_research_plan", new=plan),
        patch(f"{GRAPH_MODULE}.generate_search_queries", new=search_queries),
        patch(f"{GRAPH_MODULE}.execute_research", new=execute),
        patch(f"{GRAPH_MODULE}.fan_out_branches", new=fail_fan_out),
        patch(f"{GRAPH_MODULE}.assemble_final_context", new=assemble),
        patch(f"{GRAPH_MODULE}.generate_report", new=report),
    ):
        graph = build_deep_research_graph()
        result = await graph.ainvoke(
            _state(base_state, depth=1, current_depth=1),
            config=_thread_config(),
        )

    assert call_order == ["plan", "queries", "research", "assemble", "report"]
    assert result["report"] == "final report"


@pytest.mark.asyncio
async def test_generate_search_queries_executes_pending_branches_concurrently_with_semaphore(
    base_state, graph_config, llm_service
):
    """AC1: pending branches at the same level must run concurrently, semaphore-limited."""

    state = _state(
        base_state,
        depth=2,
        current_depth=1,
        breadth=4,
        concurrency_limit=2,
        pending_branches=[
            {"query": "branch 0", "depth": 1, "path": "0", "parent_topic": "Parent 0"},
            {"query": "branch 1", "depth": 1, "path": "1", "parent_topic": "Parent 1"},
            {"query": "branch 2", "depth": 1, "path": "2", "parent_topic": "Parent 2"},
        ],
    )

    llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())

    active = 0
    max_active = 0
    started_topics = []
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def fake_invoke_structured(llm, schema, messages):
        nonlocal active, max_active
        topic = messages[1].content.rsplit("Topic: ", 1)[1]
        started_topics.append(topic)
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            two_started.set()
        if len(started_topics) <= 2:
            await release.wait()
        active -= 1
        return SearchQueriesResponse(
            queries=[
                SearchQueryItem(query=f"{topic} / child 0", research_goal=f"{topic} goal 0"),
                SearchQueryItem(query=f"{topic} / child 1", research_goal=f"{topic} goal 1"),
            ]
        )

    llm_service.invoke_structured = AsyncMock(side_effect=fake_invoke_structured)

    task = asyncio.create_task(generate_search_queries(state, graph_config))
    await two_started.wait()

    assert max_active == 2
    assert len(started_topics) == 2

    release.set()
    result = await task

    assert max_active == 2
    assert len(started_topics) == 3
    assert {item["path"] for item in result["search_queries"]} == {
        "0.0",
        "0.1",
        "1.0",
        "1.1",
        "2.0",
        "2.1",
    }
    assert "semaphore" not in result


@pytest.mark.asyncio
async def test_execute_research_runs_same_level_queries_concurrently_and_uses_copied_visited_sets(
    base_state, graph_config, llm_service
):
    """AC1 + AC5: same-level research is concurrent and each researcher gets an isolated visited-url copy."""

    seeded = ["https://seeded.example.com"]
    state = _state(
        base_state,
        search_queries=[
            {"query": "q0", "research_goal": "Goal 0", "path": "0", "parent_topic": "Root"},
            {"query": "q1", "research_goal": "Goal 1", "path": "1", "parent_topic": "Root"},
            {"query": "q2", "research_goal": "Goal 2", "path": "2", "parent_topic": "Root"},
        ],
        all_visited_urls=seeded,
        concurrency_limit=2,
    )

    active = 0
    max_active = 0
    two_started = asyncio.Event()
    release = asyncio.Event()
    instances = []

    class TrackingResearcher:
        def __init__(self, **kwargs):
            self.query = kwargs["query"]
            self.seed_snapshot = set(kwargs["visited_urls"])
            self.visited_urls = kwargs["visited_urls"]
            self.research_sources = [{"url": f"https://{self.query}.example.com"}]
            instances.append(self)

        async def conduct_research(self):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                two_started.set()
            if self.query in {"q0", "q1"}:
                await release.wait()
            self.visited_urls.add(f"https://{self.query}.example.com")
            active -= 1
            return f"context for {self.query}"

    async def fake_analysis(llm, schema, messages):
        query = messages[1].content.split("query '", 1)[1].split("'", 1)[0]
        return _analysis_for(query)

    llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
    llm_service.invoke_structured = AsyncMock(side_effect=fake_analysis)

    with patch(f"{NODES_MODULE}.GPTResearcher", new=TrackingResearcher):
        task = asyncio.create_task(execute_research(state, graph_config))
        await two_started.wait()
        assert max_active == 2
        release.set()
        result = await task

    assert max_active == 2
    assert len(instances) == 3
    assert all(instance.seed_snapshot == set(seeded) for instance in instances)
    assert len({id(instance.visited_urls) for instance in instances}) == 3
    assert state["all_visited_urls"] == seeded
    assert len(result["research_results"]) == 3
    assert "https://q0.example.com" in result["all_visited_urls"]


@pytest.mark.asyncio
async def test_execute_research_produces_partial_results_when_only_some_branches_fail(
    base_state, graph_config, llm_service, caplog
):
    """AC3: failed branches should be skipped and successful branches should still produce output."""

    state = _state(
        base_state,
        search_queries=[
            {"query": "fail-me", "research_goal": "Fail", "path": "0", "parent_topic": "Root"},
            {"query": "keep-me", "research_goal": "Keep", "path": "1", "parent_topic": "Root"},
        ],
    )

    class FlakyResearcher:
        def __init__(self, **kwargs):
            self.query = kwargs["query"]
            self.visited_urls = kwargs["visited_urls"]
            self.research_sources = [{"url": f"https://{self.query}.example.com"}]

        async def conduct_research(self):
            if self.query == "fail-me":
                raise RuntimeError("branch exploded")
            self.visited_urls.add("https://keep-me.example.com")
            return "successful context"

    llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
    llm_service.invoke_structured = AsyncMock(return_value=_analysis_for("keep-me"))

    with patch(f"{NODES_MODULE}.GPTResearcher", new=FlakyResearcher):
        result = await execute_research(state, graph_config)

    assert len(result["research_results"]) == 1
    assert result["research_results"][0]["query"] == "keep-me"
    assert len(result["research_tree"]) == 1
    assert "fail-me" in caplog.text


@pytest.mark.asyncio
async def test_execute_research_raises_when_every_branch_fails(
    base_state, graph_config, llm_service
):
    """AC3 edge case: the node must raise only when a whole level fails."""

    state = _state(
        base_state,
        search_queries=[
            {"query": "a", "research_goal": "A", "path": "0", "parent_topic": "Root"},
            {"query": "b", "research_goal": "B", "path": "1", "parent_topic": "Root"},
        ],
    )

    class AlwaysFailResearcher:
        def __init__(self, **kwargs):
            self.visited_urls = kwargs["visited_urls"]
            self.research_sources = []

        async def conduct_research(self):
            raise RuntimeError("no branch survived")

    llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
    llm_service.invoke_structured = AsyncMock(return_value=_analysis_for("unused"))

    with patch(f"{NODES_MODULE}.GPTResearcher", new=AlwaysFailResearcher):
        with pytest.raises(RuntimeError, match="All 2 research queries failed"):
            await execute_research(state, graph_config)


def test_should_continue_deeper_stops_when_the_current_level_produced_no_results():
    """Zero-child descent bug: deeper traversal requires successful results for the current level."""

    assert should_continue_deeper({"current_depth": 2, "research_results": []}) == "done"


@pytest.mark.asyncio
async def test_generate_search_queries_does_not_regenerate_root_queries_when_deeper_level_has_no_pending_branches(
    base_state, graph_config, llm_service
):
    """Zero-child edge case: an empty deeper level should finish, not restart from combined_query."""

    state = _state(
        base_state,
        depth=3,
        current_depth=2,
        combined_query="this should not be reused",
        pending_branches=[],
        total_queries=5,
    )

    llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
    llm_service.invoke_structured = AsyncMock(
        return_value=SearchQueriesResponse(
            queries=[SearchQueryItem(query="unexpected", research_goal="unexpected")]
        )
    )

    result = await generate_search_queries(state, graph_config)

    assert result["search_queries"] == []
    assert result["total_queries"] == 5
    llm_service.invoke_structured.assert_not_called()


@pytest.mark.asyncio
async def test_generate_research_plan_degrades_gracefully_when_all_retrievers_fail(
    base_state, graph_config, llm_service
):
    """AC3: the planning stage must continue with empty search context instead of hard-failing."""

    class RetrieverA:
        __name__ = "RetrieverA"

    class RetrieverB:
        __name__ = "RetrieverB"

    llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
    llm_service.invoke_structured = AsyncMock(
        return_value=FollowUpQuestionsResponse(questions=["Q1?", "Q2?"])
    )

    with (
        patch(f"{NODES_MODULE}.get_retrievers", return_value=[RetrieverA, RetrieverB]),
        patch(
            f"{NODES_MODULE}.get_search_results",
            new=AsyncMock(side_effect=RuntimeError("retriever outage")),
        ),
    ):
        result = await generate_research_plan(base_state, graph_config)

    assert result["initial_search_results"] == []
    assert result["follow_up_questions"] == ["Q1?", "Q2?"]


@pytest.mark.asyncio
async def test_generate_research_plan_handles_mcp_only_configuration_without_hard_failure(
    base_state, graph_config, llm_service
):
    """MCP-only setups must not fail just because the initial search path needs a researcher object."""

    state = _state(base_state, mcp_configs=[{"name": "local-mcp"}])

    class MCPRetriever:
        __name__ = "MCPRetriever"

    async def fake_get_search_results(query, retriever, query_domains=None, researcher=None):
        if researcher is None:
            raise ValueError("researcher is required for MCP retrievers")
        return []

    llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
    llm_service.invoke_structured = AsyncMock(
        return_value=FollowUpQuestionsResponse(questions=["MCP follow-up?"])
    )

    with (
        patch(f"{NODES_MODULE}.get_retrievers", return_value=[MCPRetriever]),
        patch(f"{NODES_MODULE}.get_search_results", new=AsyncMock(side_effect=fake_get_search_results)),
    ):
        result = await generate_research_plan(state, graph_config)

    assert result["follow_up_questions"] == ["MCP follow-up?"]


@pytest.mark.asyncio
async def test_assemble_final_context_uses_hierarchical_headings_and_numeric_path_order_for_double_digit_paths(
    base_state, graph_config
):
    """AC2 + AC6: headings must match depth and numeric path ordering must handle breadth > 9."""

    state = _state(
        base_state,
        depth=3,
        research_tree=[
            {
                "path": "10.0.0",
                "depth_level": 1,
                "topic": "Leaf 10.0.0",
                "learnings": ["leaf detail"],
                "citations": {},
                "context": "",
            },
            {
                "path": "10",
                "depth_level": 3,
                "topic": "Topic 10",
                "learnings": ["topic 10 learning"],
                "citations": {},
                "context": "",
            },
            {
                "path": "2.0",
                "depth_level": 2,
                "topic": "Topic 2.0",
                "learnings": ["topic 2.0 learning"],
                "citations": {},
                "context": "",
            },
            {
                "path": "2",
                "depth_level": 3,
                "topic": "Topic 2",
                "learnings": ["topic 2 learning"],
                "citations": {},
                "context": "",
            },
            {
                "path": "10.0",
                "depth_level": 2,
                "topic": "Topic 10.0",
                "learnings": ["topic 10.0 learning"],
                "citations": {},
                "context": "",
            },
        ],
    )

    result = await assemble_final_context(state, graph_config)
    lines = result["final_context"].splitlines()

    idx_topic_2 = lines.index("## Topic 2")
    idx_topic_20 = lines.index("### Topic 2.0")
    idx_topic_10 = lines.index("## Topic 10")
    idx_topic_100 = lines.index("### Topic 10.0")
    idx_leaf = lines.index("#### Leaf 10.0.0")

    assert idx_topic_2 < idx_topic_20 < idx_topic_10 < idx_topic_100 < idx_leaf


@pytest.mark.asyncio
async def test_assemble_final_context_trimming_preserves_sorted_order_and_stable_dedup(
    base_state, graph_config
):
    """AC6: sort first, dedupe stably, then trim without scrambling insertion order."""

    first_context = " ".join(["first"] * 24993)
    second_context = " ".join(["second"] * 9000)

    state = _state(
        base_state,
        depth=2,
        research_tree=[
            {
                "path": "1",
                "depth_level": 2,
                "topic": "Later Topic",
                "learnings": ["later learning"],
                "citations": {},
                "context": second_context,
            },
            {
                "path": "0",
                "depth_level": 2,
                "topic": "Earlier Topic",
                "learnings": ["shared learning", "shared learning"],
                "citations": {},
                "context": first_context,
            },
        ],
    )

    result = await assemble_final_context(state, graph_config)
    text = result["final_context"]

    assert "## Earlier Topic" in text
    assert "## Later Topic" not in text
    assert text.count("shared learning") == 1
    assert len(text.split()) <= MAX_CONTEXT_WORDS


@pytest.mark.asyncio
async def test_real_graph_reducers_accumulate_research_tree_across_levels_into_final_context(base_state):
    """Use the real StateGraph reducers to verify ancestor and child nodes survive multi-level traversal."""

    async def plan(state, config=None):
        return {
            "combined_query": "combined root query",
            "current_depth": 2,
            "pending_branches": [],
            "research_tree": [],
            "all_learnings": [],
            "all_citations": {},
            "all_context": [],
            "all_sources": [],
            "total_queries": 0,
            "completed_queries": 0,
        }

    async def search_queries(state, config=None):
        if state["current_depth"] == 2:
            return {
                "search_queries": [
                    {
                        "query": "root query",
                        "research_goal": "Root Goal",
                        "path": "0",
                        "parent_topic": state["query"],
                    }
                ],
                "total_queries": state.get("total_queries", 0) + 1,
            }
        return {
            "search_queries": [
                {
                    "query": "child query",
                    "research_goal": "Child Goal",
                    "path": "0.0",
                    "parent_topic": "Root Goal",
                }
            ],
            "total_queries": state.get("total_queries", 0) + 1,
        }

    async def execute(state, config=None):
        if state["current_depth"] == 2:
            return {
                "research_results": [
                    {
                        "query": "root query",
                        "path": "0",
                        "parent_topic": state["query"],
                        "learnings": ["root learning"],
                        "follow_up_questions": ["drill deeper"],
                        "citations": {"root learning": "https://root.example.com"},
                        "context": "root context",
                        "sources": [],
                        "visited_urls": [],
                        "research_goal": "Root Goal",
                    }
                ],
                "research_tree": [
                    {
                        "path": "0",
                        "depth_level": 2,
                        "topic": "Root Goal",
                        "learnings": ["root learning"],
                        "citations": {"root learning": "https://root.example.com"},
                        "context": "root context",
                    }
                ],
                "all_learnings": ["root learning"],
                "all_citations": {"root learning": "https://root.example.com"},
                "all_visited_urls": [],
                "all_context": ["root context"],
                "all_sources": [],
                "completed_queries": 1,
                "current_breadth": 1,
            }
        return {
            "research_results": [
                {
                    "query": "child query",
                    "path": "0.0",
                    "parent_topic": "Root Goal",
                    "learnings": ["child learning"],
                    "follow_up_questions": [],
                    "citations": {"child learning": "https://child.example.com"},
                    "context": "child context",
                    "sources": [],
                    "visited_urls": [],
                    "research_goal": "Child Goal",
                }
            ],
            "research_tree": [
                {
                    "path": "0.0",
                    "depth_level": 1,
                    "topic": "Child Goal",
                    "learnings": ["child learning"],
                    "citations": {"child learning": "https://child.example.com"},
                    "context": "child context",
                }
            ],
            "all_learnings": ["child learning"],
            "all_citations": {"child learning": "https://child.example.com"},
            "all_visited_urls": [],
            "all_context": ["child context"],
            "all_sources": [],
            "completed_queries": state.get("completed_queries", 0) + 1,
            "current_breadth": 1,
        }

    async def fan_out(state, config=None):
        return {
            "pending_branches": [
                {
                    "query": "child branch query",
                    "depth": 1,
                    "path": "0",
                    "parent_topic": "Root Goal",
                }
            ],
            "current_depth": 1,
            "research_results": [],
        }

    async def report(state, config=None):
        return {"report": state["final_context"]}

    with (
        patch(f"{GRAPH_MODULE}.generate_research_plan", new=plan),
        patch(f"{GRAPH_MODULE}.generate_search_queries", new=search_queries),
        patch(f"{GRAPH_MODULE}.execute_research", new=execute),
        patch(f"{GRAPH_MODULE}.fan_out_branches", new=fan_out),
        patch(f"{GRAPH_MODULE}.generate_report", new=report),
    ):
        graph = build_deep_research_graph()
        result = await graph.ainvoke(
            _state(base_state, depth=2, current_depth=2),
            config=_thread_config(),
        )

    assert "## Root Goal" in result["report"]
    assert "### Child Goal" in result["report"]
    assert result["report"].index("## Root Goal") < result["report"].index("### Child Goal")


@pytest.mark.asyncio
async def test_graph_reducer_merges_visited_urls_as_an_ordered_union_across_levels(base_state):
    """AC5 + AC6: seeded and newly visited URLs must merge once, in stable order."""

    async def plan(state, config=None):
        return {
            "combined_query": "combined root query",
            "current_depth": 2,
            "pending_branches": [],
            "research_tree": [],
            "all_learnings": [],
            "all_citations": {},
            "all_context": [],
            "all_sources": [],
            "total_queries": 0,
            "completed_queries": 0,
        }

    async def search_queries(state, config=None):
        return {
            "search_queries": [
                {
                    "query": f"query depth {state['current_depth']}",
                    "research_goal": f"Goal depth {state['current_depth']}",
                    "path": "0" if state["current_depth"] == 2 else "0.0",
                    "parent_topic": state["query"],
                }
            ],
            "total_queries": state.get("total_queries", 0) + 1,
        }

    async def execute(state, config=None):
        if state["current_depth"] == 2:
            return {
                "research_results": [
                    {
                        "query": "query depth 2",
                        "path": "0",
                        "parent_topic": state["query"],
                        "learnings": [],
                        "follow_up_questions": ["go deeper"],
                        "citations": {},
                        "context": "",
                        "sources": [],
                        "visited_urls": ["https://shared.example.com", "https://root.example.com"],
                        "research_goal": "Root Goal",
                    }
                ],
                "research_tree": [],
                "all_learnings": [],
                "all_citations": {},
                "all_visited_urls": ["https://shared.example.com", "https://root.example.com"],
                "all_context": [],
                "all_sources": [],
                "completed_queries": 1,
                "current_breadth": 1,
            }
        return {
            "research_results": [
                {
                    "query": "query depth 1",
                    "path": "0.0",
                    "parent_topic": "Root Goal",
                    "learnings": [],
                    "follow_up_questions": [],
                    "citations": {},
                    "context": "",
                    "sources": [],
                    "visited_urls": ["https://shared.example.com", "https://child.example.com"],
                    "research_goal": "Child Goal",
                }
            ],
            "research_tree": [],
            "all_learnings": [],
            "all_citations": {},
            "all_visited_urls": ["https://shared.example.com", "https://child.example.com"],
            "all_context": [],
            "all_sources": [],
            "completed_queries": state.get("completed_queries", 0) + 1,
            "current_breadth": 1,
        }

    async def fan_out(state, config=None):
        return {
            "pending_branches": [
                {
                    "query": "child branch query",
                    "depth": 1,
                    "path": "0",
                    "parent_topic": "Root Goal",
                }
            ],
            "current_depth": 1,
            "research_results": [],
        }

    async def assemble(state, config=None):
        return {"final_context": "context"}

    async def report(state, config=None):
        return {"report": "report"}

    with (
        patch(f"{GRAPH_MODULE}.generate_research_plan", new=plan),
        patch(f"{GRAPH_MODULE}.generate_search_queries", new=search_queries),
        patch(f"{GRAPH_MODULE}.execute_research", new=execute),
        patch(f"{GRAPH_MODULE}.fan_out_branches", new=fan_out),
        patch(f"{GRAPH_MODULE}.assemble_final_context", new=assemble),
        patch(f"{GRAPH_MODULE}.generate_report", new=report),
    ):
        graph = build_deep_research_graph()
        result = await graph.ainvoke(
            _state(
                base_state,
                depth=2,
                current_depth=2,
                all_visited_urls=[
                    "https://seeded.example.com",
                    "https://shared.example.com",
                ],
            ),
            config=_thread_config(),
        )

    assert result["all_visited_urls"] == [
        "https://seeded.example.com",
        "https://shared.example.com",
        "https://root.example.com",
        "https://child.example.com",
    ]


@pytest.mark.asyncio
async def test_research_conductor_does_not_clear_seeded_visited_urls():
    """Regression test for the external visited-url clearing bug cited by the review."""

    class DummyRetriever:
        __name__ = "DummyRetriever"

    dummy_researcher = SimpleNamespace(
        query="retained visited urls",
        visited_urls={"https://seeded.example.com"},
        verbose=False,
        websocket=None,
        agent=None,
        role=None,
        retrievers=[DummyRetriever],
        cfg=MagicMock(),
        parent_query=None,
        add_costs=MagicMock(),
        headers={},
        prompt_family=MagicMock(),
        source_urls=[],
    )
    conductor = ResearchConductor(dummy_researcher)

    with patch(
        "gpt_researcher.skills.researcher.choose_agent",
        new=AsyncMock(side_effect=RuntimeError("stop after setup")),
    ):
        with pytest.raises(RuntimeError, match="stop after setup"):
            await conductor.conduct_research()

    assert dummy_researcher.visited_urls == {"https://seeded.example.com"}


@pytest.mark.asyncio
async def test_progress_callback_exceptions_do_not_crash_generate_search_queries(
    base_state, graph_config, llm_service
):
    """Progress callbacks are informational and must not abort the workflow."""

    state = _state(
        base_state,
        depth=2,
        current_depth=2,
        combined_query="combined query",
    )

    # on_progress is now in config, not state
    graph_config["configurable"]["on_progress"] = MagicMock(
        side_effect=RuntimeError("progress callback failed")
    )

    llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
    llm_service.invoke_structured = AsyncMock(
        return_value=SearchQueriesResponse(
            queries=[SearchQueryItem(query="q0", research_goal="goal 0")]
        )
    )

    result = await generate_search_queries(state, graph_config)

    assert result["search_queries"] == [
        {
            "query": "q0",
            "research_goal": "goal 0",
            "path": "0",
            "parent_topic": state["query"],
        }
    ]


@pytest.mark.asyncio
async def test_run_deep_research_accepts_runtime_configuration_and_tone():
    """AC7: breadth, depth, concurrency, and tone must be configurable at runtime."""

    mock_cfg = _mock_config_cls()
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "report": "report",
            "final_context": "context",
            "all_visited_urls": [],
            "all_sources": [],
            "all_learnings": [],
            "all_citations": {},
        }
    )

    with (
        patch(f"{MAIN_MODULE}.Config", return_value=mock_cfg),
        patch(f"{MAIN_MODULE}.build_deep_research_graph", return_value=graph),
        patch(f"{MAIN_MODULE}.TokenUsageCallbackHandler"),
        patch(f"{MAIN_MODULE}.LLMService") as llm_service_cls,
    ):
        llm_service_cls.return_value.get_usage_summary.return_value = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        await run_deep_research(
            query="configured query",
            breadth=5,
            depth=3,
            concurrency_limit=4,
            tone="Analytical",
        )

    invoked_state = graph.ainvoke.call_args[0][0]
    assert invoked_state["breadth"] == 5
    assert invoked_state["depth"] == 3
    assert invoked_state["concurrency_limit"] == 4
    assert invoked_state["tone"] == "Analytical"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"breadth": 0},
        {"breadth": -1},
        {"depth": 0},
        {"depth": -1},
        {"concurrency_limit": 0},
        {"concurrency_limit": -1},
    ],
)
async def test_run_deep_research_validates_runtime_parameters(kwargs):
    """Invalid numeric runtime inputs should raise instead of falling through to defaults."""

    mock_cfg = _mock_config_cls()
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={})

    with (
        patch(f"{MAIN_MODULE}.Config", return_value=mock_cfg),
        patch(f"{MAIN_MODULE}.build_deep_research_graph", return_value=graph),
        patch(f"{MAIN_MODULE}.TokenUsageCallbackHandler"),
        patch(f"{MAIN_MODULE}.LLMService") as llm_service_cls,
    ):
        llm_service_cls.return_value.get_usage_summary.return_value = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        with pytest.raises(ValueError):
            await run_deep_research(query="validated query", **kwargs)


@pytest.mark.asyncio
async def test_run_deep_research_enforces_timeout_and_returns_partial_snapshot(monkeypatch):
    """AC7: a wall-clock timeout should return partial checkpointed output instead of hard-failing."""

    mock_cfg = _mock_config_cls()
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "report": "unexpected full report",
            "final_context": "unexpected full context",
            "all_visited_urls": [],
            "all_sources": [],
            "all_learnings": [],
            "all_citations": {},
        }
    )
    partial_state = {
        "report": "partial timeout report",
        "final_context": "partial timeout context",
        "all_visited_urls": ["https://timeout.example.com"],
        "all_sources": [{"url": "https://timeout.example.com"}],
        "all_learnings": ["partial learning"],
        "all_citations": {"partial learning": "https://timeout.example.com"},
    }
    graph.aget_state = AsyncMock(
        return_value=StateSnapshot(
            values=partial_state,
            next=(),
            config={"configurable": {"thread_id": "timeout-thread"}},
            metadata=None,
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )
    )

    timeouts = []

    async def fake_wait_for(awaitable, timeout):
        timeouts.append(timeout)
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(main_module.asyncio, "wait_for", fake_wait_for)

    with (
        patch(f"{MAIN_MODULE}.Config", return_value=mock_cfg),
        patch(f"{MAIN_MODULE}.build_deep_research_graph", return_value=graph),
        patch(f"{MAIN_MODULE}.TokenUsageCallbackHandler"),
        patch(f"{MAIN_MODULE}.LLMService") as llm_service_cls,
    ):
        llm_service_cls.return_value.get_usage_summary.return_value = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        result = await run_deep_research(query="timeout query", thread_id="timeout-thread")

    assert timeouts == [1800]
    assert result["report"] == "partial timeout report"
    assert result["final_context"] == "partial timeout context"
    assert result["visited_urls"] == ["https://timeout.example.com"]
    assert result["learnings"] == ["partial learning"]


@pytest.mark.asyncio
async def test_run_deep_research_deduplicates_learnings_in_stable_insertion_order(monkeypatch):
    """AC6: final learnings output must preserve insertion order instead of relying on set()."""

    mock_cfg = _mock_config_cls()
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "report": "report",
            "final_context": "context",
            "all_visited_urls": [],
            "all_sources": [],
            "all_learnings": ["first", "second", "first", "third"],
            "all_citations": {},
        }
    )

    monkeypatch.setattr(main_module, "set", _ReorderingSet, raising=False)

    with (
        patch(f"{MAIN_MODULE}.Config", return_value=mock_cfg),
        patch(f"{MAIN_MODULE}.build_deep_research_graph", return_value=graph),
        patch(f"{MAIN_MODULE}.TokenUsageCallbackHandler"),
        patch(f"{MAIN_MODULE}.LLMService") as llm_service_cls,
    ):
        llm_service_cls.return_value.get_usage_summary.return_value = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        result = await run_deep_research(query="stable learnings query")

    assert result["learnings"] == ["first", "second", "third"]


def test_generate_report_prompt_has_no_fixed_minimum_word_count():
    """Spec constraint: best-effort partial outputs are valid; no hard minimum length is allowed."""

    human_prompt = GENERATE_REPORT_PROMPT.format_messages(
        query="partial results query",
        context="sparse context",
        tone="Objective",
        current_date="April 2, 2026",
    )[1].content.lower()

    assert "at least 1200 words" not in human_prompt


def test_trim_context_keeps_earliest_items_in_insertion_order():
    """AC6 helper coverage: trimming should preserve the original order rather than reordering items."""

    trimmed = _trim_context(
        [
            "zero one",
            "two three",
            "four five",
        ],
        max_words=4,
    )

    assert trimmed == ["zero one", "two three"]
