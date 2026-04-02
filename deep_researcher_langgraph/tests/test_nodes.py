"""Tests for deep_researcher_langgraph.nodes — every node function, conditional edge, and helper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from langchain_core.messages import AIMessage, HumanMessage

from deep_researcher_langgraph.nodes import (
    generate_research_plan,
    generate_search_queries,
    execute_research,
    fan_out_branches,
    assemble_final_context,
    generate_report,
    should_continue_deeper,
    _count_words,
    _trim_context,
    MAX_CONTEXT_WORDS,
)
from deep_researcher_langgraph.schemas import (
    FollowUpQuestionsResponse,
    LearningItem,
    ResearchAnalysis,
    SearchQueriesResponse,
    SearchQueryItem,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestCountWords:
    def test_string_input(self):
        assert _count_words("hello world foo bar") == 4

    def test_empty_string(self):
        assert _count_words("") == 0

    def test_list_input(self):
        result = _count_words(["hello world", "foo bar baz"])
        # Joined: "hello world foo bar baz" -> 5 words
        assert result == 5

    def test_list_with_non_strings(self):
        result = _count_words([1, 2, 3])
        # Joined: "1 2 3" -> 3 words
        assert result == 3


class TestCountWordsExact:
    """Re-verify _count_words with known behavior."""

    def test_string(self):
        assert _count_words("one two three") == 3

    def test_list(self):
        assert _count_words(["a b", "c d e"]) == 5


class TestTrimContext:
    def test_under_limit_returns_all(self):
        items = ["short sentence one", "short sentence two"]
        result = _trim_context(items, max_words=100)
        assert result == items

    def test_trims_to_limit_keeping_earliest(self):
        # Each item is 3 words. With max_words=5, only the first item fits.
        items = ["aaa bbb ccc", "ddd eee fff", "ggg hhh iii"]
        result = _trim_context(items, max_words=5)
        # forward iteration: "aaa bbb ccc" (3 words, ok), "ddd eee fff" (3+3=6 > 5, break)
        assert result == ["aaa bbb ccc"]

    def test_trims_large_input(self):
        # Build items that total well over MAX_CONTEXT_WORDS
        big_item = " ".join(["word"] * 10000)
        items = [big_item, big_item, big_item]
        result = _trim_context(items, max_words=MAX_CONTEXT_WORDS)
        total = sum(_count_words(r) for r in result)
        assert total <= MAX_CONTEXT_WORDS

    def test_empty_list(self):
        assert _trim_context([]) == []


# ---------------------------------------------------------------------------
# Node: generate_research_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGenerateResearchPlan:
    @patch("deep_researcher_langgraph.nodes.get_search_results", new_callable=AsyncMock)
    @patch("deep_researcher_langgraph.nodes.get_retrievers")
    async def test_returns_follow_up_questions_and_combined_query(
        self, mock_get_retrievers, mock_get_search_results,
        base_state, graph_config, llm_service,
    ):
        # Setup mock retrievers
        mock_retriever_a = MagicMock()
        mock_retriever_a.__name__ = "RetrieverA"
        mock_retriever_b = MagicMock()
        mock_retriever_b.__name__ = "RetrieverB"
        mock_get_retrievers.return_value = [mock_retriever_a, mock_retriever_b]
        mock_get_search_results.return_value = [
            {"title": "Result 1", "body": "body1", "href": "http://example.com/1"},
        ]

        # Mock LLM service
        follow_up = FollowUpQuestionsResponse(
            questions=["Q1?", "Q2?", "Q3?", "Q4?"]
        )
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=follow_up)

        result = await generate_research_plan(base_state, graph_config)

        # Follow-up questions returned
        assert result["follow_up_questions"] == ["Q1?", "Q2?", "Q3?", "Q4?"]

        # Combined query built with Q&A pairs
        assert "Initial Query:" in result["combined_query"]
        assert "Q: Q1?" in result["combined_query"]
        assert "A: Automatically proceeding with research" in result["combined_query"]

        # current_depth set to state["depth"]
        assert result["current_depth"] == base_state["depth"]

    @patch("deep_researcher_langgraph.nodes.get_search_results", new_callable=AsyncMock)
    @patch("deep_researcher_langgraph.nodes.get_retrievers")
    async def test_resets_accumulated_state(
        self, mock_get_retrievers, mock_get_search_results,
        base_state, graph_config, llm_service,
    ):
        mock_get_retrievers.return_value = []
        follow_up = FollowUpQuestionsResponse(questions=["Q1?"])
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=follow_up)

        result = await generate_research_plan(base_state, graph_config)

        assert result["all_learnings"] == []
        assert result["all_citations"] == {}
        assert result["all_context"] == []
        assert result["all_sources"] == []
        assert result["pending_branches"] == []
        assert result["research_tree"] == []
        assert result["total_queries"] == 0
        assert result["completed_queries"] == 0

    @patch("deep_researcher_langgraph.nodes.get_search_results", new_callable=AsyncMock)
    @patch("deep_researcher_langgraph.nodes.get_retrievers")
    async def test_uses_strategic_llm_with_correct_params(
        self, mock_get_retrievers, mock_get_search_results,
        base_state, graph_config, llm_service,
    ):
        mock_get_retrievers.return_value = []
        follow_up = FollowUpQuestionsResponse(questions=["Q1?"])
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=follow_up)

        await generate_research_plan(base_state, graph_config)

        llm_service.get_strategic_llm.assert_called_once_with(
            temperature=0.4,
            max_tokens=4000,
            reasoning_effort="high",
        )

    @patch("deep_researcher_langgraph.nodes.get_search_results", new_callable=AsyncMock)
    @patch("deep_researcher_langgraph.nodes.get_retrievers")
    async def test_returns_human_message_with_query(
        self, mock_get_retrievers, mock_get_search_results,
        base_state, graph_config, llm_service,
    ):
        mock_get_retrievers.return_value = []
        follow_up = FollowUpQuestionsResponse(questions=["Q1?"])
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=follow_up)

        result = await generate_research_plan(base_state, graph_config)

        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, HumanMessage)
        assert msg.content == base_state["query"]

    @patch("deep_researcher_langgraph.nodes.get_search_results", new_callable=AsyncMock)
    @patch("deep_researcher_langgraph.nodes.get_retrievers")
    async def test_limits_questions_to_breadth(
        self, mock_get_retrievers, mock_get_search_results,
        base_state, graph_config, llm_service,
    ):
        mock_get_retrievers.return_value = []
        # Return more questions than breadth
        many_qs = FollowUpQuestionsResponse(
            questions=["Q1?", "Q2?", "Q3?", "Q4?", "Q5?", "Q6?"]
        )
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=many_qs)

        base_state["breadth"] = 3
        result = await generate_research_plan(base_state, graph_config)

        assert len(result["follow_up_questions"]) == 3


# ---------------------------------------------------------------------------
# Node: generate_search_queries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGenerateSearchQueries:
    async def _run(self, state, graph_config, llm_service, num_queries=4):
        queries_resp = SearchQueriesResponse(
            queries=[
                SearchQueryItem(query=f"query {i}", research_goal=f"goal {i}")
                for i in range(num_queries)
            ]
        )
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=queries_resp)
        return await generate_search_queries(state, graph_config)

    async def test_breadth_at_top_level(self, base_state, graph_config, llm_service):
        # depth=2, current_depth=2 -> levels_deep=0 -> breadth stays original (4)
        base_state["depth"] = 2
        base_state["current_depth"] = 2
        base_state["breadth"] = 4
        result = await self._run(base_state, graph_config, llm_service, num_queries=10)
        assert len(result["search_queries"]) == 4

    async def test_breadth_one_level_deep_from_depth2(self, base_state, graph_config, llm_service):
        # depth=2, current_depth=1 -> levels_deep=1 -> breadth = max(2, 4//2) = 2
        # At a deeper level, pending_branches must be provided
        base_state["depth"] = 2
        base_state["current_depth"] = 1
        base_state["breadth"] = 4
        base_state["pending_branches"] = [
            {"query": "branch query", "depth": 1, "path": "0", "parent_topic": "Test"}
        ]
        result = await self._run(base_state, graph_config, llm_service, num_queries=10)
        assert len(result["search_queries"]) == 2

    async def test_breadth_two_levels_deep_from_depth3(self, base_state, graph_config, llm_service):
        # depth=3, current_depth=1 -> levels_deep=2
        # level 1: max(2, 4//2)=2, level 2: max(2, 2//2)=2
        base_state["depth"] = 3
        base_state["current_depth"] = 1
        base_state["breadth"] = 4
        base_state["pending_branches"] = [
            {"query": "branch query", "depth": 1, "path": "0", "parent_topic": "Test"}
        ]
        result = await self._run(base_state, graph_config, llm_service, num_queries=10)
        assert len(result["search_queries"]) == 2

    async def test_returns_correct_structure(self, base_state, graph_config, llm_service):
        result = await self._run(base_state, graph_config, llm_service, num_queries=4)
        for sq in result["search_queries"]:
            assert "query" in sq
            assert "research_goal" in sq

    async def test_total_queries_incremented(self, base_state, graph_config, llm_service):
        base_state["total_queries"] = 5
        result = await self._run(base_state, graph_config, llm_service, num_queries=4)
        assert result["total_queries"] == 5 + len(result["search_queries"])


# ---------------------------------------------------------------------------
# Node: execute_research
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecuteResearch:
    def _make_analysis(self, learnings_data, follow_ups):
        items = [
            LearningItem(insight=ins, source_url=url)
            for ins, url in learnings_data
        ]
        return ResearchAnalysis(learnings=items, follow_up_questions=follow_ups)

    @patch("deep_researcher_langgraph.nodes.GPTResearcher")
    async def test_spawns_researcher_with_correct_params(
        self, mock_gpt_cls, base_state, graph_config, llm_service,
    ):
        mock_researcher = AsyncMock()
        mock_researcher.conduct_research = AsyncMock(return_value="context text")
        mock_researcher.visited_urls = ["http://a.com"]
        mock_researcher.research_sources = [{"url": "http://a.com"}]
        mock_gpt_cls.return_value = mock_researcher

        analysis = self._make_analysis(
            [("insight1", "http://a.com")], ["follow up?"]
        )
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=analysis)

        base_state["search_queries"] = [{"query": "test query", "research_goal": "goal"}]
        base_state["tone"] = "Formal"
        base_state["mcp_configs"] = [{"server": "s1"}]
        base_state["mcp_strategy"] = "parallel"

        await execute_research(base_state, graph_config)

        mock_gpt_cls.assert_called_once()
        kwargs = mock_gpt_cls.call_args
        assert kwargs.kwargs["query"] == "test query"
        assert kwargs.kwargs["tone"] == "Formal"
        assert kwargs.kwargs["mcp_configs"] == [{"server": "s1"}]
        assert kwargs.kwargs["mcp_strategy"] == "parallel"
        assert kwargs.kwargs["config_path"] is None
        assert kwargs.kwargs["websocket"] is None  # from config
        assert kwargs.kwargs["headers"] == {}

    @patch("deep_researcher_langgraph.nodes.GPTResearcher")
    async def test_aggregates_results(
        self, mock_gpt_cls, base_state, graph_config, llm_service,
    ):
        mock_researcher = AsyncMock()
        mock_researcher.conduct_research = AsyncMock(return_value="context A")
        mock_researcher.visited_urls = ["http://a.com"]
        mock_researcher.research_sources = [{"url": "http://a.com"}]
        mock_gpt_cls.return_value = mock_researcher

        analysis = self._make_analysis(
            [("insight A", "http://a.com"), ("insight B", "")],
            ["followup A?"],
        )
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=analysis)

        base_state["search_queries"] = [
            {"query": "q1", "research_goal": "g1"},
            {"query": "q2", "research_goal": "g2"},
        ]

        result = await execute_research(base_state, graph_config)

        # Two queries -> two sets of learnings aggregated
        assert "insight A" in result["all_learnings"]
        assert "insight B" in result["all_learnings"]
        assert len(result["all_learnings"]) == 4  # 2 insights x 2 queries
        assert "http://a.com" in result["all_visited_urls"]
        assert len(result["all_context"]) == 2
        assert len(result["research_results"]) == 2

    @patch("deep_researcher_langgraph.nodes.GPTResearcher")
    async def test_handles_per_query_errors_gracefully(
        self, mock_gpt_cls, base_state, graph_config, llm_service,
    ):
        call_count = 0

        def make_researcher(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_r = AsyncMock()
            if call_count == 1:
                # First query fails
                mock_r.conduct_research = AsyncMock(side_effect=RuntimeError("boom"))
            else:
                # Second succeeds
                mock_r.conduct_research = AsyncMock(return_value="ok context")
                mock_r.visited_urls = ["http://b.com"]
                mock_r.research_sources = [{"url": "http://b.com"}]
            return mock_r

        mock_gpt_cls.side_effect = make_researcher

        analysis = self._make_analysis([("insight ok", "")], [])
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=analysis)

        base_state["search_queries"] = [
            {"query": "fail_q", "research_goal": "g1"},
            {"query": "ok_q", "research_goal": "g2"},
        ]

        result = await execute_research(base_state, graph_config)

        # Only 1 successful result
        assert len(result["research_results"]) == 1
        assert result["research_results"][0]["query"] == "ok_q"
        assert result["completed_queries"] == 1

    @patch("deep_researcher_langgraph.nodes.GPTResearcher")
    async def test_uses_strategic_llm_with_correct_params(
        self, mock_gpt_cls, base_state, graph_config, llm_service,
    ):
        mock_researcher = AsyncMock()
        mock_researcher.conduct_research = AsyncMock(return_value="ctx")
        mock_researcher.visited_urls = []
        mock_researcher.research_sources = []
        mock_gpt_cls.return_value = mock_researcher

        analysis = self._make_analysis([("x", "")], [])
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=analysis)

        base_state["search_queries"] = [{"query": "q", "research_goal": "g"}]

        await execute_research(base_state, graph_config)

        llm_service.get_strategic_llm.assert_called_with(
            temperature=0.4,
            max_tokens=4000,
            reasoning_effort="high",
        )

    @patch("deep_researcher_langgraph.nodes.GPTResearcher")
    async def test_respects_concurrency_limit(
        self, mock_gpt_cls, base_state, graph_config, llm_service,
    ):
        """Verify the semaphore is created with the right concurrency_limit.

        We cannot easily intercept asyncio.Semaphore creation, but we can verify
        that all queries complete correctly with a concurrency_limit of 1
        (serialised execution).
        """
        mock_researcher = AsyncMock()
        mock_researcher.conduct_research = AsyncMock(return_value="ctx")
        mock_researcher.visited_urls = []
        mock_researcher.research_sources = []
        mock_gpt_cls.return_value = mock_researcher

        analysis = self._make_analysis([("a", "")], [])
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke_structured = AsyncMock(return_value=analysis)

        base_state["concurrency_limit"] = 1
        base_state["search_queries"] = [
            {"query": f"q{i}", "research_goal": f"g{i}"} for i in range(4)
        ]

        result = await execute_research(base_state, graph_config)

        assert len(result["research_results"]) == 4


# ---------------------------------------------------------------------------
# Node: fan_out_branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFanOutBranches:
    async def test_creates_one_branch_per_result(self, base_state, graph_config):
        base_state["research_results"] = [
            {"research_goal": "goal1", "follow_up_questions": ["fu1"], "path": "0"},
            {"research_goal": "goal2", "follow_up_questions": ["fu2"], "path": "1"},
        ]
        base_state["current_depth"] = 3

        result = await fan_out_branches(base_state, graph_config)

        assert len(result["pending_branches"]) == 2

    async def test_inherits_path_from_parent(self, base_state, graph_config):
        base_state["research_results"] = [
            {"research_goal": "new_goal", "follow_up_questions": ["fu"], "path": "2"},
        ]
        base_state["current_depth"] = 3

        result = await fan_out_branches(base_state, graph_config)

        assert result["pending_branches"][0]["path"] == "2"
        assert result["pending_branches"][0]["parent_topic"] == "new_goal"

    async def test_each_branch_has_correct_depth(self, base_state, graph_config):
        base_state["research_results"] = [
            {"research_goal": "g", "follow_up_questions": [], "path": "0"},
        ]
        base_state["current_depth"] = 5

        result = await fan_out_branches(base_state, graph_config)

        assert result["pending_branches"][0]["depth"] == 4  # current_depth - 1

    async def test_clears_research_results(self, base_state, graph_config):
        base_state["research_results"] = [
            {"research_goal": "g", "follow_up_questions": [], "path": "0"},
        ]
        base_state["current_depth"] = 2

        result = await fan_out_branches(base_state, graph_config)

        assert result["research_results"] == []

    async def test_branch_query_format(self, base_state, graph_config):
        base_state["research_results"] = [
            {"research_goal": "my goal", "follow_up_questions": ["q1?", "q2?"], "path": "0"},
        ]
        base_state["current_depth"] = 2

        result = await fan_out_branches(base_state, graph_config)

        query = result["pending_branches"][0]["query"]
        assert "Previous research goal: my goal" in query
        assert "Follow-up questions: q1? q2?" in query


# ---------------------------------------------------------------------------
# Node: assemble_final_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAssembleFinalContext:
    async def test_deduplicates_learnings(self, base_state, graph_config, llm_service):
        base_state["research_tree"] = [
            {"path": "0", "depth_level": 1, "topic": "t",
             "learnings": ["dup", "dup", "unique"], "citations": {}, "context": ""},
        ]
        base_state["all_context"] = []

        result = await assemble_final_context(base_state, graph_config)

        # "dup" should appear only once
        assert result["final_context"].count("dup") == 1
        assert "unique" in result["final_context"]

    async def test_attaches_citations(self, base_state, graph_config, llm_service):
        base_state["research_tree"] = [
            {"path": "0", "depth_level": 1, "topic": "t",
             "learnings": ["insight A", "insight B"],
             "citations": {"insight A": "http://source.com"}, "context": ""},
        ]
        base_state["all_context"] = []

        result = await assemble_final_context(base_state, graph_config)

        assert "[Source: http://source.com]" in result["final_context"]

    async def test_extends_with_raw_context(self, base_state, graph_config, llm_service):
        base_state["research_tree"] = [
            {"path": "0", "depth_level": 1, "topic": "t",
             "learnings": ["insight"], "citations": {}, "context": ""},
        ]
        base_state["all_context"] = ["raw context block"]

        result = await assemble_final_context(base_state, graph_config)

        assert "raw context block" in result["final_context"]

    async def test_trims_to_max_context_words(self, base_state, graph_config, llm_service):
        big_learning = " ".join(["word"] * 20000)
        big_context = " ".join(["extra"] * 20000)
        base_state["research_tree"] = [
            {"path": "0", "depth_level": 1, "topic": "t",
             "learnings": [big_learning], "citations": {}, "context": ""},
        ]
        base_state["all_context"] = [big_context]

        result = await assemble_final_context(base_state, graph_config)

        word_count = len(result["final_context"].split())
        assert word_count <= MAX_CONTEXT_WORDS


# ---------------------------------------------------------------------------
# Node: generate_report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGenerateReport:
    async def test_returns_report_string(self, base_state, graph_config, llm_service):
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke = AsyncMock(return_value="# Final Report\nContent here.")

        base_state["final_context"] = "assembled context"

        result = await generate_report(base_state, graph_config)

        assert result["report"] == "# Final Report\nContent here."

    async def test_uses_strategic_llm_with_correct_params(self, base_state, graph_config, llm_service):
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke = AsyncMock(return_value="report")

        base_state["final_context"] = "ctx"

        await generate_report(base_state, graph_config)

        llm_service.get_strategic_llm.assert_called_once_with(
            temperature=0.4, max_tokens=6000,
        )

    async def test_returns_ai_message_with_report(self, base_state, graph_config, llm_service):
        report_text = "# Deep Research Report\nFindings here."
        llm_service.get_strategic_llm = MagicMock(return_value=MagicMock())
        llm_service.invoke = AsyncMock(return_value=report_text)

        base_state["final_context"] = "assembled context"

        result = await generate_report(base_state, graph_config)

        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert msg.content == report_text

    async def test_passes_query_context_tone_to_prompt(self, base_state, graph_config, llm_service):
        mock_llm = MagicMock()
        llm_service.get_strategic_llm = MagicMock(return_value=mock_llm)
        llm_service.invoke = AsyncMock(return_value="report")

        base_state["query"] = "my query"
        base_state["final_context"] = "my context"
        base_state["tone"] = "Formal"

        with patch("deep_researcher_langgraph.nodes.GENERATE_REPORT_PROMPT") as mock_prompt:
            mock_prompt.format_messages = MagicMock(return_value=["formatted"])
            await generate_report(base_state, graph_config)

            mock_prompt.format_messages.assert_called_once()
            call_kwargs = mock_prompt.format_messages.call_args[1]
            assert call_kwargs["query"] == "my query"
            assert call_kwargs["context"] == "my context"
            assert call_kwargs["tone"] == "Formal"
            assert "current_date" in call_kwargs
            llm_service.invoke.assert_called_once_with(mock_llm, ["formatted"])


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


class TestShouldContinueDeeper:
    def test_returns_go_deeper_when_depth_gt_1(self):
        state = {"current_depth": 2, "research_results": [{"query": "test"}]}
        assert should_continue_deeper(state) == "go_deeper"

    def test_returns_go_deeper_when_depth_is_3(self):
        state = {"current_depth": 3, "research_results": [{"query": "test"}]}
        assert should_continue_deeper(state) == "go_deeper"

    def test_returns_done_when_depth_is_1(self):
        state = {"current_depth": 1}
        assert should_continue_deeper(state) == "done"

    def test_returns_done_when_depth_is_0(self):
        state = {"current_depth": 0}
        assert should_continue_deeper(state) == "done"

    def test_returns_done_when_depth_missing(self):
        state = {}
        assert should_continue_deeper(state) == "done"
