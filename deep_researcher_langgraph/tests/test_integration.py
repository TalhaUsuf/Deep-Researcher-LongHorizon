"""Integration tests for run_deep_research and the main() CLI entry point."""

import argparse
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from deep_researcher_langgraph.main import run_deep_research, main
from deep_researcher_langgraph.state import ResearchProgress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "deep_researcher_langgraph.main"


def _fake_final_state():
    """Return a plausible final state dict as would come from graph.ainvoke."""
    return {
        "report": "Test report content",
        "final_context": "Assembled context",
        "all_visited_urls": ["https://example.com"],
        "all_sources": [{"url": "https://example.com", "title": "Example"}],
        "all_learnings": ["learning1", "learning2", "learning1"],  # duplicates
        "all_citations": {"learning1": "https://example.com"},
        "query": "test query",
        "breadth": 4,
        "depth": 2,
    }


def _mock_graph(final_state=None):
    """Return a MagicMock graph whose ainvoke returns *final_state*."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=final_state or _fake_final_state())
    return graph


def _mock_config_cls(breadth=4, depth=2, concurrency=2):
    """Return a MagicMock that behaves like gpt_researcher Config."""
    cfg = MagicMock()
    cfg.deep_research_breadth = breadth
    cfg.deep_research_depth = depth
    cfg.deep_research_concurrency = concurrency
    return cfg


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

class TestRunDeepResearch:

    @pytest.mark.asyncio
    async def test_returns_expected_keys(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {
                "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
            }
            result = await run_deep_research(query="test query")

        expected_keys = {
            "report", "final_context", "visited_urls",
            "sources", "learnings", "citations", "usage_summary", "thread_id",
        }
        assert set(result.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_report_and_context_values(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            result = await run_deep_research(query="test query")

        assert result["report"] == "Test report content"
        assert result["final_context"] == "Assembled context"
        assert result["visited_urls"] == ["https://example.com"]

    @pytest.mark.asyncio
    async def test_learnings_are_deduplicated(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            result = await run_deep_research(query="test query")

        # The fake state has ["learning1", "learning2", "learning1"] -> set removes dup
        assert sorted(result["learnings"]) == ["learning1", "learning2"]

    @pytest.mark.asyncio
    async def test_returns_thread_id(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            result = await run_deep_research(query="test query")

        assert "thread_id" in result
        assert isinstance(result["thread_id"], str)
        assert len(result["thread_id"]) > 0

    @pytest.mark.asyncio
    async def test_explicit_thread_id_returned_unchanged(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()
        explicit_id = "my-custom-thread-id-123"

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            result = await run_deep_research(query="test query", thread_id=explicit_id)

        assert result["thread_id"] == explicit_id

    @pytest.mark.asyncio
    async def test_thread_id_passed_to_graph_configurable(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()
        explicit_id = "configurable-thread-456"

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            await run_deep_research(query="test query", thread_id=explicit_id)

        config_arg = mock_g.ainvoke.call_args[1]["config"]
        assert config_arg["configurable"]["thread_id"] == explicit_id


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------

class TestParameterResolution:

    @pytest.mark.asyncio
    async def test_defaults_from_config(self):
        """When breadth/depth/concurrency are not provided, they come from Config."""
        mock_cfg = _mock_config_cls(breadth=6, depth=3, concurrency=4)
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            await run_deep_research(query="q")

        # Verify values passed in initial state
        invoked_state = mock_g.ainvoke.call_args[0][0]
        assert invoked_state["breadth"] == 6
        assert invoked_state["depth"] == 3
        assert invoked_state["concurrency_limit"] == 4

    @pytest.mark.asyncio
    async def test_explicit_params_override_config(self):
        mock_cfg = _mock_config_cls(breadth=6, depth=3, concurrency=4)
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            await run_deep_research(
                query="q", breadth=10, depth=5, concurrency_limit=8,
            )

        invoked_state = mock_g.ainvoke.call_args[0][0]
        assert invoked_state["breadth"] == 10
        assert invoked_state["depth"] == 5
        assert invoked_state["concurrency_limit"] == 8


# ---------------------------------------------------------------------------
# Callback / progress plumbing
# ---------------------------------------------------------------------------

class TestCallbackPlumbing:

    @pytest.mark.asyncio
    async def test_on_progress_passed_in_config(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()
        progress_cb = MagicMock()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            await run_deep_research(query="q", on_progress=progress_cb)

        invoked_config = mock_g.ainvoke.call_args[0][1]
        assert invoked_config["configurable"]["on_progress"] is progress_cb

    @pytest.mark.asyncio
    async def test_progress_tracker_in_config(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
            patch(f"{_MODULE}.ResearchProgress") as mock_progress_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            await run_deep_research(query="q", breadth=5, depth=3)

        mock_progress_cls.assert_called_once_with(total_depth=3, total_breadth=5)

        # Verify progress was passed in graph config
        config_arg = mock_g.ainvoke.call_args[1]["config"]
        assert config_arg["configurable"]["progress"] is mock_progress_cls.return_value


# ---------------------------------------------------------------------------
# Visited URLs handling
# ---------------------------------------------------------------------------

class TestVisitedUrls:

    @pytest.mark.asyncio
    async def test_visited_urls_set_converted_to_list(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            await run_deep_research(
                query="q",
                visited_urls={"https://a.com", "https://b.com"},
            )

        invoked_state = mock_g.ainvoke.call_args[0][0]
        assert isinstance(invoked_state["all_visited_urls"], list)
        assert set(invoked_state["all_visited_urls"]) == {"https://a.com", "https://b.com"}

    @pytest.mark.asyncio
    async def test_visited_urls_none_gives_empty_list(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            await run_deep_research(query="q", visited_urls=None)

        invoked_state = mock_g.ainvoke.call_args[0][0]
        assert invoked_state["all_visited_urls"] == []


# ---------------------------------------------------------------------------
# LLMService creation
# ---------------------------------------------------------------------------

class TestLLMServiceCreation:

    @pytest.mark.asyncio
    async def test_llm_service_created_with_config_and_callback(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler") as mock_cb_cls,
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            await run_deep_research(query="q")

        mock_llm_svc_cls.assert_called_once_with(mock_cfg, mock_cb_cls.return_value)

    @pytest.mark.asyncio
    async def test_llm_service_in_graph_config(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            await run_deep_research(query="q")

        config_arg = mock_g.ainvoke.call_args[1]["config"]
        assert config_arg["configurable"]["llm_service"] is mock_llm_svc_cls.return_value


# ---------------------------------------------------------------------------
# Usage summary
# ---------------------------------------------------------------------------

class TestUsageSummary:

    @pytest.mark.asyncio
    async def test_usage_summary_from_llm_service(self):
        mock_cfg = _mock_config_cls()
        mock_g = _mock_graph()
        expected_usage = {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700}

        with (
            patch(f"{_MODULE}.Config", return_value=mock_cfg),
            patch(f"{_MODULE}.build_deep_research_graph", return_value=mock_g),
            patch(f"{_MODULE}.TokenUsageCallbackHandler"),
            patch(f"{_MODULE}.LLMService") as mock_llm_svc_cls,
        ):
            mock_llm_svc_cls.return_value.get_usage_summary.return_value = expected_usage
            result = await run_deep_research(query="q")

        assert result["usage_summary"] == expected_usage


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------

class TestMainCLI:

    def test_main_calls_asyncio_run(self):
        mock_args = MagicMock()
        mock_args.query = "CLI test query"
        mock_args.breadth = 3
        mock_args.depth = 1
        mock_args.concurrency = 2
        mock_args.tone = "Analytical"
        mock_args.config_path = None

        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args

        with (
            patch("argparse.ArgumentParser", return_value=mock_parser),
            patch(f"{_MODULE}.asyncio.run") as mock_asyncio_run,
            patch(f"{_MODULE}.logging.basicConfig"),
        ):
            mock_asyncio_run.return_value = {
                "report": "CLI report",
                "usage_summary": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            main()

        mock_asyncio_run.assert_called_once()
        # Verify the coroutine passed to asyncio.run was created with correct args
        coro = mock_asyncio_run.call_args[0][0]
        # The coroutine object should exist (we can't easily inspect args,
        # but we verify asyncio.run was called exactly once)
        assert coro is not None

    def test_main_parses_arguments(self):
        mock_args = MagicMock()
        mock_args.query = "arg test"
        mock_args.breadth = None
        mock_args.depth = None
        mock_args.concurrency = None
        mock_args.tone = "Objective"
        mock_args.config_path = None

        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args

        with (
            patch("argparse.ArgumentParser", return_value=mock_parser),
            patch(f"{_MODULE}.asyncio.run", return_value={
                "report": "", "usage_summary": {},
            }),
            patch(f"{_MODULE}.logging.basicConfig"),
        ):
            main()

        mock_parser.parse_args.assert_called_once()
        # Verify arguments were registered
        add_argument_calls = [c[0][0] for c in mock_parser.add_argument.call_args_list]
        assert "query" in add_argument_calls
        assert "--breadth" in add_argument_calls
        assert "--depth" in add_argument_calls
        assert "--tone" in add_argument_calls
