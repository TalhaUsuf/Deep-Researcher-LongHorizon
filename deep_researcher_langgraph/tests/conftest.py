"""Shared pytest fixtures for deep_researcher_langgraph tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def pytest_collection_modifyitems(items):
    """Automatically apply pytest.mark.asyncio to all async test functions.

    This avoids reliance on the global asyncio_mode setting in pyproject.toml,
    which is configured as 'strict' for the top-level test suite but may not
    be compatible with the installed pytest-asyncio version (1.3.0).
    """
    for item in items:
        if isinstance(item, pytest.Function) and item.get_closest_marker("asyncio") is None:
            if hasattr(item.obj, "__wrapped__"):
                obj = item.obj.__wrapped__
            else:
                obj = item.obj
            if hasattr(obj, "__code__") and obj.__code__.co_flags & 0x100:  # CO_COROUTINE
                item.add_marker(pytest.mark.asyncio)

from deep_researcher_langgraph.callbacks import TokenUsageCallbackHandler
from deep_researcher_langgraph.llm_service import LLMService
from deep_researcher_langgraph.state import DeepResearchState, ResearchProgress


@pytest.fixture
def token_callback():
    """Fresh TokenUsageCallbackHandler instance."""
    return TokenUsageCallbackHandler()


@pytest.fixture
def mock_config():
    """Mock gpt_researcher Config object with standard defaults."""
    cfg = MagicMock()
    cfg.strategic_llm_provider = "openai"
    cfg.strategic_llm_model = "gpt-4o"
    cfg.smart_llm_provider = "openai"
    cfg.smart_llm_model = "gpt-4o"
    cfg.fast_llm_provider = "openai"
    cfg.fast_llm_model = "gpt-4o-mini"
    cfg.llm_kwargs = {}
    cfg.reasoning_effort = "medium"
    cfg.fast_llm_base_url = None
    cfg.smart_llm_base_url = None
    cfg.strategic_llm_base_url = None
    cfg.strategic_token_limit = 4000
    cfg.smart_token_limit = 6000
    cfg.deep_research_breadth = 4
    cfg.deep_research_depth = 2
    cfg.deep_research_concurrency = 2
    cfg.config_path = None
    cfg.retrievers = ["tavily"]
    return cfg


@pytest.fixture
def llm_service(mock_config, token_callback):
    """LLMService wired with mock config and callback handler."""
    return LLMService(mock_config, token_callback)


@pytest.fixture
def mock_llm():
    """Mock BaseChatModel that returns a predictable response."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.with_structured_output = MagicMock()
    return llm


@pytest.fixture
def progress():
    """ResearchProgress tracker for tests."""
    return ResearchProgress(total_depth=2, total_breadth=4)


@pytest.fixture
def base_state() -> DeepResearchState:
    """Minimal valid DeepResearchState for node tests."""
    return {
        "query": "What are the latest advances in quantum computing?",
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


@pytest.fixture
def graph_config(llm_service, progress, mock_config):
    """LangGraph runnable config with LLMService and progress."""
    return {
        "configurable": {
            "llm_service": llm_service,
            "progress": progress,
            "config": mock_config,
            "thread_id": "test-thread",
        }
    }
