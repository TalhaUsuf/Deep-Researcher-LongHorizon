"""Tests for deep_researcher_langgraph state and schema definitions."""

import pytest
from pydantic import ValidationError

from deep_researcher_langgraph.state import DeepResearchState, ResearchProgress
from deep_researcher_langgraph.schemas import (
    SearchQueryItem,
    SearchQueriesResponse,
    FollowUpQuestionsResponse,
    LearningItem,
    ResearchAnalysis,
)


# ---------------------------------------------------------------------------
# ResearchProgress
# ---------------------------------------------------------------------------

class TestResearchProgressDefaults:
    """Verify ResearchProgress initialises with the documented defaults."""

    def test_current_depth_starts_at_one(self, progress):
        assert progress.current_depth == 1

    def test_current_breadth_starts_at_zero(self, progress):
        assert progress.current_breadth == 0

    def test_current_query_is_none(self, progress):
        assert progress.current_query is None

    def test_total_queries_starts_at_zero(self, progress):
        assert progress.total_queries == 0

    def test_completed_queries_starts_at_zero(self, progress):
        assert progress.completed_queries == 0

    def test_total_depth_set_from_constructor(self, progress):
        assert progress.total_depth == 2

    def test_total_breadth_set_from_constructor(self, progress):
        assert progress.total_breadth == 4


class TestResearchProgressUpdates:
    """Verify mutable fields can be updated."""

    def test_update_current_depth(self, progress):
        progress.current_depth = 3
        assert progress.current_depth == 3

    def test_update_current_query(self, progress):
        progress.current_query = "quantum computing breakthroughs"
        assert progress.current_query == "quantum computing breakthroughs"

    def test_update_completed_queries(self, progress):
        progress.total_queries = 10
        progress.completed_queries = 5
        assert progress.total_queries == 10
        assert progress.completed_queries == 5

    def test_update_current_breadth(self, progress):
        progress.current_breadth = 3
        assert progress.current_breadth == 3


# ---------------------------------------------------------------------------
# SearchQueriesResponse
# ---------------------------------------------------------------------------

class TestSearchQueriesResponse:

    def test_valid_data(self):
        data = {
            "queries": [
                {"query": "latest quantum computing papers", "research_goal": "find recent results"},
                {"query": "quantum error correction", "research_goal": "understand error rates"},
            ]
        }
        resp = SearchQueriesResponse(**data)
        assert len(resp.queries) == 2
        assert resp.queries[0].query == "latest quantum computing papers"

    def test_empty_queries_list_is_valid(self):
        resp = SearchQueriesResponse(queries=[])
        assert resp.queries == []

    def test_missing_queries_field_raises(self):
        with pytest.raises(ValidationError):
            SearchQueriesResponse()  # type: ignore[call-arg]

    def test_query_item_missing_research_goal_raises(self):
        with pytest.raises(ValidationError):
            SearchQueriesResponse(queries=[{"query": "test"}])  # type: ignore[list-item]

    def test_query_item_missing_query_raises(self):
        with pytest.raises(ValidationError):
            SearchQueriesResponse(queries=[{"research_goal": "goal only"}])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# FollowUpQuestionsResponse
# ---------------------------------------------------------------------------

class TestFollowUpQuestionsResponse:

    def test_valid_data(self):
        resp = FollowUpQuestionsResponse(questions=["What about X?", "How does Y work?"])
        assert len(resp.questions) == 2

    def test_empty_questions_valid(self):
        resp = FollowUpQuestionsResponse(questions=[])
        assert resp.questions == []

    def test_missing_questions_raises(self):
        with pytest.raises(ValidationError):
            FollowUpQuestionsResponse()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# LearningItem
# ---------------------------------------------------------------------------

class TestLearningItem:

    def test_default_source_url_is_empty_string(self):
        item = LearningItem(insight="Quantum advantage demonstrated")
        assert item.source_url == ""

    def test_explicit_source_url(self):
        item = LearningItem(insight="insight", source_url="https://example.com")
        assert item.source_url == "https://example.com"

    def test_missing_insight_raises(self):
        with pytest.raises(ValidationError):
            LearningItem()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ResearchAnalysis
# ---------------------------------------------------------------------------

class TestResearchAnalysis:

    def test_valid_with_source_urls(self):
        analysis = ResearchAnalysis(
            learnings=[
                LearningItem(insight="insight A", source_url="https://a.com"),
                LearningItem(insight="insight B", source_url="https://b.com"),
            ],
            follow_up_questions=["Why?"],
        )
        assert len(analysis.learnings) == 2
        assert analysis.learnings[0].source_url == "https://a.com"

    def test_valid_without_source_urls(self):
        analysis = ResearchAnalysis(
            learnings=[LearningItem(insight="insight C")],
            follow_up_questions=[],
        )
        assert analysis.learnings[0].source_url == ""

    def test_missing_learnings_raises(self):
        with pytest.raises(ValidationError):
            ResearchAnalysis(follow_up_questions=["q"])  # type: ignore[call-arg]

    def test_missing_follow_up_questions_raises(self):
        with pytest.raises(ValidationError):
            ResearchAnalysis(
                learnings=[LearningItem(insight="x")]
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# DeepResearchState (TypedDict — instantiated as plain dict)
# ---------------------------------------------------------------------------

class TestDeepResearchState:

    def test_base_state_has_all_keys(self, base_state):
        """base_state fixture should contain every key defined in the TypedDict."""
        expected_keys = {
            "query", "breadth", "depth", "concurrency_limit", "tone",
            "config_path", "headers", "websocket", "mcp_configs", "mcp_strategy",
            "on_progress",
            "initial_search_results", "follow_up_questions", "combined_query",
            "current_depth", "current_breadth", "search_queries", "research_results",
            "branch_stack",
            "all_learnings", "all_citations", "all_visited_urls", "all_context", "all_sources",
            "total_queries", "completed_queries",
            "final_context", "report",
            "messages",
        }
        assert set(base_state.keys()) == expected_keys

    def test_base_state_is_dict(self, base_state):
        assert isinstance(base_state, dict)

    @pytest.mark.parametrize(
        "key, expected",
        [
            ("query", "What are the latest advances in quantum computing?"),
            ("breadth", 4),
            ("depth", 2),
            ("report", ""),
        ],
    )
    def test_base_state_default_values(self, base_state, key, expected):
        assert base_state[key] == expected
