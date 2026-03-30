"""Tests for deep_researcher_langgraph.prompts module."""

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from deep_researcher_langgraph.prompts import (
    GENERATE_SEARCH_QUERIES_PROMPT,
    GENERATE_RESEARCH_PLAN_PROMPT,
    PROCESS_RESEARCH_RESULTS_PROMPT,
    GENERATE_REPORT_PROMPT,
)


ALL_PROMPTS = [
    GENERATE_SEARCH_QUERIES_PROMPT,
    GENERATE_RESEARCH_PLAN_PROMPT,
    PROCESS_RESEARCH_RESULTS_PROMPT,
    GENERATE_REPORT_PROMPT,
]


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------


class TestPromptTypes:
    @pytest.mark.parametrize("prompt", ALL_PROMPTS)
    def test_is_chat_prompt_template(self, prompt):
        assert isinstance(prompt, ChatPromptTemplate)

    @pytest.mark.parametrize("prompt", ALL_PROMPTS)
    def test_has_exactly_two_messages(self, prompt):
        assert len(prompt.messages) == 2

    @pytest.mark.parametrize("prompt", ALL_PROMPTS)
    def test_first_message_is_system(self, prompt):
        # MessagePromptTemplate stores the role; format with dummy vars to check
        # Use the message_prompt attribute to inspect the role
        msg_template = prompt.messages[0]
        # ChatPromptTemplate message templates expose a `.prompt` and role info
        # The easiest reliable check: format with dummy values and inspect type
        dummy_vars = {v: "dummy" for v in prompt.input_variables}
        messages = prompt.format_messages(**dummy_vars)
        assert isinstance(messages[0], SystemMessage)

    @pytest.mark.parametrize("prompt", ALL_PROMPTS)
    def test_second_message_is_human(self, prompt):
        dummy_vars = {v: "dummy" for v in prompt.input_variables}
        messages = prompt.format_messages(**dummy_vars)
        assert isinstance(messages[1], HumanMessage)


# ---------------------------------------------------------------------------
# GENERATE_SEARCH_QUERIES_PROMPT
# ---------------------------------------------------------------------------


class TestGenerateSearchQueriesPrompt:
    def test_format_messages_produces_correct_types(self):
        messages = GENERATE_SEARCH_QUERIES_PROMPT.format_messages(
            query="quantum computing",
            num_queries=3,
        )
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_format_messages_includes_variables(self):
        messages = GENERATE_SEARCH_QUERIES_PROMPT.format_messages(
            query="quantum computing",
            num_queries=3,
        )
        human_content = messages[1].content
        assert "quantum computing" in human_content
        assert "3" in human_content


# ---------------------------------------------------------------------------
# GENERATE_RESEARCH_PLAN_PROMPT
# ---------------------------------------------------------------------------


class TestGenerateResearchPlanPrompt:
    def test_format_messages_includes_all_variables(self):
        messages = GENERATE_RESEARCH_PLAN_PROMPT.format_messages(
            query="AI safety",
            current_time="2026-03-30",
            search_results="Result 1\nResult 2",
            num_questions=5,
        )
        assert len(messages) == 2
        human_content = messages[1].content
        assert "AI safety" in human_content
        assert "2026-03-30" in human_content
        assert "Result 1" in human_content
        assert "5" in human_content


# ---------------------------------------------------------------------------
# PROCESS_RESEARCH_RESULTS_PROMPT
# ---------------------------------------------------------------------------


class TestProcessResearchResultsPrompt:
    def test_format_messages_includes_query_and_context(self):
        messages = PROCESS_RESEARCH_RESULTS_PROMPT.format_messages(
            query="climate change",
            context="Some research findings here.",
        )
        assert len(messages) == 2
        human_content = messages[1].content
        assert "climate change" in human_content
        assert "Some research findings here." in human_content


# ---------------------------------------------------------------------------
# GENERATE_REPORT_PROMPT
# ---------------------------------------------------------------------------


class TestGenerateReportPrompt:
    def test_format_messages_includes_query_context_tone(self):
        messages = GENERATE_REPORT_PROMPT.format_messages(
            query="renewable energy",
            context="Solar and wind data.",
            tone="Objective",
        )
        assert len(messages) == 2
        human_content = messages[1].content
        assert "renewable energy" in human_content
        assert "Solar and wind data." in human_content
        assert "Objective" in human_content


# ---------------------------------------------------------------------------
# Missing variables
# ---------------------------------------------------------------------------


class TestMissingVariables:
    def test_search_queries_missing_variable_raises(self):
        with pytest.raises(KeyError):
            GENERATE_SEARCH_QUERIES_PROMPT.format_messages(query="test")

    def test_research_plan_missing_variable_raises(self):
        with pytest.raises(KeyError):
            GENERATE_RESEARCH_PLAN_PROMPT.format_messages(query="test")

    def test_process_results_missing_variable_raises(self):
        with pytest.raises(KeyError):
            PROCESS_RESEARCH_RESULTS_PROMPT.format_messages(query="test")

    def test_report_missing_variable_raises(self):
        with pytest.raises(KeyError):
            GENERATE_REPORT_PROMPT.format_messages(query="test")
