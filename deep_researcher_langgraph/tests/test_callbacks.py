"""Tests for deep_researcher_langgraph.callbacks.TokenUsageCallbackHandler."""

import logging
from unittest.mock import MagicMock

import pytest

from deep_researcher_langgraph.callbacks import TokenUsageCallbackHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_result(*, llm_output=None, generation_infos=None):
    """Build a mock LLMResult with the desired token-usage paths.

    Args:
        llm_output: dict to place at ``response.llm_output`` (e.g.
            ``{"token_usage": {...}}``).
        generation_infos: list of dicts, each placed in
            ``response.generations[0][i].generation_info``.
    """
    result = MagicMock()
    result.llm_output = llm_output

    if generation_infos is not None:
        gens = []
        for info in generation_infos:
            gen = MagicMock()
            gen.generation_info = info
            gens.append(gen)
        result.generations = [gens]
    else:
        result.generations = [[]]

    return result


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInitialisation:

    def test_all_counters_zero(self, token_callback):
        assert token_callback.prompt_tokens == 0
        assert token_callback.completion_tokens == 0
        assert token_callback.total_tokens == 0
        assert token_callback.call_count == 0


# ---------------------------------------------------------------------------
# on_llm_end — token extraction
# ---------------------------------------------------------------------------

class TestOnLlmEnd:

    def test_extracts_from_llm_output(self, token_callback):
        result = _make_llm_result(
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            },
        )
        token_callback.on_llm_end(result)

        assert token_callback.prompt_tokens == 100
        assert token_callback.completion_tokens == 50
        assert token_callback.total_tokens == 150

    def test_extracts_from_generation_info(self, token_callback):
        result = _make_llm_result(
            generation_infos=[
                {
                    "token_usage": {
                        "prompt_tokens": 80,
                        "completion_tokens": 40,
                        "total_tokens": 120,
                    }
                }
            ],
        )
        token_callback.on_llm_end(result)

        assert token_callback.prompt_tokens == 80
        assert token_callback.completion_tokens == 40
        assert token_callback.total_tokens == 120

    def test_accumulates_across_multiple_calls(self, token_callback):
        for _ in range(3):
            result = _make_llm_result(
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    }
                },
            )
            token_callback.on_llm_end(result)

        assert token_callback.prompt_tokens == 30
        assert token_callback.completion_tokens == 15
        assert token_callback.total_tokens == 45
        assert token_callback.call_count == 3

    def test_llm_output_takes_precedence_over_generation_info(self, token_callback):
        """When both llm_output and generation_info carry usage, only
        llm_output is used to avoid double-counting."""
        result = _make_llm_result(
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            },
            generation_infos=[
                {
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    }
                }
            ],
        )
        token_callback.on_llm_end(result)

        assert token_callback.prompt_tokens == 100
        assert token_callback.completion_tokens == 50
        assert token_callback.total_tokens == 150


# ---------------------------------------------------------------------------
# on_llm_error
# ---------------------------------------------------------------------------

class TestOnLlmError:

    def test_logs_warning_without_crash(self, token_callback, caplog):
        error = RuntimeError("something went wrong")
        with caplog.at_level(logging.WARNING):
            token_callback.on_llm_error(error)
        assert "something went wrong" in caplog.text


# ---------------------------------------------------------------------------
# call_count property
# ---------------------------------------------------------------------------

class TestCallCount:

    def test_increments_per_on_llm_end(self, token_callback):
        empty = _make_llm_result()
        token_callback.on_llm_end(empty)
        token_callback.on_llm_end(empty)
        assert token_callback.call_count == 2


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------

class TestGetSummary:

    def test_returns_correct_structure(self, token_callback):
        result = _make_llm_result(
            llm_output={
                "token_usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                }
            },
        )
        token_callback.on_llm_end(result)

        summary = token_callback.get_summary()

        assert summary == {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
            "call_count": 1,
            "error_count": 0,
            "errors": [],
        }


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:

    def test_clears_all_counters(self, token_callback):
        # Accumulate some usage first
        result = _make_llm_result(
            llm_output={
                "token_usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25,
                    "total_tokens": 75,
                }
            },
        )
        token_callback.on_llm_end(result)

        token_callback.reset()

        assert token_callback.prompt_tokens == 0
        assert token_callback.completion_tokens == 0
        assert token_callback.total_tokens == 0
        assert token_callback.call_count == 0


