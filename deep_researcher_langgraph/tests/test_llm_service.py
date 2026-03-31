"""Tests for deep_researcher_langgraph.llm_service module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from deep_researcher_langgraph.callbacks import TokenUsageCallbackHandler
from deep_researcher_langgraph.llm_service import LLMService


MODULE = "deep_researcher_langgraph.llm_service"


# ---------------------------------------------------------------------------
# __init__ and from_config
# ---------------------------------------------------------------------------


class TestInit:
    def test_stores_config_and_callback(self, mock_config, token_callback):
        service = LLMService(mock_config, token_callback)
        assert service._config is mock_config
        assert service._callback_handler is token_callback

    def test_creates_default_callback_when_none(self, mock_config):
        service = LLMService(mock_config)
        assert isinstance(service._callback_handler, TokenUsageCallbackHandler)

    def test_from_config_creates_instance(self, mock_config):
        service = LLMService.from_config(mock_config)
        assert isinstance(service, LLMService)
        assert service._config is mock_config


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_callback_handler_returns_injected_handler(self, llm_service, token_callback):
        assert llm_service.callback_handler is token_callback

    def test_reasoning_effort_reads_from_config(self, llm_service):
        assert llm_service.reasoning_effort == "medium"

    def test_reasoning_effort_defaults_when_missing(self, mock_config, token_callback):
        del mock_config.reasoning_effort
        mock_config.configure_mock(**{})
        # Make getattr fall through to default
        type(mock_config).reasoning_effort = PropertyMock(
            side_effect=AttributeError
        )
        service = LLMService(mock_config, token_callback)
        assert service.reasoning_effort == "medium"


# ---------------------------------------------------------------------------
# _build_llm
# ---------------------------------------------------------------------------


class TestBuildLlm:
    @patch(f"{MODULE}.GenericLLMProvider")
    def test_calls_from_provider_with_correct_args(self, mock_provider_cls, llm_service):
        mock_llm_instance = MagicMock()
        mock_provider_cls.from_provider.return_value.llm = mock_llm_instance

        result = llm_service._build_llm("openai", "gpt-4o", temperature=0.5)

        mock_provider_cls.from_provider.assert_called_once()
        call_kwargs = mock_provider_cls.from_provider.call_args
        assert call_kwargs[0][0] == "openai"
        assert call_kwargs[1]["model"] == "gpt-4o"
        assert call_kwargs[1]["temperature"] == 0.5
        assert result is mock_llm_instance

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_caching_same_params_returns_cached(self, mock_provider_cls, llm_service):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        first = llm_service._build_llm("openai", "gpt-4o", temperature=0.4)
        second = llm_service._build_llm("openai", "gpt-4o", temperature=0.4)

        assert first is second
        assert mock_provider_cls.from_provider.call_count == 1

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_caching_different_params_creates_new(self, mock_provider_cls, llm_service):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        first = llm_service._build_llm("openai", "gpt-4o", temperature=0.4)
        # Different temperature -> different cache key
        mock_provider_cls.from_provider.return_value.llm = MagicMock()
        second = llm_service._build_llm("openai", "gpt-4o", temperature=0.9)

        assert first is not second
        assert mock_provider_cls.from_provider.call_count == 2

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_passes_temperature_when_model_supports_it(self, mock_provider_cls, llm_service):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        llm_service._build_llm("openai", "gpt-4o", temperature=0.7)

        call_kwargs = mock_provider_cls.from_provider.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @patch(f"{MODULE}.NO_SUPPORT_TEMPERATURE_MODELS", ["o1-mini"])
    @patch(f"{MODULE}.GenericLLMProvider")
    def test_skips_temperature_for_unsupported_models(self, mock_provider_cls, llm_service):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        llm_service._build_llm("openai", "o1-mini", temperature=0.7, max_tokens=1000)

        call_kwargs = mock_provider_cls.from_provider.call_args[1]
        assert "temperature" not in call_kwargs
        assert "max_tokens" not in call_kwargs

    @patch(f"{MODULE}.SUPPORT_REASONING_EFFORT_MODELS", ["o3-mini"])
    @patch(f"{MODULE}.GenericLLMProvider")
    def test_passes_reasoning_effort_for_supported_models(self, mock_provider_cls, llm_service):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        llm_service._build_llm("openai", "o3-mini", reasoning_effort="high")

        call_kwargs = mock_provider_cls.from_provider.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    @patch(f"{MODULE}.SUPPORT_REASONING_EFFORT_MODELS", [])
    @patch(f"{MODULE}.GenericLLMProvider")
    def test_skips_reasoning_effort_for_unsupported_models(self, mock_provider_cls, llm_service):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        llm_service._build_llm("openai", "gpt-4o", reasoning_effort="high")

        call_kwargs = mock_provider_cls.from_provider.call_args[1]
        assert "reasoning_effort" not in call_kwargs


# ---------------------------------------------------------------------------
# get_strategic_llm / get_smart_llm / get_fast_llm
# ---------------------------------------------------------------------------


class TestLlmGetters:
    @patch(f"{MODULE}.GenericLLMProvider")
    def test_get_strategic_llm_uses_strategic_config(self, mock_provider_cls, llm_service, mock_config):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        llm_service.get_strategic_llm()

        call_args = mock_provider_cls.from_provider.call_args
        assert call_args[0][0] == mock_config.strategic_llm_provider
        assert call_args[1]["model"] == mock_config.strategic_llm_model

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_get_smart_llm_uses_smart_config(self, mock_provider_cls, llm_service, mock_config):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        llm_service.get_smart_llm()

        call_args = mock_provider_cls.from_provider.call_args
        assert call_args[0][0] == mock_config.smart_llm_provider
        assert call_args[1]["model"] == mock_config.smart_llm_model

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_get_fast_llm_uses_fast_config(self, mock_provider_cls, llm_service, mock_config):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        llm_service.get_fast_llm()

        call_args = mock_provider_cls.from_provider.call_args
        assert call_args[0][0] == mock_config.fast_llm_provider
        assert call_args[1]["model"] == mock_config.fast_llm_model

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_get_strategic_llm_passes_base_url(self, mock_provider_cls, mock_config, token_callback):
        mock_config.strategic_llm_base_url = "http://example.com/v1"
        service = LLMService(mock_config, token_callback)
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        service.get_strategic_llm()

        call_kwargs = mock_provider_cls.from_provider.call_args[1]
        assert call_kwargs["openai_api_base"] == "http://example.com/v1"

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_get_smart_llm_passes_base_url(self, mock_provider_cls, mock_config, token_callback):
        mock_config.smart_llm_base_url = "http://example.com/v1"
        service = LLMService(mock_config, token_callback)
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        service.get_smart_llm()

        call_kwargs = mock_provider_cls.from_provider.call_args[1]
        assert call_kwargs["openai_api_base"] == "http://example.com/v1"

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_get_fast_llm_passes_base_url(self, mock_provider_cls, mock_config, token_callback):
        mock_config.fast_llm_base_url = "http://example.com/v1"
        service = LLMService(mock_config, token_callback)
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        service.get_fast_llm()

        call_kwargs = mock_provider_cls.from_provider.call_args[1]
        assert call_kwargs["openai_api_base"] == "http://example.com/v1"

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_base_url_not_passed_when_none(self, mock_provider_cls, llm_service):
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        llm_service.get_fast_llm()

        call_kwargs = mock_provider_cls.from_provider.call_args[1]
        assert "openai_api_base" not in call_kwargs

    @patch(f"{MODULE}.GenericLLMProvider")
    def test_explicit_kwarg_overrides_config_base_url(self, mock_provider_cls, mock_config, token_callback):
        mock_config.fast_llm_base_url = "http://config-url/v1"
        service = LLMService(mock_config, token_callback)
        mock_provider_cls.from_provider.return_value.llm = MagicMock()

        service.get_fast_llm(openai_api_base="http://override-url/v1")

        call_kwargs = mock_provider_cls.from_provider.call_args[1]
        assert call_kwargs["openai_api_base"] == "http://override-url/v1"


# ---------------------------------------------------------------------------
# get_callbacks
# ---------------------------------------------------------------------------


class TestGetCallbacks:
    def test_returns_list_with_callback_handler(self, llm_service, token_callback):
        callbacks = llm_service.get_callbacks()
        assert callbacks == [token_callback]


# ---------------------------------------------------------------------------
# invoke_structured / invoke
# ---------------------------------------------------------------------------


class TestInvoke:
    @pytest.mark.asyncio
    async def test_invoke_structured(self, llm_service, mock_llm):
        schema = MagicMock()
        messages = [("human", "hello")]
        expected_result = MagicMock()

        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=expected_result)
        mock_llm.with_structured_output.return_value = structured_llm

        result = await llm_service.invoke_structured(mock_llm, schema, messages)

        mock_llm.with_structured_output.assert_called_once_with(schema)
        structured_llm.ainvoke.assert_awaited_once()
        call_kwargs = structured_llm.ainvoke.call_args
        assert call_kwargs[0][0] is messages
        assert "callbacks" in call_kwargs[1]["config"]
        assert result is expected_result

    @pytest.mark.asyncio
    async def test_invoke_returns_content(self, llm_service, mock_llm):
        response = MagicMock()
        response.content = "LLM response text"
        mock_llm.ainvoke = AsyncMock(return_value=response)

        result = await llm_service.invoke(mock_llm, [("human", "hi")])

        assert result == "LLM response text"
        mock_llm.ainvoke.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_usage_summary
# ---------------------------------------------------------------------------


class TestGetUsageSummary:
    def test_delegates_to_callback_handler(self, llm_service, token_callback):
        summary = llm_service.get_usage_summary()
        assert summary == token_callback.get_summary()
        assert "prompt_tokens" in summary
        assert "call_count" in summary
