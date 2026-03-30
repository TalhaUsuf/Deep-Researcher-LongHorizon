"""Unified AI service for all LLM operations in the deep research workflow.

All LLM usage in this package should go through this service to ensure
consistent model selection, callback tracking, and configuration.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from gpt_researcher.config.config import Config
from gpt_researcher.llm_provider.generic.base import (
    GenericLLMProvider,
    NO_SUPPORT_TEMPERATURE_MODELS,
    SUPPORT_REASONING_EFFORT_MODELS,
)

from .callbacks import TokenUsageCallbackHandler

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Unified AI service that provides LLM instances for deep research.

    All nodes in the LangGraph workflow obtain their LLM through this service,
    ensuring consistent configuration, model selection, and token usage tracking.
    """

    def __init__(self, config: Config, callback_handler: Optional[TokenUsageCallbackHandler] = None):
        self._config = config
        self._callback_handler = callback_handler or TokenUsageCallbackHandler()
        self._model_cache: Dict[str, BaseChatModel] = {}

    @classmethod
    def from_config(cls, config: Config) -> "LLMService":
        return cls(config)

    @property
    def callback_handler(self) -> TokenUsageCallbackHandler:
        return self._callback_handler

    @property
    def reasoning_effort(self) -> str:
        return getattr(self._config, "reasoning_effort", "medium")

    def _build_llm(
        self,
        provider: str,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        # Build unique cache key including parameters that affect the instance
        cache_key = f"{provider}:{model}:t={temperature}:mt={max_tokens}:re={reasoning_effort}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        provider_kwargs: Dict[str, Any] = {"model": model}
        provider_kwargs.update(self._config.llm_kwargs or {})

        # Apply temperature and max_tokens only for models that support them
        if model not in NO_SUPPORT_TEMPERATURE_MODELS:
            if temperature is not None:
                provider_kwargs["temperature"] = temperature
            if max_tokens is not None:
                provider_kwargs["max_tokens"] = max_tokens

        # Apply reasoning effort for supported models
        if model in SUPPORT_REASONING_EFFORT_MODELS and reasoning_effort:
            provider_kwargs["reasoning_effort"] = reasoning_effort

        provider_kwargs.update(kwargs)

        llm_provider = GenericLLMProvider.from_provider(provider, **provider_kwargs)
        llm = llm_provider.llm

        self._model_cache[cache_key] = llm
        return llm

    def get_strategic_llm(
        self,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """LLM for complex reasoning: query generation, research analysis."""
        return self._build_llm(
            self._config.strategic_llm_provider,
            self._config.strategic_llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort or self.reasoning_effort,
            **kwargs,
        )

    def get_smart_llm(
        self,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """LLM for report writing and general tasks."""
        return self._build_llm(
            self._config.smart_llm_provider,
            self._config.smart_llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_fast_llm(
        self,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """LLM for quick, low-cost operations."""
        return self._build_llm(
            self._config.fast_llm_provider,
            self._config.fast_llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_callbacks(self) -> List[BaseCallbackHandler]:
        """Return callback list including the token usage tracker."""
        return [self._callback_handler]

    async def invoke_structured(
        self,
        llm: BaseChatModel,
        schema: Type[T],
        messages: list,
    ) -> T:
        """Invoke an LLM with with_structured_output and callback tracking.

        Args:
            llm: The base LLM to use.
            schema: Pydantic model class for structured output.
            messages: Formatted prompt messages.

        Returns:
            An instance of the schema populated by the LLM response.
        """
        structured_llm = llm.with_structured_output(schema)
        return await structured_llm.ainvoke(messages, config={"callbacks": self.get_callbacks()})

    async def invoke(self, llm: BaseChatModel, messages: list) -> str:
        """Invoke an LLM and return the string response with callback tracking."""
        response = await llm.ainvoke(messages, config={"callbacks": self.get_callbacks()})
        return response.content

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return accumulated token usage summary."""
        return self._callback_handler.get_summary()
