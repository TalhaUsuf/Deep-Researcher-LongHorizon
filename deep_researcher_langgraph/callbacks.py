"""LangChain callback handler for tracking LLM token usage."""

from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
import logging

logger = logging.getLogger(__name__)


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Tracks token usage across all LLM calls via LangChain callbacks."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self._call_count: int = 0
        self._error_count: int = 0
        self._errors: List[str] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if response.llm_output and "token_usage" in response.llm_output:
            self._accumulate_usage(response.llm_output["token_usage"])
        else:
            for generations in response.generations:
                for gen in generations:
                    if gen.generation_info and "token_usage" in gen.generation_info:
                        self._accumulate_usage(gen.generation_info["token_usage"])

        self._call_count += 1

    def _accumulate_usage(self, usage: Dict[str, Any]) -> None:
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._error_count += 1
        self._errors.append(str(error))
        logger.error(f"LLM call failed ({self._error_count} total failures): {error}")

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def get_summary(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "errors": self._errors[-10:],
        }

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._call_count = 0
        self._error_count = 0
        self._errors = []
