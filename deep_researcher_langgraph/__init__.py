"""LangGraph-based Deep Research implementation.

This package re-implements the deep research workflow from
gpt_researcher/skills/deep_research.py using LangGraph primitives:
  - StateGraph for workflow orchestration
  - ChatPromptTemplate for prompt management
  - with_structured_output for structured LLM responses
  - LangChain callback handlers for token usage tracking
  - A unified LLMService for all LLM operations
"""

from .graph import build_deep_research_graph
from .llm_service import LLMService
from .main import run_deep_research
from .state import BranchItem, DeepResearchState, ResearchProgress

__all__ = [
    "BranchItem",
    "build_deep_research_graph",
    "DeepResearchState",
    "LLMService",
    "ResearchProgress",
    "run_deep_research",
]
