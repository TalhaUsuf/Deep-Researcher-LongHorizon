"""Pydantic schemas for structured LLM outputs via with_structured_output."""

from pydantic import BaseModel, Field
from typing import List


class SearchQueryItem(BaseModel):
    """A search query paired with its research goal."""
    query: str = Field(description="A specific search query to research the topic")
    research_goal: str = Field(description="The research goal this query aims to achieve")


class SearchQueriesResponse(BaseModel):
    """Collection of search queries generated for a research topic."""
    queries: List[SearchQueryItem] = Field(
        description="List of unique search queries with research goals"
    )


class FollowUpQuestionsResponse(BaseModel):
    """Follow-up questions generated from initial search results."""
    questions: List[str] = Field(
        description="Targeted questions exploring different aspects or time periods of the topic"
    )


class LearningItem(BaseModel):
    """A single learning/insight extracted from research results."""
    insight: str = Field(description="Key learning or insight from the research")
    source_url: str = Field(default="", description="Source URL for this learning, if available")


class ResearchAnalysis(BaseModel):
    """Analysis of research results: learnings and follow-up questions."""
    learnings: List[LearningItem] = Field(
        description="Key learnings extracted from the research context"
    )
    follow_up_questions: List[str] = Field(
        description="Suggested follow-up questions for deeper research"
    )
