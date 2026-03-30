"""ChatPromptTemplate definitions for deep research workflow."""

from langchain_core.prompts import ChatPromptTemplate


GENERATE_SEARCH_QUERIES_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert researcher generating search queries."
    ),
    (
        "human",
        "Given the following prompt, generate {num_queries} unique search queries "
        "to research the topic thoroughly. For each query, provide a research goal.\n\n"
        "Topic: {query}"
    ),
])


GENERATE_RESEARCH_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert researcher. Your task is to analyze the original query "
        "and search results, then generate targeted questions that explore different "
        "aspects and time periods of the topic."
    ),
    (
        "human",
        "Original query: {query}\n\n"
        "Current time: {current_time}\n\n"
        "Search results:\n{search_results}\n\n"
        "Based on these results, the original query, and the current time, generate "
        "{num_questions} unique questions. Each question should explore a different "
        "aspect or time period of the topic, considering recent developments up to "
        "{current_time}."
    ),
])


PROCESS_RESEARCH_RESULTS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert researcher analyzing search results. "
        "Extract key learnings with source citations and suggest follow-up questions."
    ),
    (
        "human",
        "Given the following research results for the query '{query}', extract key "
        "learnings and suggest follow-up questions. For each learning, include a "
        "citation to the source URL if available.\n\n"
        "Research context:\n{context}"
    ),
])


GENERATE_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research report writer. Write detailed, factual, and "
        "unbiased reports based on provided research context and citations."
    ),
    (
        "human",
        "Based on the following research query and context, write a comprehensive, "
        "detailed research report. Include all relevant citations as markdown links.\n\n"
        "Query: {query}\n\n"
        "Research Context:\n{context}\n\n"
        "Tone: {tone}\n\n"
        "Write a well-structured report with introduction, main sections, and "
        "conclusion. The report should be at least 1200 words and include references."
    ),
])
