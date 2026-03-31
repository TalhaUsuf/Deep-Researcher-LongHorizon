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
        "Extract detailed, factual learnings with source citations and suggest follow-up questions.\n\n"
        "Guidelines for extracting learnings:\n"
        "- Extract 8-15 distinct learnings from the research context.\n"
        "- Focus on specific facts, statistics, benchmark numbers, quantitative comparisons, "
        "named methods/techniques, dates, and concrete findings.\n"
        "- Preserve exact numbers, percentages, measurements, and benchmark scores verbatim from the source.\n"
        "- Each insight should be a self-contained factual statement of 1-3 sentences, "
        "detailed enough to be useful without the original source.\n"
        "- Always include the source URL for each learning when available.\n"
        "- Do NOT paraphrase numbers or replace specific data with vague qualifiers like 'significant' or 'high'."
    ),
    (
        "human",
        "Given the following research results for the query '{query}', extract detailed "
        "learnings and suggest follow-up questions. For each learning, preserve all "
        "specific data points and include a citation to the source URL.\n\n"
        "Research context:\n{context}"
    ),
])


GENERATE_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research report writer. Write detailed, factual, and "
        "unbiased reports based on provided research context and citations. "
        "You MUST determine your own concrete and valid opinion based on the given "
        "information. Do NOT defer to general and meaningless conclusions."
    ),
    (
        "human",
        'Information: """{context}"""\n'
        "---\n"
        "Using the above information, answer the following query or task: "
        '"{query}" in a detailed report.\n\n'
        "The report should focus on the answer to the query, should be well structured, "
        "informative, in-depth, and comprehensive, with facts and numbers if available "
        "and at least 1200 words.\n"
        "You should strive to write the report as long as you can using all relevant "
        "and necessary information provided.\n\n"
        "Please follow all of the following guidelines in your report:\n"
        "- You MUST write the report with markdown syntax.\n"
        "- Structure your report with clear markdown headers: use # for the main title, "
        "## for major sections, and ### for subsections.\n"
        "- Use markdown tables when presenting structured data or comparisons to enhance readability.\n"
        "- You MUST prioritize the relevance, reliability, and significance of the sources you use.\n"
        "- You must also prioritize new articles over older articles if the source can be trusted.\n"
        "- Use in-text citation references with markdown hyperlinks placed at the end of the "
        "sentence or paragraph that references them like this: ([source title](url)).\n"
        "- You MUST write all used source URLs at the end of the report as references, "
        "and make sure to not add duplicated sources, but only one reference for each.\n"
        "- Every URL should be hyperlinked: [url website](url)\n"
        "- Write the report in a {tone} tone.\n"
        "- Assume that the current date is {current_date}.\n"
        "Please do your best, this is very important to my career."
    ),
])
