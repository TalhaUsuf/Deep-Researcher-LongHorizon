# Deep Researcher LangGraph — Implementation Plan

## Architecture Overview

This is a **LangGraph StateGraph** re-implementation of the original `gpt_researcher/skills/deep_research.py`, replacing recursive `asyncio` calls with a stack-based DFS workflow.

## Graph Flow

```
generate_research_plan → generate_search_queries → execute_research
                                    ↑                      |
                                    |            should_continue_deeper?
                                    |              /              \
                                    |        "go_deeper"      "check_stack"
                                    |            |                 |
                                    |     fan_out_branches    assemble_final_context
                                    |            |                 |
                                    |       has_more_work?    has_more_work?
                                    |        /       \         /       \
                                    |   "next"    "done"  "next"    "done"
                                    |     |          |      |         |
                                    +-- pick_next_branch    |    generate_report → END
                                              ↑             |
                                              +-------------+
```

## Module Breakdown

| File | Purpose |
|------|---------|
| **`state.py`** | `DeepResearchState` TypedDict — all workflow state including `branch_stack: List[BranchItem]` for DFS, accumulated learnings/citations/context with `Annotated[..., operator.add]` reducers |
| **`schemas.py`** | Pydantic models for structured LLM output: `SearchQueriesResponse`, `FollowUpQuestionsResponse`, `ResearchAnalysis` (with `LearningItem` containing `insight` + `source_url`) |
| **`prompts.py`** | `ChatPromptTemplate` definitions for all 4 LLM calls (plan, search queries, process results, report) |
| **`llm_service.py`** | `LLMService` — unified LLM factory with `get_strategic_llm()`, `get_smart_llm()`, `get_fast_llm()`, model caching, `with_structured_output` support, and token tracking via callbacks |
| **`callbacks.py`** | `TokenUsageCallbackHandler` — LangChain callback tracking prompt/completion/total tokens and errors |
| **`nodes.py`** | All 7 node functions + 2 conditional edge functions |
| **`graph.py`** | `build_deep_research_graph()` — assembles the `StateGraph`, wires edges, compiles with `MemorySaver` checkpointer |
| **`main.py`** | `run_deep_research()` — entry point that builds config, `LLMService`, initial state, invokes the graph, and returns results + usage summary |

## Node Details

### 1. `generate_research_plan`

- **Input**: `query`, `breadth`
- **Output**: `initial_search_results`, `follow_up_questions`, `combined_query`, initialized accumulators
- **Behavior**: Runs initial search across all configured retrievers (Tavily, etc.), then uses strategic LLM (ReasoningEfforts.High, temp=0.4) to generate follow-up questions. Builds a combined query from original + Q&A pairs.
- **Source**: Mirrors `DeepResearchSkill.generate_research_plan()`

### 2. `generate_search_queries`

- **Input**: `combined_query`, `breadth`, `depth`, `current_depth`
- **Output**: `search_queries`, updated `total_queries`
- **Behavior**: Uses strategic LLM (temp=0.4) with structured output (`SearchQueriesResponse`) to generate diverse search queries. Breadth halves at each depth level: `max(2, breadth // 2)`.
- **Source**: Mirrors `DeepResearchSkill.generate_search_queries()`

### 3. `execute_research`

- **Input**: `search_queries`, `concurrency_limit`, config
- **Output**: `research_results`, accumulated learnings/citations/urls/context/sources
- **Behavior**: For each query, spawns a `GPTResearcher` instance with `asyncio.Semaphore` concurrency control. Conducts research, then extracts learnings via strategic LLM (temp=0.4, ReasoningEfforts.High, max_tokens from config). Uses structured output (`ResearchAnalysis`) instead of regex parsing. Raises `RuntimeError` if ALL queries fail.
- **Source**: Mirrors `DeepResearchSkill.deep_research()` inner loop

### 4. `fan_out_branches`

- **Input**: `research_results`, `current_depth`
- **Output**: updated `branch_stack`
- **Behavior**: For each result, builds a branch query from `research_goal` + `follow_up_questions`, pushes as `BranchItem(query, depth=current_depth-1)` ON TOP of existing stack (DFS ordering).
- **Source**: Replaces the original's per-result recursive call

### 5. `pick_next_branch`

- **Input**: `branch_stack`
- **Output**: `combined_query`, `current_depth`, updated `branch_stack`
- **Behavior**: Pops the top item from the stack, sets it as the current query and depth for the next research cycle.

### 6. `assemble_final_context`

- **Input**: `all_learnings`, `all_citations`, `all_context`
- **Output**: `final_context` (trimmed string)
- **Behavior**: Deduplicates learnings, attaches citations as `[Source: url]` suffixes, extends with full research context, trims to 25K words (forward iteration — keeps curated learnings first).
- **Source**: Mirrors `DeepResearchSkill.run()` final context assembly

### 7. `generate_report`

- **Input**: `final_context`, `query`, `tone`
- **Output**: `report`, conversation `messages`
- **Behavior**: Uses strategic LLM to generate a detailed markdown research report (1200+ words) with in-text citations and references section.

### Conditional Edges

| Function | After Node | Condition | Routes |
|----------|-----------|-----------|--------|
| `should_continue_deeper` | `execute_research` | `current_depth > 1` | `"go_deeper"` → `fan_out_branches`, `"check_stack"` → `assemble_final_context` |
| `has_more_work` | `fan_out_branches` / `assemble_final_context` | `branch_stack` non-empty | `"next_branch"` → `pick_next_branch`, `"done"` → `assemble_final_context` / `generate_report` |

## State Design

```python
class DeepResearchState(TypedDict):
    # --- Input ---
    query: str
    breadth: int
    depth: int
    concurrency_limit: int
    tone: str
    config_path: Optional[str]
    headers: dict
    websocket: Any
    mcp_configs: Optional[list]
    mcp_strategy: Optional[str]
    on_progress: Optional[Callable]

    # --- Research plan ---
    initial_search_results: List[dict]
    follow_up_questions: List[str]
    combined_query: str

    # --- Current level ---
    current_depth: int
    current_breadth: int
    search_queries: List[SearchQuery]
    research_results: List[ResearchResult]

    # --- Branch stack for tree recursion (DFS) ---
    branch_stack: List[BranchItem]

    # --- Accumulated across levels (reducer: operator.add / _merge_dicts) ---
    all_learnings: Annotated[List[str], operator.add]
    all_citations: Annotated[Dict[str, str], _merge_dicts]
    all_visited_urls: Annotated[List[str], operator.add]
    all_context: Annotated[List[str], operator.add]
    all_sources: Annotated[List[dict], operator.add]

    # --- Progress ---
    total_queries: int
    completed_queries: int

    # --- Final output ---
    final_context: str
    report: str

    # --- Conversation history (checkpointer persistence) ---
    messages: Annotated[List[BaseMessage], add_messages]
```

## Key Design Decisions

### 1. Stack-based DFS Instead of Recursion

The `branch_stack` replaces recursive `deep_research()` calls. `fan_out_branches` pushes per-result follow-up queries ON TOP of the stack (DFS order), `pick_next_branch` pops from top. This faithfully reproduces the original's call-stack semantics where sub-branches at depth N-1 are fully processed before sibling branches at depth N resume.

### 2. Structured Outputs

Uses `with_structured_output(schema)` with Pydantic models instead of parsing text with regex (the original's approach). This gives type-safe responses for:
- `SearchQueriesResponse` — search queries with research goals
- `FollowUpQuestionsResponse` — targeted follow-up questions
- `ResearchAnalysis` — learnings with `insight` + `source_url` + follow-up questions

### 3. Checkpointer Support

Graph compiles with `MemorySaver` by default, but accepts custom checkpointers (SQLite, Postgres) for persistence/resumability. Each run gets a `thread_id` for checkpointer tracking.

### 4. Centralized LLM Management

All LLM calls go through `LLMService` (injected via `config["configurable"]["llm_service"]`), with:
- Model caching by `provider:model:temperature:max_tokens:reasoning_effort`
- Three tiers: `get_strategic_llm()`, `get_smart_llm()`, `get_fast_llm()`
- Automatic `NO_SUPPORT_TEMPERATURE_MODELS` / `SUPPORT_REASONING_EFFORT_MODELS` handling
- Token usage tracking via `TokenUsageCallbackHandler`

### 5. Context Trimming

Forward iteration (keeps curated learnings + citations first) vs. the original's reverse iteration (keeps most recent). Max 25K words. Learnings with citations appear at the front of the list from `assemble_final_context`, so forward iteration preserves curated insights over verbose raw scrape text.

### 6. Error Handling

- `execute_research` raises `RuntimeError` if ALL queries fail (vs. original silently returning empty)
- Config/programming errors (`TypeError`, `KeyError`, `ImportError`, `ValueError`) are re-raised immediately as `RuntimeError` rather than swallowed
- `asyncio.CancelledError` is re-raised to respect cancellation
- Network/transient errors are logged and the query is skipped (returns `None`)

## Entry Point API

```python
from deep_researcher_langgraph import run_deep_research

result = await run_deep_research(
    query="Your research question",
    breadth=4,          # queries per level (default from config)
    depth=2,            # recursive depth levels (default from config)
    concurrency_limit=2,# max concurrent GPTResearcher instances
    tone="Objective",
    on_progress=callback_fn,  # receives ResearchProgress updates
    checkpointer=None,        # MemorySaver default, or custom backend
)

# Result keys:
# report, final_context, visited_urls, sources, learnings, citations, usage_summary, thread_id
```

## Differences from Original (`gpt_researcher/skills/deep_research.py`)

| Aspect | Original | LangGraph Version |
|--------|----------|-------------------|
| Recursion | `asyncio` recursive calls | Stack-based DFS via `branch_stack` |
| LLM outputs | Regex text parsing | `with_structured_output` + Pydantic schemas |
| LLM management | Direct `create_chat_completion()` | Centralized `LLMService` with caching |
| Token tracking | Manual cost tracking on researcher | LangChain `TokenUsageCallbackHandler` |
| Prompts | Inline f-strings | `ChatPromptTemplate` definitions |
| Persistence | None | `MemorySaver` / custom checkpointers |
| Error on all-fail | Silent empty return | `RuntimeError` raised |
| Context trim direction | Reverse (keeps recent) | Forward (keeps curated learnings first) |

## Test Coverage

Tests are located in `tests/` and cover:

| Test File | Coverage |
|-----------|----------|
| `test_state_and_schemas.py` | State TypedDict structure, Pydantic schema validation |
| `test_prompts.py` | Prompt template formatting and variable substitution |
| `test_callbacks.py` | Token usage callback accumulation and error tracking |
| `test_llm_service.py` | LLM service model building, caching, and structured invocation |
| `test_nodes.py` | Individual node function logic with mocked LLM responses |
| `test_graph.py` | Graph structure, edge wiring, and compilation |
| `test_integration.py` | End-to-end workflow with mocked GPTResearcher |
