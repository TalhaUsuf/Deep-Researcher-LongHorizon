# Deep Research: Original vs LangGraph Implementation -- Verification Report

**Date:** 2026-03-30
**Original:** `gpt_researcher/skills/deep_research.py`
**New:** `deep_researcher_langgraph/` (9 files)

---

## Feature Verification Checklist

### 1. `generate_search_queries()` -- strategic LLM, temperature=0.4

**YES**

- `nodes.py` line 193: `llm_service.get_strategic_llm(temperature=0.4)` -- matches original's strategic LLM + temperature=0.4.
- `llm_service.py` line 87-101: `get_strategic_llm()` uses `self._config.strategic_llm_provider` and `self._config.strategic_llm_model`, matching the original's `self.researcher.cfg.strategic_llm_provider` / `strategic_llm_model`.
- Original also passes `reasoning_effort=self.researcher.cfg.reasoning_effort`; the new implementation passes `reasoning_effort` via `get_strategic_llm()` default which calls `self.reasoning_effort` (line 100), which reads from config (line 49). Equivalent behavior.
- Output parsing: original uses manual text parsing (`Query:`/`Goal:` prefixes); new uses `with_structured_output(SearchQueriesResponse)` (Pydantic schema). Functionally equivalent, more robust.

### 2. `generate_research_plan()` -- strategic LLM, ReasoningEfforts.High, temperature=0.4, max_tokens=1000, initial retriever search

**YES**

- `nodes.py` lines 122-126: `llm_service.get_strategic_llm(temperature=0.4, max_tokens=1000, reasoning_effort=ReasoningEfforts.High.value)` -- all three parameters match the original (lines 134-139).
- Initial retriever search: `nodes.py` lines 101-108 iterates retrievers via `get_retrievers()` and calls `get_search_results()`, matching the original's loop (lines 102-113).
- Minor difference: original passes `researcher=self.researcher` to `get_search_results()`, new does not. This could matter for retrievers that depend on the researcher instance.
- Minor difference: original uses `num_questions=3` as default parameter; new uses `breadth` (typically 4). Different number of follow-up questions generated.

### 3. `process_research_results()` -- strategic LLM, temperature=0.4, max_tokens=1000, ReasoningEfforts.High

**YES**

- `nodes.py` lines 276-279: `llm_service.get_strategic_llm(temperature=0.4, max_tokens=1000, reasoning_effort=ReasoningEfforts.High.value)` -- all three parameters match the original (lines 155-161).
- Uses `PROCESS_RESEARCH_RESULTS_PROMPT` (ChatPromptTemplate) and `with_structured_output(ResearchAnalysis)` instead of the original's regex-based parsing. More reliable.

### 4. Tree recursion -- per-result branching (not per-level), breadth halving: max(2, breadth//2)

**PARTIAL**

The original recurses **per-result**: each result at depth N spawns its own recursive `deep_research()` call at depth N-1 with `max(2, breadth//2)`. This creates a **tree** where 4 results each spawn 2 sub-queries, yielding 4 + 8 = 12 total queries at depth=2, breadth=4.

The new implementation models this via `fan_out_branches` (nodes.py lines 352-382) which creates one branch query **per result**, and `pick_next_branch` (lines 389-398) which processes them sequentially. The graph loop (`graph.py` lines 96-113) wires:
- `fan_out_branches` -> `pick_next_branch` -> `generate_search_queries` -> `execute_research` -> `should_continue_deeper`
- After assembly, `has_pending_branches` checks if more branches remain and loops back to `pick_next_branch`.

**The per-result branching structure is present**, but with a key difference: branches are processed **sequentially** (one at a time) rather than spawning true parallel recursive calls as the original does. The breadth halving formula `max(2, breadth // 2)` is correctly implemented in `generate_search_queries` (nodes.py lines 179-183).

**Behavioral gap:** The graph topology has a subtle issue. When `execute_research` finishes processing a branch at depth > 1, `should_continue_deeper` routes to `fan_out_branches`, which creates new sub-branches AND decrements depth. But the `has_pending_branches` conditional on `assemble_final_context` can route back to `pick_next_branch` for **sibling** branches that were at the **previous** depth -- these siblings would now execute at the decremented depth. The original avoids this because each recursive call has its own local `depth` variable. This means the new implementation may not correctly handle depth tracking across sibling branches at different levels of the tree.

### 5. `ResearchProgress` class with all fields

**YES**

- `state.py` lines 25-35: `ResearchProgress` class with all 7 fields matching the original:
  - `current_depth` (init 1)
  - `total_depth`
  - `current_breadth` (init 0)
  - `total_breadth`
  - `current_query` (Optional[str], init None)
  - `total_queries` (init 0)
  - `completed_queries` (init 0)
- Identical to original (lines 39-48).

### 6. `on_progress` callback mechanism

**YES**

- `state.py` line 50: `on_progress: Optional[Callable]` in state.
- `main.py` line 42: `on_progress` parameter accepted and placed into initial state (line 99).
- `nodes.py` lines 75-80: `_notify_progress()` helper fires the callback with the `ResearchProgress` object.
- Called in `generate_search_queries` (line 212), `execute_research` (lines 249, 297), and `fan_out_branches` (implicit via progress update at line 377).
- The `ResearchProgress` instance is stored in the configurable dict (main.py line 128) and accessed in nodes via `_get_progress()`.

### 7. Context trimming to 25K words, keeping most recent items

**YES**

- `nodes.py` line 43: `MAX_CONTEXT_WORDS = 25000`
- `nodes.py` lines 61-72: `_trim_context()` iterates `reversed(context_list)`, keeps most recent items within word limit. Algorithm is identical to the original's `trim_context_to_word_limit()` (lines 23-37).
- Applied in `assemble_final_context` (line 428).
- Minor difference: original trims at each depth level AND at the end; new trims only at the end. Could lead to larger intermediate state.

### 8. Concurrency limiting with asyncio.Semaphore (default 2)

**YES**

- `nodes.py` line 239: `semaphore = asyncio.Semaphore(concurrency_limit)`
- `nodes.py` line 242: `async with semaphore:` wraps each query.
- `main.py` line 70: default from config with fallback `getattr(cfg, "deep_research_concurrency", 2)` -- matches the original's default of 2.

### 9. Cost tracking via estimate_llm_cost

**YES**

- `llm_service.py` line 22: `from gpt_researcher.utils.costs import estimate_llm_cost`
- `llm_service.py` lines 138-143: `_track_cost()` calls `estimate_llm_cost(messages, response)` and adds the result to `self._callback_handler.total_cost`.
- Called after every `invoke_structured()` (line 164) and `invoke()` (line 170).
- Additionally, `TokenUsageCallbackHandler` (callbacks.py) tracks token counts via the LangChain callback interface, providing dual tracking.

### 10. GPTResearcher spawning per query with full parameter propagation

**YES**

- `nodes.py` lines 251-262: `GPTResearcher(...)` spawned per query inside `_process_single_query()`.
- All parameters propagated:
  - `query` -- from `serp_query["query"]`
  - `report_type` -- `ReportType.ResearchReport.value`
  - `report_source` -- `ReportSource.Web.value`
  - `tone` -- from state
  - `websocket` -- from state
  - `config_path` -- from state
  - `headers` -- from state
  - `visited_urls` -- from accumulated set
  - `mcp_configs` -- from state
  - `mcp_strategy` -- from state
- Matches the original (lines 246-258) exactly.

### 11. Visited URL deduplication across recursive calls

**YES**

- `state.py` line 72: `all_visited_urls: Annotated[List[str], operator.add]` accumulates across nodes.
- `nodes.py` line 237: `visited_urls: Set[str] = set(state.get("all_visited_urls", []))` -- converts to set before passing to `GPTResearcher`, ensuring deduplication.
- `nodes.py` line 259: `visited_urls=visited_urls` passes the deduped set.
- Results collected back as list (line 304) and accumulated via `operator.add`.
- Minor difference: the accumulated list may contain duplicates, but dedup happens at the point of use (GPTResearcher instantiation). The original uses a `set` throughout.

### 12. Per-query error handling (continue on failure)

**YES**

- `nodes.py` lines 309-315: `except Exception as e:` with `traceback.format_exc()`, `logger.error()`, returns `None`.
- `nodes.py` line 319: `results = [r for r in raw_results if r is not None]` -- filters out failures.
- Matches original pattern exactly (lines 289-294, 299).

### 13. Citation assembly: "{learning} [Source: {url}]"

**YES**

- `nodes.py` lines 419-424:
  ```python
  if citation:
      context_with_citations.append(f"{learning} [Source: {citation}]")
  else:
      context_with_citations.append(learning)
  ```
- Identical format to original (lines 399-403).

### 14. Combined query building with Q&A pairs

**YES**

- `nodes.py` lines 134-138:
  ```python
  qa_pairs = [f"Q: {q}\nA: Automatically proceeding with research" for q in questions]
  combined_query = (
      f"Initial Query: {query}\nFollow-up Questions and Answers:\n"
      + "\n".join(qa_pairs)
  )
  ```
- Matches original (lines 374-377). Minor formatting difference: original uses `"Follow - up Questions"` (with space-dash-space); new uses `"Follow-up Questions"` (hyphenated). Semantically equivalent.

### 15. All prompts use ChatPromptTemplate.from_messages

**YES**

- `prompts.py`: All 4 prompts use `ChatPromptTemplate.from_messages`:
  - `GENERATE_SEARCH_QUERIES_PROMPT` (line 6)
  - `GENERATE_RESEARCH_PLAN_PROMPT` (line 20)
  - `PROCESS_RESEARCH_RESULTS_PROMPT` (line 40)
  - `GENERATE_REPORT_PROMPT` (line 56)
- All invoked via `.format_messages()` in nodes.

### 16. All structured outputs use with_structured_output

**YES**

- `llm_service.py` line 162: `structured_llm = llm.with_structured_output(schema)` in `invoke_structured()`.
- Three structured calls:
  - `generate_research_plan` -> `FollowUpQuestionsResponse` (nodes.py line 127)
  - `generate_search_queries` -> `SearchQueriesResponse` (nodes.py line 194)
  - `execute_research` (result analysis) -> `ResearchAnalysis` (nodes.py line 281)
- The fourth LLM call (`generate_report`) correctly uses plain `invoke()` since it returns free-form text.

### 17. Unified LLMService for all LLM calls

**YES**

- `llm_service.py`: Single `LLMService` class with:
  - `get_strategic_llm()` (line 87)
  - `get_smart_llm()` (line 104)
  - `get_fast_llm()` (line 119)
  - `invoke_structured()` (line 146)
  - `invoke()` (line 167)
- All 4 node LLM calls go through `LLMService`:
  - `generate_research_plan` -> `invoke_structured` (line 127)
  - `generate_search_queries` -> `invoke_structured` (line 194)
  - `execute_research` -> `invoke_structured` (line 281)
  - `generate_report` -> `invoke` (line 459)
- Model caching via `_model_cache` (line 62).
- Consistent callback injection via `get_callbacks()`.

### 18. LangChain callback handler for token tracking

**YES**

- `callbacks.py`: `TokenUsageCallbackHandler` extends `BaseCallbackHandler` (line 11).
- Implements `on_llm_end` (line 22): extracts token usage from `response.generations` and `response.llm_output`.
- Implements `on_llm_error` (line 39): logs warnings.
- Tracks `prompt_tokens`, `completion_tokens`, `total_tokens`, `total_cost`, `_call_count`.
- Passed to all LLM calls via `config={"callbacks": self.get_callbacks()}` (llm_service.py lines 163, 169).
- Cost estimation via `estimate_llm_cost` supplements the token tracking (llm_service.py lines 138-143).

### 19. StateGraph with proper TypedDict and Annotated[List, operator.add]

**YES**

- `state.py`: `DeepResearchState(TypedDict)` (line 38) with `Annotated[List, operator.add]` on 4 accumulation fields:
  - `all_learnings: Annotated[List[str], operator.add]` (line 68)
  - `all_visited_urls: Annotated[List[str], operator.add]` (line 72)
  - `all_context: Annotated[List[str], operator.add]` (line 73)
  - `all_sources: Annotated[List[dict], operator.add]` (line 74)
- `graph.py` line 67: `StateGraph(DeepResearchState)` -- properly typed.
- `all_citations: Dict[str, str]` does not use an accumulation annotation (correct -- `operator.add` does not work for dicts); manually merged via `dict.update()` in `execute_research`.

---

## Summary Table

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1 | `generate_search_queries()` -- strategic LLM, temp=0.4 | **YES** | Uses `get_strategic_llm(temperature=0.4)` |
| 2 | `generate_research_plan()` -- strategic LLM, High reasoning, temp=0.4, max_tokens=1000, initial search | **YES** | All params match; minor: `researcher` kwarg omitted from `get_search_results()` |
| 3 | `process_research_results()` -- strategic LLM, temp=0.4, max_tokens=1000, High reasoning | **YES** | All params match; structured output replaces regex parsing |
| 4 | Tree recursion -- per-result branching, breadth halving | **PARTIAL** | Per-result branches exist via `fan_out_branches` + `pick_next_branch` loop, but processed sequentially not in parallel; depth tracking across sibling branches may be incorrect |
| 5 | `ResearchProgress` class with all 7 fields | **YES** | Identical class definition |
| 6 | `on_progress` callback mechanism | **YES** | Wired through state and configurable dict; fired at key points |
| 7 | Context trimming to 25K words, most recent | **YES** | Same algorithm; applied only at end (not per-level) |
| 8 | Concurrency limiting with Semaphore (default 2) | **YES** | Same pattern and default |
| 9 | Cost tracking via `estimate_llm_cost` | **YES** | Called after every LLM invocation; dual tracking with callback handler |
| 10 | GPTResearcher spawning with full param propagation | **YES** | All 10 parameters propagated identically |
| 11 | Visited URL deduplication | **YES** | Converts to set at point of use; list accumulation may have dupes in state |
| 12 | Per-query error handling | **YES** | Same try/except + None-filter pattern |
| 13 | Citation assembly `{learning} [Source: {url}]` | **YES** | Identical format string |
| 14 | Combined query with Q&A pairs | **YES** | Same structure; trivial formatting difference |
| 15 | All prompts use ChatPromptTemplate.from_messages | **YES** | All 4 prompts |
| 16 | All structured outputs use with_structured_output | **YES** | 3 structured + 1 plain text (correct) |
| 17 | Unified LLMService for all LLM calls | **YES** | All 4 calls route through LLMService |
| 18 | LangChain callback handler for token tracking | **YES** | Full implementation with on_llm_end/on_llm_error |
| 19 | StateGraph with TypedDict and Annotated[List, operator.add] | **YES** | 4 accumulation fields; graph properly typed |

---

## Remaining Gaps and Differences

### Architectural

1. **Sequential branch processing vs parallel recursion (item 4).** The original spawns all per-result recursive calls concurrently via `asyncio.gather` within the same `deep_research()` invocation. The new implementation processes branches one at a time through the graph loop. This is functionally correct (same branches are explored) but slower -- branches do not execute in parallel.

2. **Depth tracking across sibling branches.** When `fan_out_branches` decrements `current_depth`, all subsequent sibling branches picked by `pick_next_branch` inherit the decremented depth. In the original, each recursive call has its own local `depth` variable, so siblings are independent. This could cause incorrect depth handling when multiple branches exist.

### Minor

3. **`researcher` kwarg omitted from `get_search_results()`.** Original passes `researcher=self.researcher` (line 108); new omits it (nodes.py line 106). Could affect retrievers that depend on the researcher instance.

4. **`num_questions` default differs.** Original: `num_questions=3`; new: uses `breadth` (typically 4).

5. **Intermediate context trimming.** Original trims at each depth level; new trims only at the end.

6. **Combined query formatting.** Original: `"Follow - up Questions and Answers:\n"` (space-dash-space); new: `"Follow-up Questions and Answers:\n"` (hyphenated).

7. **GPTResearcher log handler integration.** Original calls `self.researcher._log_event()` for cost logging. New uses Python logging only.

8. **Class-level state mutation.** Original mutates `self.researcher.context`, `self.researcher.visited_urls`, `self.researcher.research_sources` for external access. New returns a dict -- cannot be used as a drop-in replacement for `DeepResearchSkill`.

---

## Improvements in the New Implementation

1. **Structured output** via Pydantic schemas replaces fragile regex/string parsing.
2. **Clean separation of concerns** across 9 focused files vs one 427-line monolith.
3. **LLM model caching** avoids redundant provider instantiation.
4. **Standalone CLI** entry point via `main.py`.
5. **End-to-end pipeline** including report generation.
6. **Typed state** with annotated accumulators documents data flow.
7. **Dual cost tracking** via both `estimate_llm_cost` and LangChain callback token counting.
