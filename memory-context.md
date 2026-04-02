# Memory Context

## Project Overview

GPT Researcher deep_researcher_langgraph module — fixed for custom local LLM endpoints and improved report quality. Branch `fix/errors` merged into `main`.

## Local LLM Endpoints

| Tier | Model | Endpoint | API Key |
|------|-------|----------|---------|
| FAST_LLM | llama-3.1-70b | http://69.48.159.10:30000/v1 | not-needed |
| SMART_LLM | llama-3.1-70b | http://69.48.159.10:30000/v1 | not-needed |
| STRATEGIC_LLM | qwen3-235b | http://69.48.159.8:30005/v1 | not-needed |
| EMBEDDING | Nexus_Embedding_Model_seq_8192_embd_1024 | http://69.48.159.8:30007/v1 | not-needed |

All endpoints are OpenAI-compatible (vLLM/SGLang). OPENAI_API_KEY set to "not-needed".

## Fixes Applied

1. **Per-tier LLM base URLs** — `FAST_LLM_BASE_URL`, `SMART_LLM_BASE_URL`, `STRATEGIC_LLM_BASE_URL` env vars added to BaseConfig, DEFAULT_CONFIG, Config, and LLMService
2. **Tavily query truncation** — Initial search query truncated to 400 chars (Tavily API limit)
3. **GPTResearcher base URL routing** — `_resolve_openai_base_url()` in `gpt_researcher/utils/llm.py` routes sub-researcher LLM calls to correct endpoints
4. **Embedding base URL** — `EMBEDDING_BASE_URL` env var added, `gpt_researcher/memory/embeddings.py` checks it before `OPENAI_BASE_URL`
5. **Report quality fixes**:
   - `max_tokens` for learning extraction: 1000 → `cfg.strategic_token_limit` (4000)
   - Enriched learning extraction prompt (8-15 learnings, preserve exact numbers)
   - `_trim_context` LIFO → FIFO (preserve curated learnings over raw scrape text)
   - Enriched report prompt (tables, citations, "no generic conclusions", date awareness)
   - Report uses strategic LLM (qwen3-235b) instead of smart LLM
   - Report `max_tokens`: 4000 → `cfg.smart_token_limit` (6000)
6. **Rich output exports** — Runner script produces MD + DOCX + hyperlinked references + sources/learnings files

## Key Files Modified

| File | Change |
|------|--------|
| `gpt_researcher/config/variables/base.py` | Added `*_BASE_URL` type definitions |
| `gpt_researcher/config/variables/default.py` | Added `*_BASE_URL` defaults (None) |
| `gpt_researcher/config/config.py` | Parse per-tier base URLs from env |
| `gpt_researcher/utils/llm.py` | `_resolve_openai_base_url()` for per-model routing |
| `gpt_researcher/memory/embeddings.py` | `EMBEDDING_BASE_URL` support |
| `deep_researcher_langgraph/llm_service.py` | Pass `openai_api_base` per tier |
| `deep_researcher_langgraph/nodes.py` | Token limits, FIFO trim, strategic LLM for report |
| `deep_researcher_langgraph/prompts.py` | Enriched learning + report prompts |
| `deep_researcher_langgraph/main.py` | `load_dotenv()` |
| `backend/server/app.py` | `load_dotenv()` |
| `run_quantization_research.py` | Runner with MD/DOCX export |
| `.env.sample` | Documented config template |

## Research Parameter Guidance

- **depth=4 + breadth>=3** creates exponential tree (81 leaf queries, 58+ min). Avoid.
- **depth=2, breadth=2** — smoke test (~6 queries, 5-7 min)
- **depth=2, breadth=4** — full run (~12 queries, 10-15 min)

## Smoke Test Results (depth=2, breadth=2)

| Metric | Before Fixes | After Fixes |
|--------|-------------|-------------|
| LengthFinishReasonError | 5 | 0 |
| Embedding success | 0/127 (all 401) | 100% |
| LLM routing leaks to api.openai.com | 127 calls | 0 |
| Report word count | 978 (generic) | 1,504 (substantive) |
| Markdown tables | Qualitative (High/Low) | Quantitative (tok/s, perplexity %) |
| Inline citations | 0 | 112 |
| Export formats | None | MD + DOCX + sources + learnings |
| Duration | 10m 21s | 6m 51s |

## Test Suite

184 tests pass in `deep_researcher_langgraph/tests/` including 26 new functional requirement tests and 5 per-tier base URL passthrough tests.

---

## Session 2: Parallel Execution + Functional Spec (2026-04-01 – 2026-04-02)

Branch: `worktree-feature+arxiv-search` (worktree off `fork/main`)

### What Was Done

1. **Level-by-level parallel execution** — Replaced DFS stack (sequential, one branch at a time) with concurrent level-by-level execution. All branches at the same depth level now run concurrently via `asyncio.gather`, throttled by a local `asyncio.Semaphore(concurrency_limit)`.

2. **Hierarchical report structure** — Added `TreeNode` with dot-separated paths (e.g., "0", "0.1", "1.0.2"). Report headings now map to tree depth: `##` for level 1, `###` for level 2, `####` for level 3.

3. **Deep Interview functional spec** — 9-round Socratic interview crystallized functional requirements into `.omc/specs/deep-interview-deep-researcher-langgraph.md` (17.2% ambiguity, 7 acceptance criteria).

4. **CCG tri-model review** — Codex + Gemini independently reviewed the parallelism plan. Key consensus: don't store Semaphore in state (breaks checkpointing), use `asyncio.gather` not LangGraph Send API, each researcher gets a copy of visited_urls.

5. **Codex adversarial review** — Found 3 issues: no same-level concurrency, broken visited-URL dedup, nondeterministic context trimming.

6. **Codex code review against spec** — Found 9 issues across 3 severity levels. Generated comprehensive test suite (26 tests) and fix plan.

7. **Ralph fix loop** — Applied all fixes until 184/184 tests green.

### Architecture Change (Before → After)

```
BEFORE (DFS stack — sequential):
  pick_next_branch → generate_search_queries → execute_research → fan_out_branches
       ↑                   (1 branch)           (1 batch)              │
       └───────────────────────────────────────────────────────────────┘

AFTER (level-by-level parallel):
  generate_research_plan → generate_search_queries → execute_research
                                ↑                          │
                                │                   should_continue_deeper?
                                │                    /              \
                                │              "go_deeper"        "done"
                                │                   │                │
                                └── fan_out_branches─┘     assemble_final_context
                                                                  │
                                                           generate_report → END
```

Graph nodes: 7 → 6 (removed `pick_next_branch`, `has_more_work`)

### Performance Impact

| Config | Old (DFS) phases | New (level) phases | Speedup |
|--------|------------------|--------------------|---------|
| depth=2, breadth=2 | 3 | 2 | 1.5x |
| depth=2, breadth=4 | 7 | 2 | 3.5x |
| depth=3, breadth=2 | 7 | 3 | 2.3x |
| depth=3, breadth=4 | 28+ | 3 | ~9x |

### Fixes Applied (Ralph Session)

| Fix | File | AC |
|-----|------|----|
| Ordered-unique reducer for visited URLs | `state.py` | AC5, AC6 |
| Preserve seeded visited URLs (don't clear) | `gpt_researcher/skills/researcher.py` | AC5 |
| Graceful degradation on retriever failures | `nodes.py` | AC3 |
| Progress callback exception safety | `nodes.py` | AC3 |
| Guard against empty pending_branches at deep levels | `nodes.py` | AC4 |
| Stop descent when no results produced | `nodes.py` | AC3, AC4 |
| 30-min wall-clock timeout with partial recovery | `main.py` | AC7 |
| Input validation (positive int) | `main.py` | AC7 |
| Fix falsy config handling (`or` → `is None`) | `main.py` | AC7 |
| Stable dedup for learnings (`dict.fromkeys`) | `main.py` | AC6 |
| Remove hard-coded 1200-word minimum | `prompts.py` | Spec compliance |

### Key Files Modified (This Session)

| File | Change |
|------|--------|
| `deep_researcher_langgraph/state.py` | Added `TreeNode`, `BranchItem` path/parent_topic, `_merge_ordered_unique` reducer, `pending_branches`, `research_tree`; removed `branch_stack` |
| `deep_researcher_langgraph/schemas.py` | Added `path`, `parent_topic` to `SearchQueryItem` |
| `deep_researcher_langgraph/nodes.py` | Rewrote 5 nodes for concurrent execution, deleted `pick_next_branch`/`has_more_work`, added graceful degradation, callback safety |
| `deep_researcher_langgraph/graph.py` | Simplified to 6-node topology, removed stack loop |
| `deep_researcher_langgraph/prompts.py` | Hierarchical heading instructions, removed 1200-word minimum |
| `deep_researcher_langgraph/main.py` | Added timeout (`asyncio.wait_for`), input validation, stable dedup, fixed config handling |
| `gpt_researcher/skills/researcher.py` | Preserve seeded visited_urls instead of clearing |
| `deep_researcher_langgraph/tests/test_functional_requirements.py` | 26 new tests covering all 7 ACs + edge cases |
| `.omc/specs/deep-interview-deep-researcher-langgraph.md` | Functional requirements spec |
| `.omc/specs/fix-plan.md` | Codex-generated fix plan |
| `.env` | Runtime config (GPT-5.4 models, Tavily, custom embedding endpoint) |

### LLM Configuration (Updated)

| Tier | Model | Endpoint |
|------|-------|----------|
| FAST_LLM | gpt-5.4-nano | OpenAI API standard |
| SMART_LLM | gpt-5.4-mini | OpenAI API standard |
| STRATEGIC_LLM | gpt-5.4 | OpenAI API standard |
| EMBEDDING | Nexus_Embedding_Model_seq_8192_embd_1024 | http://69.48.159.8:30007/v1 |
