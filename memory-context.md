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

165 tests pass in `deep_researcher_langgraph/tests/` including 5 new tests for per-tier base URL passthrough.
