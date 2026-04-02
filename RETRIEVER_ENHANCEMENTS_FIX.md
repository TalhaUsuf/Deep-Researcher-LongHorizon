# Retriever Enhancements Fix Note

This note documents how retrievers are wired today, which environment variables they need, and which files should be changed if the LangGraph deep-research flow must use all retrievers that are both selected and actually usable.

No source code was changed while preparing this note.

## Relevant Repo Areas

```text
gpt_researcher/
  actions/
    retriever.py
    query_processing.py
  config/
    config.py
  retrievers/
    */*.py
  skills/
    researcher.py
    deep_research.py

deep_researcher_langgraph/
  main.py
  nodes.py
  state.py
  tests/

backend/
  server/
    server_utils.py
    websocket_manager.py

docs/docs/gpt-researcher/
  search-engines/search-engines.md
  retrievers/mcp-configs.mdx
```

## How Retrievers Are Used Today

### 1. Selection

- `gpt_researcher/config/config.py`
  - `Config._set_attributes()` parses `RETRIEVER` into `cfg.retrievers`.
- `gpt_researcher/actions/retriever.py`
  - `get_retrievers()` chooses retriever names from:
    1. `headers["retrievers"]`
    2. `headers["retriever"]`
    3. `cfg.retrievers`
    4. `cfg.retriever`
    5. default Tavily
  - It then maps names to classes via `get_retriever()`.

### 2. Initial Deep-Research Planning

- Legacy deep research:
  - `gpt_researcher/skills/deep_research.py:99`
  - `generate_research_plan()` loops over every configured retriever and aggregates results.
- LangGraph deep research:
  - `deep_researcher_langgraph/nodes.py:113`
  - `generate_research_plan()` also loops over every configured retriever and aggregates results.

### 3. Per-Query Research Inside LangGraph

- `deep_researcher_langgraph/nodes.py:300`
  - `execute_research()` creates a fresh `GPTResearcher` for each generated query.
- Inside `GPTResearcher`, the standard research flow runs:
  - `gpt_researcher/skills/researcher.py:48`
    - `plan_research()` uses only `self.researcher.retrievers[0]` for the planning search.
  - `gpt_researcher/skills/researcher.py:751`
    - `_search_relevant_source_urls()` loops over all configured non-MCP retrievers when collecting URLs to scrape.
  - `gpt_researcher/skills/researcher.py:816`
    - `_search()` passes `headers` for normal search execution and passes `researcher` for MCP.

## Retriever Matrix

`RETRIEVER` is the top-level selector. Example:

```bash
RETRIEVER=tavily,google,arxiv
```

### Supported Retrievers and Required Configuration

| Retriever | Required env/config | Optional env/config | Notes |
|---|---|---|---|
| `tavily` | `TAVILY_API_KEY` | none | Default retriever. In the retriever class, missing key becomes `""` instead of a hard failure. |
| `google` | `GOOGLE_API_KEY`, `GOOGLE_CX_KEY` | headers can carry `google_api_key` and `google_cx_key`, but `get_search_results()` does not pass headers today | Uses Google Custom Search. |
| `bing` | `BING_API_KEY` | none | Standard web search. |
| `searchapi` | `SEARCHAPI_API_KEY` | none | Standard web search. |
| `serpapi` | `SERPAPI_API_KEY` | none | Standard web search. |
| `serper` | `SERPER_API_KEY` | `SERPER_REGION`, `SERPER_LANGUAGE`, `SERPER_TIME_RANGE`, `SERPER_EXCLUDE_SITES` | Standard web search. |
| `searx` | `SEARX_URL` | none | Requires reachable SearxNG instance. |
| `exa` | `EXA_API_KEY` | none | Also needs `exa_py` installed. |
| `bocha` | `BOCHA_API_KEY` | none | Supported in code, but under-documented elsewhere. |
| `duckduckgo` | none | none | Needs `ddgs` installed. |
| `arxiv` | none | none | Needs `arxiv` installed. |
| `semantic_scholar` | none | none | No API key used in current implementation. |
| `pubmed_central` | none strictly required | `NCBI_API_KEY`, `PUBMED_DB`, `PUBMED_ARG_*` | Returns full-text documents as `url/raw_content`, not `href/body`. |
| `custom` | `RETRIEVER_ENDPOINT` | `RETRIEVER_ARG_*` | Returns documents as `url/raw_content`, not `href/body`. |
| `mcp` | runtime `mcp_configs` plus retriever selection including `mcp` | `MCP_STRATEGY` | Also needs `langchain_mcp_adapters` installed. |

## Important Current Gaps

### 1. "Configured" Is Not The Same As "Usable"

- `gpt_researcher/actions/retriever.py:99`
  - `get_retrievers()` returns classes for selected names, but it does not validate:
    - required env vars
    - required packages
    - runtime `mcp_configs`
    - response-shape compatibility
- If `get_retriever(name)` returns `None`, the code silently falls back to Tavily.
  - This can hide bad config and can cause a requested retriever such as `mcp` to become an unintended Tavily fallback.

### 2. LangGraph Initial Planning Does Not Auto-Add MCP

- `deep_researcher_langgraph/main.py` passes `mcp_configs` into graph state.
- `deep_researcher_langgraph/nodes.py:132`
  - `generate_research_plan()` calls `get_retrievers(state["headers"], cfg)`.
  - That path only sees config and headers.
  - It does not mirror `GPTResearcher._process_mcp_configs()`, which auto-adds `mcp` when `mcp_configs` exist and no explicit `RETRIEVER` override was provided.
- Result:
  - query-level researchers can use MCP later,
  - but the initial LangGraph planning step may ignore MCP completely unless `RETRIEVER` already contains `mcp`.

### 3. `get_search_results()` Does Not Pass Headers To Normal Retrievers

- `gpt_researcher/actions/query_processing.py:12`
  - For non-MCP retrievers, it calls:
    - `retriever(query, query_domains=query_domains)`
  - It does not pass `headers`.
- Result:
  - header-based auth overrides for retrievers like Google and Tavily are ignored in planning paths that call `get_search_results()`.

### 4. Inner `GPTResearcher` Planning Still Uses Only The First Retriever

- `gpt_researcher/skills/researcher.py:48`
  - `plan_research()` uses `self.researcher.retrievers[0]`.
- Result:
  - Even when LangGraph initial planning uses multiple retrievers, every deeper query still plans its outline from only the first retriever's search results.
  - The later URL-discovery step does use all configured retrievers, but the planning context is still biased toward the first provider.

### 5. Some Retrievers Return Documents, Not Search Links

- `custom` and `pubmed_central` return `url/raw_content`.
- `gpt_researcher/skills/researcher.py:773`
  - `_search_relevant_source_urls()` only reads `href`.
- `gpt_researcher/skills/researcher.py:921`
  - `_extract_content()` also only reads `href`.
- Result:
  - those retrievers are selected successfully,
  - but the standard web-research scraping path effectively ignores their returned content.

### 6. Backend/Web Config Surfaces Only Cover A Subset Of Retrievers

- `backend/server/server_utils.py:267`
  - currently exposes only:
    - `TAVILY_API_KEY`
    - `GOOGLE_API_KEY`
    - `GOOGLE_CX_KEY`
    - `BING_API_KEY`
    - `SEARCHAPI_API_KEY`
    - `SERPAPI_API_KEY`
    - `SERPER_API_KEY`
    - `SEARX_URL`
- Missing from this surface:
  - `EXA_API_KEY`
  - `BOCHA_API_KEY`
  - `NCBI_API_KEY`
  - `PUBMED_DB`
  - `RETRIEVER_ENDPOINT`
  - `RETRIEVER_ARG_*`
  - `SERPER_*` optionals

### 7. Docs Are Not Fully Aligned With The Actual Retriever Set

- `docs/docs/gpt-researcher/search-engines/search-engines.md`
  - missing or incomplete for:
    - `bocha`
    - `semantic_scholar`
    - `mcp`
    - `pubmed_central` details
    - `custom` response-shape caveat in the deep-research path
- `deep_researcher_langgraph/README.md`
  - only documents a small subset of retriever env vars.

## Recommended Enhancement Plan

### Goal

Make LangGraph deep research use all retrievers that are:

1. selected by config or headers,
2. actually usable in the current runtime,
3. compatible with the current deep-research execution path.

### A. Add A Shared "usable retriever" resolver

Primary files:

- `gpt_researcher/actions/retriever.py`
- `gpt_researcher/config/config.py`
- possibly `gpt_researcher/retrievers/utils.py`

What to add:

- A resolver that returns:
  - requested retrievers
  - usable retrievers
  - skipped retrievers with reasons
- Validation should check:
  - required env vars
  - required packages
  - MCP runtime requirements (`mcp_configs`, adapter install)
  - output mode (`search_links` vs `document_results`)

Why here:

- this is the central place where retriever names become retriever classes.
- today the system knows names, not capability.

### B. Mirror GPTResearcher MCP auto-inclusion in LangGraph

Primary files:

- `deep_researcher_langgraph/main.py`
- `deep_researcher_langgraph/nodes.py`
- `gpt_researcher/agent.py`

What to change:

- LangGraph should resolve active retrievers using the same MCP inclusion rules that `GPTResearcher` already applies.
- If `mcp_configs` exist and the user did not explicitly exclude MCP, the planner should include `mcp`.

Why here:

- this is the main parity gap between LangGraph and the legacy researcher behavior.

### C. Pass Headers Through Every Planning Search Path

Primary files:

- `gpt_researcher/actions/query_processing.py`
- `gpt_researcher/skills/deep_research.py`
- `deep_researcher_langgraph/nodes.py`

What to change:

- extend `get_search_results()` so it can pass:
  - `headers`
  - `researcher`
  - `mcp_configs` or equivalent context
- update callers so initial planning searches use the same auth/config inputs as normal search execution.

Why here:

- otherwise header-configured retrievers look "configured" but are not actually used in planning.

### D. Make Inner Query Planning Multi-Retriever Too

Primary files:

- `gpt_researcher/skills/researcher.py`

What to change:

- update `plan_research()` so it aggregates search results across all usable retrievers, not only `retrievers[0]`.

Why here:

- LangGraph delegates actual query execution to `GPTResearcher`, so this file controls most of the depth-level behavior.

### E. Normalize Retriever Output Shapes

Primary files:

- `gpt_researcher/actions/query_processing.py`
- `gpt_researcher/skills/researcher.py`
- possibly `gpt_researcher/retrievers/custom/custom.py`
- possibly `gpt_researcher/retrievers/pubmed_central/pubmed_central.py`

What to change:

- introduce a normalized internal result contract.
- support both:
  - link-style results: `href/body/title`
  - document-style results: `url/raw_content/title`
- if a retriever returns `raw_content`, bypass scraping and load that content directly into context/vector-store flow.

Why here:

- without this, `custom` and `pubmed_central` can be configured but still contribute little or nothing to deep research.

### F. Stop Silent Fallback To Tavily For Missing/Unavailable Requested Retrievers

Primary files:

- `gpt_researcher/actions/retriever.py`

What to change:

- replace silent `or get_default_retriever()` fallback with explicit skip + warning metadata.
- only use default Tavily when the user did not request anything.

Why here:

- fallback hides configuration errors and makes "successful retriever" accounting impossible.

### G. Expand Backend Config Surface If UI/API Runtime Configuration Matters

Primary files:

- `backend/server/server_utils.py`
- any request/response schema or settings UI that feeds it

What to change:

- add missing retriever env fields so backend-configured sessions can use the full retriever set, not only the currently hard-coded subset.

Suggested additions:

- `EXA_API_KEY`
- `BOCHA_API_KEY`
- `NCBI_API_KEY`
- `PUBMED_DB`
- `RETRIEVER_ENDPOINT`
- `SERPER_REGION`
- `SERPER_LANGUAGE`
- `SERPER_TIME_RANGE`
- `SERPER_EXCLUDE_SITES`

## Files That Should Be Updated

### Core behavior

- `gpt_researcher/actions/retriever.py`
  - add usable-retriever resolution and remove silent Tavily fallback for requested-but-unavailable retrievers.
- `gpt_researcher/actions/query_processing.py`
  - pass headers for normal retrievers and normalize result shapes.
- `gpt_researcher/skills/researcher.py`
  - make planning multi-retriever and support document-style retriever results.
- `gpt_researcher/agent.py`
  - reuse MCP auto-inclusion logic from a shared resolver instead of keeping it only inside `GPTResearcher`.

### LangGraph deep research

- `deep_researcher_langgraph/main.py`
  - prepare retriever-resolution context from `mcp_configs` and config.
- `deep_researcher_langgraph/nodes.py`
  - use the shared usable-retriever resolver in `generate_research_plan()`.
  - pass full retriever context into planning searches.
- `deep_researcher_langgraph/state.py`
  - optionally store `active_retrievers` and `skipped_retrievers` for traceability.

### Backend/runtime configuration

- `backend/server/server_utils.py`
  - expose the missing retriever env vars.
- `backend/server/websocket_manager.py`
  - if runtime config is set per request, keep MCP/retriever resolution consistent with the shared resolver.

### Tests

- `deep_researcher_langgraph/tests/test_nodes.py`
  - add coverage for partial retriever usability and header-based auth.
- `deep_researcher_langgraph/tests/test_functional_requirements.py`
  - add coverage for:
    - mixed configured/unconfigured retriever lists
    - MCP auto-inclusion from `mcp_configs`
    - document-style retrievers (`custom`, `pubmed_central`)
    - no unintended Tavily fallback
- `tests/test-your-retriever.py`
  - optional smoke coverage for normalized output shapes.

### Docs

- `deep_researcher_langgraph/README.md`
  - expand env var documentation for all retrievers.
- `docs/docs/gpt-researcher/search-engines/search-engines.md`
  - align the doc with the actual retriever set and env requirements.
- `docs/docs/gpt-researcher/retrievers/mcp-configs.mdx`
  - document LangGraph parity expectations and the need for `mcp` in active retriever resolution.

## Bottom Line

If the requirement is "LangGraph deep research should focus on all possible retrievers that are successfully configured", the main implementation change is not in the retriever classes themselves.

The main work is in:

1. central retriever capability resolution,
2. LangGraph planner parity with `GPTResearcher`,
3. multi-retriever planning inside `GPTResearcher`,
4. output-shape normalization for document-returning retrievers,
5. backend/docs alignment so configuration surfaces match the code.
