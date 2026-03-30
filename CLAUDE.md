# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPT Researcher is an autonomous deep research agent that produces detailed, factual research reports with citations. It uses a multi-agent architecture to plan sub-queries, search the web (or local documents), scrape content, compress context via embeddings, and generate long-form reports (2,000+ words).

## Common Commands

```bash
# Backend (FastAPI on port 8000)
python -m uvicorn backend.server.app:app --reload

# Frontend (Next.js on port 3000)
cd frontend/nextjs && npm run dev

# Multi-agent system (uses multi_agents/task.json config)
cd multi_agents && python main.py

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_mcp.py -v

# Run a specific test class
pytest tests/test_security_fix.py::TestSecureFilename -v

# Docker full stack
docker-compose up --build
```

**pytest config**: asyncio_mode is "strict", test files must match `test_*.py` pattern, test directory is `tests/`.

## Architecture

### Central Orchestrator: `GPTResearcher` (`gpt_researcher/agent.py`)

The main class that coordinates the entire pipeline. It initializes and wires together all components:
- `ResearchConductor` — plans sub-queries and executes search/scrape loops
- `ReportGenerator` — writes the final report via LLM
- `ContextManager` — compresses and manages retrieved context via embeddings
- `BrowserManager` — scrapes URLs for content extraction
- `SourceCurator` — ranks and filters sources
- `DeepResearchSkill` — recursive tree-like exploration for exhaustive research

### Research Flow

```
User query → GPTResearcher.conduct_research()
  → choose_agent() (LLM selects agent role)
  → plan_research() (generate sub-queries)
  → for each sub-query (parallel via asyncio.gather):
      → retriever.search() → scraper.scrape() → context_manager.compress()
  → optional MCP tool calls (strategy: fast/deep/disabled)
  → source curation
  → GPTResearcher.write_report() → LLM generates markdown report
```

### Three-Tier LLM System

The config (`gpt_researcher/config/`) defines three LLM tiers set via env vars:
- `FAST_LLM` (default `openai:gpt-4o-mini`) — quick tasks like agent selection
- `SMART_LLM` (default `openai:gpt-4o`) — report writing
- `STRATEGIC_LLM` (default `openai:gpt-4o`) — complex reasoning, sub-query planning

Format is `provider:model_name`. 30+ LLM providers supported via LangChain.

### Retrievers (`gpt_researcher/retrievers/`)

All retrievers implement `search(max_results) -> List[Dict]`. Selection priority: request headers → config → `RETRIEVER` env var → default (Tavily). Includes: Tavily, Google, Bing, DuckDuckGo, Arxiv, Semantic Scholar, PubMed, SerpAPI, Serper, Exa, MCP retriever, and others.

### Scrapers (`gpt_researcher/scraper/`)

All scrapers implement `scrape() -> (content, images, title)`. The `Scraper` class auto-selects: PyMuPDF for PDFs, ArxivScraper for arxiv.org URLs, configurable default (BeautifulSoup, Browser, FireCrawl, etc.). URLs are scraped in parallel.

### Multi-Agent System (`multi_agents/`)

LangGraph-based workflow with specialized agents forming a pipeline:

```
ResearchAgent → EditorAgent → (ReviewerAgent ↔ ReviserAgent loop) → WriterAgent → PublisherAgent
```

- **ChiefEditorAgent** (`orchestrator.py`) — coordinates the full workflow via a LangGraph StateGraph
- **EditorAgent** — plans outline, runs parallel section research
- **ReviewerAgent/ReviserAgent** — iterative review loop until quality threshold met
- **WriterAgent** — compiles sections into final report
- **PublisherAgent** — exports to PDF, DOCX, Markdown
- **HumanAgent** — optional human-in-the-loop feedback

Configured via `multi_agents/task.json`. Shared state via `ResearchState` and `DraftState`.

### Backend (`backend/server/`)

FastAPI app in `app.py`. Key endpoints:
- `WebSocket /ws` — real-time research streaming (ping/pong heartbeat at 30s)
- `POST /report/` — generate research report
- `POST /api/multi_agents` — run multi-agent workflow
- CRUD on `/api/reports` — JSON-based storage in `data/reports.json`
- File upload/download at `/upload/` and `/files/`

`WebSocketManager` bridges frontend and research agents. `CustomLogsHandler` captures streaming events.

### Frontend (`frontend/nextjs/`)

Next.js 14 + React 18 + TypeScript + Tailwind CSS. Key areas:
- `/app/` — pages and API route proxies
- `/components/Task/` — ResearchForm, AgentLogs, Report
- `/components/Settings/` — ChatBox, ToneSelector, FileUpload, MCPSelector
- `/hooks/` — custom React hooks for state
- WebSocket integration for real-time log/report streaming

### MCP Integration (`gpt_researcher/mcp/`)

Model Context Protocol for external tool integration. `MCPClientManager` manages server connections, `MCPToolSelector` picks tools, `MCPResearchSkill` executes queries. Three strategies: "fast" (one query), "deep" (all sub-queries), "disabled".

## Key Configuration

Required env vars: `OPENAI_API_KEY`, `TAVILY_API_KEY`

Other important env vars:
- `RETRIEVER` — comma-separated retriever list
- `EMBEDDING` — embedding provider:model (default `openai:text-embedding-3-large`)
- `DOC_PATH` — local document directory for local/hybrid research
- `MAX_SCRAPER_WORKERS` — concurrent scraper count (default 15)
- `REPORT_SOURCE` — web, local, hybrid, azure, langchain
- `MCP_STRATEGY` — fast, deep, disabled

Config loaded in order: env vars (highest) → JSON config file → defaults in `config/variables/default.py`.

## Conventions

- Python 3.11+ required
- Async-first codebase — research pipeline uses `asyncio.gather` for parallel execution
- Retrievers and scrapers follow simple interface contracts (see above)
- Skills pattern: components take the `GPTResearcher` instance in their constructor for access to config, memory, and state
- Prompts live in `gpt_researcher/prompts.py` using `PromptFamily` base class
- Frontend follows Next.js app router patterns with TypeScript strict mode
- Minimize AI-generated comments; prefer self-documenting code
