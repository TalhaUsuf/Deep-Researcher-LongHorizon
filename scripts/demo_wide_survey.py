"""Standalone wide-survey demo for the LangGraph deep researcher."""

import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_researcher_langgraph.main import run_deep_research  # noqa: E402


QUERY = "Compare the top 5 open source LLM frameworks for building production AI agents in 2025"
BREADTH = 5
DEPTH = 2
CONCURRENCY_LIMIT = 4
TONE = "Informative"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "demo_wide_survey_report.md"


def progress_callback(progress) -> None:
    """Print coarse progress updates from the deep research workflow."""
    current_query = progress.current_query or "starting"
    print(
        "[progress] "
        f"depth={progress.current_depth}/{progress.total_depth} "
        f"breadth={progress.current_breadth}/{progress.total_breadth} "
        f"queries={progress.completed_queries}/{progress.total_queries} "
        f"current={current_query}"
    )


async def main() -> None:
    start_time = time.perf_counter()

    result = await run_deep_research(
        query=QUERY,
        breadth=BREADTH,
        depth=DEPTH,
        concurrency_limit=CONCURRENCY_LIMIT,
        tone=TONE,
        on_progress=progress_callback,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = result.get("report", "")
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    usage_summary = result.get("usage_summary", {})

    print("\nSummary")
    print(f"Query: {QUERY}")
    print(
        "Params: "
        f"breadth={BREADTH}, depth={DEPTH}, "
        f"concurrency_limit={CONCURRENCY_LIMIT}, tone={TONE}"
    )
    print(f"Token usage: {usage_summary}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
