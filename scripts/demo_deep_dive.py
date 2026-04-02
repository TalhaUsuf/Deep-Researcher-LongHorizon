"""Standalone demo for the LangGraph deep research workflow."""

import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()
load_dotenv(PROJECT_ROOT / ".env")

from deep_researcher_langgraph.main import run_deep_research  # noqa: E402
from deep_researcher_langgraph.state import ResearchProgress  # noqa: E402

QUERY = (
    "What are the environmental and economic tradeoffs of nuclear fusion "
    "vs advanced fission reactors?"
)
DEPTH = 3
BREADTH = 2
CONCURRENCY_LIMIT = 2
TONE = "Objective"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "demo_deep_dive_report.md"


def progress_handler(progress: ResearchProgress) -> None:
    current_query = progress.current_query or "planning research branches"
    print(
        "[progress] "
        f"depth {progress.current_depth}/{progress.total_depth} | "
        f"breadth {progress.current_breadth}/{progress.total_breadth} | "
        f"queries {progress.completed_queries}/{progress.total_queries} | "
        f"{current_query}"
    )


def print_summary(output_path: Path, usage_summary: object, elapsed_seconds: float) -> None:
    print("\nSummary")
    print(f"Query: {QUERY}")
    print(
        "Parameters: "
        f"breadth={BREADTH}, depth={DEPTH}, "
        f"concurrency_limit={CONCURRENCY_LIMIT}, tone={TONE}"
    )
    print(f"Token usage: {usage_summary}")
    print(f"Elapsed time: {elapsed_seconds:.2f}s")
    print(f"Output path: {output_path}")


async def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    result = await run_deep_research(
        query=QUERY,
        breadth=BREADTH,
        depth=DEPTH,
        concurrency_limit=CONCURRENCY_LIMIT,
        tone=TONE,
        on_progress=progress_handler,
    )
    elapsed_seconds = time.perf_counter() - start_time

    report = result.get("report", "")
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    print_summary(
        output_path=OUTPUT_PATH,
        usage_summary=result.get("usage_summary", {}),
        elapsed_seconds=elapsed_seconds,
    )


if __name__ == "__main__":
    asyncio.run(main())
