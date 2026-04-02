#!/usr/bin/env python3
"""Standalone balanced deep research demo for the LangGraph workflow."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from deep_researcher_langgraph.main import run_deep_research
from deep_researcher_langgraph.state import ResearchProgress

QUERY = (
    "How do transformer architectures compare to state space models "
    "for long sequence modeling?"
)
BREADTH = 3
DEPTH = 2
CONCURRENCY_LIMIT = 3
TONE = "Analytical"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def progress_callback(progress: ResearchProgress) -> None:
    """Print progress updates from the deep research workflow."""
    current_query = progress.current_query or "planning"
    print(
        "[progress] "
        f"depth {progress.current_depth}/{progress.total_depth} | "
        f"breadth {progress.current_breadth}/{progress.total_breadth} | "
        f"queries {progress.completed_queries}/{progress.total_queries} | "
        f"current: {current_query}"
    )


async def run_demo() -> None:
    """Run the balanced research demo and persist the markdown report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"demo_balanced_research_{timestamp}.md"

    start_time = time.perf_counter()
    result = await run_deep_research(
        query=QUERY,
        breadth=BREADTH,
        depth=DEPTH,
        concurrency_limit=CONCURRENCY_LIMIT,
        tone=TONE,
        on_progress=progress_callback,
    )
    elapsed_seconds = time.perf_counter() - start_time

    report = result.get("report", "")
    output_path.write_text(report, encoding="utf-8")

    usage_summary = result.get("usage_summary", {})
    print("\nBalanced research demo complete")
    print(f"Query: {QUERY}")
    print(
        "Params: "
        f"breadth={BREADTH}, depth={DEPTH}, "
        f"concurrency_limit={CONCURRENCY_LIMIT}, tone={TONE}"
    )
    print(
        "Token usage: "
        f"prompt={usage_summary.get('prompt_tokens', 0)}, "
        f"completion={usage_summary.get('completion_tokens', 0)}, "
        f"total={usage_summary.get('total_tokens', 0)}, "
        f"calls={usage_summary.get('call_count', 0)}, "
        f"errors={usage_summary.get('error_count', 0)}"
    )
    print(f"Elapsed time: {elapsed_seconds:.2f}s")
    print(f"Output path: {output_path}")


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
