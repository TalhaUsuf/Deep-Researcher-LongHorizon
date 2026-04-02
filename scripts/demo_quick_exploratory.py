import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from deep_researcher_langgraph.main import run_deep_research

QUERY = "What are the latest advances in quantum error correction?"
BREADTH = 2
DEPTH = 1
CONCURRENCY_LIMIT = 2
TONE = "Objective"
SCRIPT_NAME = "demo_quick_exploratory"


def build_progress_callback():
    last_message = ""

    def on_progress(progress) -> None:
        nonlocal last_message

        depth_current = getattr(progress, "current_depth", 0)
        depth_total = getattr(progress, "total_depth", 0)
        breadth_current = getattr(progress, "current_breadth", 0)
        breadth_total = getattr(progress, "total_breadth", 0)
        completed_queries = getattr(progress, "completed_queries", 0)
        total_queries = getattr(progress, "total_queries", 0)
        current_query = getattr(progress, "current_query", None)

        message = (
            f"[progress] depth {depth_current}/{depth_total} | "
            f"breadth {breadth_current}/{breadth_total} | "
            f"queries {completed_queries}/{total_queries}"
        )
        if current_query:
            message += f" | current query: {current_query}"

        if message != last_message:
            print(message, flush=True)
            last_message = message

    return on_progress


async def main() -> None:
    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"{SCRIPT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    start_time = time.perf_counter()
    result = await run_deep_research(
        query=QUERY,
        breadth=BREADTH,
        depth=DEPTH,
        concurrency_limit=CONCURRENCY_LIMIT,
        tone=TONE,
        on_progress=build_progress_callback(),
    )
    elapsed_time = time.perf_counter() - start_time

    report = result.get("report", "")
    output_path.write_text(report, encoding="utf-8")

    print("\nSummary")
    print(f"Query: {QUERY}")
    print(
        "Params: "
        f"breadth={BREADTH}, depth={DEPTH}, "
        f"concurrency_limit={CONCURRENCY_LIMIT}, tone={TONE}"
    )
    print(
        "Token usage: "
        f"{json.dumps(result.get('usage_summary', {}), indent=2, sort_keys=True)}"
    )
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
