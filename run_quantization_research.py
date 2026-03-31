"""Runner script for LLM quantization deep research.

Loads environment from .env and runs the LangGraph deep research workflow
with a comprehensive query about LLM quantization methods.

Produces rich output: Markdown (with hyperlinked references), PDF, and DOCX.

Usage:
    python run_quantization_research.py
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Load .env before any other imports that might read env vars
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
if not env_path.exists():
    print(f"Error: .env file not found at {env_path}", file=sys.stderr)
    print("Please create a .env file with the required configuration.", file=sys.stderr)
    sys.exit(1)

load_dotenv(env_path, override=True)

from deep_researcher_langgraph import run_deep_research  # noqa: E402
from gpt_researcher.actions.markdown_processing import add_references  # noqa: E402

QUERY = (
    "What are the most effective quantization methods for Large Language Models (LLMs)? "
    "Compare GPTQ, AWQ, GGUF/GGML, SqueezeLLM, and bitsandbytes approaches. "
    "Analyze their impact on model quality (perplexity, benchmark scores), inference speed, "
    "memory footprint, and hardware compatibility. Include recent advances in 2024-2025 "
    "such as QuIP#, AQLM, and HQQ. What are the trade-offs between different bit-widths "
    "(2-bit, 3-bit, 4-bit, 8-bit) and which methods work best for different deployment "
    "scenarios (edge devices, consumer GPUs, data center)?"
)

DEPTH = 2
BREADTH = 2
CONCURRENCY = 4

LOG_DIR = Path(__file__).resolve().parent / "logs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LOG_FILE = LOG_DIR / "research_trace.log"


async def export_report(report: str, filename_base: str, logger: logging.Logger) -> None:
    """Export report to Markdown, PDF, and DOCX formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Markdown ---
    md_path = OUTPUT_DIR / f"{filename_base}.md"
    md_path.write_text(report, encoding="utf-8")
    logger.info("Markdown report saved to %s", md_path)

    # --- PDF ---
    try:
        from backend.utils import write_md_to_pdf
        pdf_result = await write_md_to_pdf(report, filename_base)
        if pdf_result:
            logger.info("PDF report saved to outputs/%s.pdf", filename_base)
        else:
            logger.warning("PDF export returned empty (check md2pdf / weasyprint installation)")
    except ImportError as e:
        logger.warning("PDF export skipped — missing dependency: %s", e)
    except Exception as e:
        logger.warning("PDF export failed: %s", e)

    # --- DOCX ---
    try:
        from backend.utils import write_md_to_word
        docx_result = await write_md_to_word(report, filename_base)
        if docx_result:
            logger.info("DOCX report saved to outputs/%s.docx", filename_base)
        else:
            logger.warning("DOCX export returned empty (check python-docx / htmldocx)")
    except ImportError as e:
        logger.warning("DOCX export skipped — missing dependency: %s", e)
    except Exception as e:
        logger.warning("DOCX export failed: %s", e)


async def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Dual logging: console + file with detailed trace
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    file_handler = logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)-40s | %(funcName)-25s | %(message)s"
    ))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("STARTING LLM QUANTIZATION DEEP RESEARCH")
    logger.info("=" * 80)
    logger.info("Query: %s", QUERY)
    logger.info("Parameters: depth=%d, breadth=%d, concurrency=%d", DEPTH, BREADTH, CONCURRENCY)
    logger.info("FAST_LLM=%s  BASE_URL=%s", os.getenv("FAST_LLM"), os.getenv("FAST_LLM_BASE_URL"))
    logger.info("SMART_LLM=%s  BASE_URL=%s", os.getenv("SMART_LLM"), os.getenv("SMART_LLM_BASE_URL"))
    logger.info("STRATEGIC_LLM=%s  BASE_URL=%s", os.getenv("STRATEGIC_LLM"), os.getenv("STRATEGIC_LLM_BASE_URL"))
    logger.info("EMBEDDING=%s  BASE_URL=%s", os.getenv("EMBEDDING"), os.getenv("EMBEDDING_BASE_URL"))
    logger.info("TAVILY_API_KEY=%s...", os.getenv("TAVILY_API_KEY", "")[:10])

    start_time = time.time()

    try:
        result = await run_deep_research(
            query=QUERY,
            depth=DEPTH,
            breadth=BREADTH,
            concurrency_limit=CONCURRENCY,
            tone="Analytical",
        )
    except KeyboardInterrupt:
        logger.warning("Research interrupted by user after %.1fs", time.time() - start_time)
        sys.exit(130)
    except RuntimeError as exc:
        logger.error("Research workflow failed after %.1fs: %s", time.time() - start_time, exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected error after %.1fs: %s", time.time() - start_time, exc, exc_info=True)
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Research completed in %.1fs", elapsed)

    # --- Post-process: add hyperlinked references ---
    report = result.get("report", "(no report generated)")
    visited_urls = set(result.get("visited_urls", []))
    if visited_urls:
        report = add_references(report, visited_urls)
        logger.info("Added %d hyperlinked references to report", len(visited_urls))

    # --- Print report to console ---
    print("\n" + "=" * 80)
    print("RESEARCH REPORT")
    print("=" * 80)
    print(report)

    # --- Print token usage ---
    print("\n" + "=" * 80)
    print("TOKEN USAGE SUMMARY")
    print("=" * 80)
    usage = result.get("usage_summary", {})
    if isinstance(usage, dict):
        for key, value in usage.items():
            print(f"  {key}: {value}")
    else:
        print(usage)

    # --- Print metadata ---
    print("\n" + "-" * 80)
    print(f"Thread ID: {result.get('thread_id', 'N/A')}")
    print(f"Sources collected: {len(result.get('sources', []))}")
    print(f"URLs visited: {len(visited_urls)}")
    print(f"Learnings extracted: {len(result.get('learnings', []))}")
    print(f"Citations: {len(result.get('citations', {}))}")
    print(f"Total elapsed: {elapsed:.1f}s")
    print("-" * 80)

    # --- Export to all formats ---
    filename_base = "llm_quantization_deep_research"
    await export_report(report, filename_base, logger)

    # Also save to logs/ for trace monitor
    log_report_path = LOG_DIR / "research_report.md"
    log_report_path.write_text(report, encoding="utf-8")

    # Save sources and learnings as supplementary files
    sources_path = OUTPUT_DIR / f"{filename_base}_sources.md"
    sources_content = "# Research Sources\n\n"
    for i, src in enumerate(result.get("sources", []), 1):
        url = src.get("href", src.get("url", "N/A"))
        body = src.get("body", src.get("content", ""))[:200]
        sources_content += f"## Source {i}\n- **URL**: [{url}]({url})\n- **Preview**: {body}...\n\n"
    sources_path.write_text(sources_content, encoding="utf-8")

    learnings_path = OUTPUT_DIR / f"{filename_base}_learnings.md"
    citations = result.get("citations", {})
    learnings_content = "# Research Learnings\n\n"
    for i, learning in enumerate(result.get("learnings", []), 1):
        citation = citations.get(learning, "")
        if citation:
            learnings_content += f"{i}. {learning} ([source]({citation}))\n\n"
        else:
            learnings_content += f"{i}. {learning}\n\n"
    learnings_path.write_text(learnings_content, encoding="utf-8")

    logger.info("Sources saved to %s", sources_path)
    logger.info("Learnings saved to %s", learnings_path)
    logger.info("Trace log at %s", LOG_FILE)
    logger.info("All outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
