#!/usr/bin/env python3
"""Render an animated GIF for deep_researcher_langgraph tree parallelism."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "static" / "img" / "deep-research-parallelism-b2-d3.gif"

WIDTH = 1600
HEIGHT = 900

BG = "#f7f1e7"
PANEL = "#fffaf2"
INK = "#1f1d1a"
MUTED = "#6e665b"
BORDER = "#d4c3ab"
ACCENT = "#c45c2b"
ACCENT_SOFT = "#f7d8c7"
SEARCH = "#2f70c9"
SEARCH_SOFT = "#dce9ff"
ACTIVE = "#f2a93b"
ACTIVE_SOFT = "#ffe2ad"
DONE = "#2d8a4d"
DONE_SOFT = "#d9efda"
FUTURE = "#ece4d7"
FUTURE_TEXT = "#8a7558"
QUEUE = "#a4482b"

GRAPH_POSITIONS = {
    "generate_research_plan": (90, 110, 320, 185),
    "generate_search_queries": (390, 110, 660, 185),
    "execute_research": (720, 110, 975, 185),
    "fan_out_branches": (720, 235, 975, 310),
    "assemble_final_context": (1080, 110, 1380, 185),
    "generate_report": (1180, 235, 1425, 310),
}

TREE_PANEL = (40, 335, 1050, 850)
WORKER_PANEL = (1085, 335, 1560, 850)


@dataclass(frozen=True)
class WorkerTask:
    label: str
    phase: str
    detail: str


@dataclass(frozen=True)
class Stage:
    title: str
    subtitle: str
    active_node: str | None
    statuses: Dict[str, str]
    workers: Sequence[WorkerTask]
    queued_count: int
    stat_lines: Sequence[str]
    duration_ms: int


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
        if bold
        else ["DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_H1 = load_font(32, bold=True)
FONT_H2 = load_font(22, bold=True)
FONT_H3 = load_font(18, bold=True)
FONT_BODY = load_font(17)
FONT_SMALL = load_font(15)
FONT_TINY = load_font(13)


def breadth_for_depth(original_breadth: int, total_depth: int, current_depth: int) -> int:
    breadth = original_breadth
    for _ in range(total_depth - current_depth):
        breadth = max(2, breadth // 2)
    return breadth


def build_tree_levels(breadth: int, depth: int) -> Dict[int, List[str]]:
    levels: Dict[int, List[str]] = {}
    parents = [""]
    for current_depth in range(depth, 0, -1):
        level_breadth = breadth_for_depth(breadth, depth, current_depth)
        if current_depth == depth:
            current = [str(i) for i in range(level_breadth)]
        else:
            current = [
                f"{parent}.{i}"
                for parent in parents
                for i in range(level_breadth)
            ]
        levels[current_depth] = current
        parents = current
    return levels


def tuple_sort_key(path: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in path.split("."))


def compute_node_positions(levels: Dict[int, List[str]], depth: int) -> Dict[str, Tuple[int, int, int, int]]:
    left, top, right, bottom = TREE_PANEL
    inner_left = left + 40
    inner_top = top + 70
    inner_right = right - 35
    inner_bottom = bottom - 40
    node_w = 155
    node_h = 42

    usable_w = inner_right - inner_left - node_w
    col_gap = usable_w // max(depth - 1, 1)

    positions: Dict[str, Tuple[int, int, int, int]] = {}
    for col_index, current_depth in enumerate(range(depth, 0, -1)):
        paths = sorted(levels[current_depth], key=tuple_sort_key)
        x0 = inner_left + col_index * col_gap
        x1 = x0 + node_w
        count = len(paths)
        gap = 18
        total_h = count * node_h + max(count - 1, 0) * gap
        start_y = inner_top + max((inner_bottom - inner_top - total_h) // 2, 0)
        for row_index, path in enumerate(paths):
            y0 = start_y + row_index * (node_h + gap)
            positions[path] = (x0, y0, x1, y0 + node_h)
    return positions


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_spacing: int = 5,
) -> int:
    x, y = xy
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing
    return y


def center_text(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - text_w) / 2
    y = y0 + (y1 - y0 - text_h) / 2 - 1
    draw.text((x, y), text, font=font, fill=fill)


def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], color: str, width: int = 3) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    dx = ex - sx
    dy = ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux = dx / length
    uy = dy / length
    size = 10
    left = (ex - ux * size - uy * size * 0.6, ey - uy * size + ux * size * 0.6)
    right = (ex - ux * size + uy * size * 0.6, ey - uy * size - ux * size * 0.6)
    draw.polygon([end, left, right], fill=color)


def parent_path(path: str) -> str | None:
    if "." not in path:
        return None
    return path.rsplit(".", 1)[0]


def status_colors(status: str) -> Tuple[str, str, str]:
    if status == "queued":
        return SEARCH_SOFT, SEARCH, INK
    if status == "active":
        return ACTIVE_SOFT, ACTIVE, INK
    if status == "done":
        return DONE_SOFT, DONE, INK
    return FUTURE, BORDER, FUTURE_TEXT


def draw_structure_graph(draw: ImageDraw.ImageDraw, active_node: str | None) -> None:
    draw.rounded_rectangle((40, 55, 1560, 320), radius=28, fill=PANEL, outline=BORDER, width=2)
    draw.text((70, 78), "Current deep_researcher_langgraph structure", font=FONT_H2, fill=INK)
    draw.text(
        (70, 105),
        "Six nodes, one conditional branch, and a level-by-level loop back into query generation.",
        font=FONT_BODY,
        fill=MUTED,
    )

    labels = {
        "generate_research_plan": "generate_research_plan",
        "generate_search_queries": "generate_search_queries",
        "execute_research": "execute_research",
        "fan_out_branches": "fan_out_branches",
        "assemble_final_context": "assemble_final_context",
        "generate_report": "generate_report",
    }

    for key, box in GRAPH_POSITIONS.items():
        fill = ACCENT_SOFT if key == active_node else "#fdf6eb"
        outline = ACCENT if key == active_node else BORDER
        width = 4 if key == active_node else 2
        draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=width)
        center_text(draw, box, labels[key], FONT_SMALL, INK)

    draw_arrow(draw, (320, 147), (390, 147), MUTED)
    draw_arrow(draw, (660, 147), (720, 147), MUTED)
    draw_arrow(draw, (975, 147), (1080, 147), MUTED)
    draw_arrow(draw, (847, 185), (847, 235), MUTED)
    draw_arrow(draw, (720, 273), (525, 273), MUTED)
    draw_arrow(draw, (525, 273), (525, 185), MUTED)
    draw_arrow(draw, (1380, 185), (1302, 235), MUTED)
    draw.text((988, 118), "depth > 1", font=FONT_TINY, fill=MUTED)
    draw.text((1002, 152), "done", font=FONT_TINY, fill=MUTED)
    draw.text((857, 201), "go deeper", font=FONT_TINY, fill=MUTED)


def draw_tree_panel(
    draw: ImageDraw.ImageDraw,
    depth: int,
    levels: Dict[int, List[str]],
    positions: Dict[str, Tuple[int, int, int, int]],
    statuses: Dict[str, str],
) -> None:
    left, top, right, bottom = TREE_PANEL
    draw.rounded_rectangle(TREE_PANEL, radius=28, fill=PANEL, outline=BORDER, width=2)
    draw.text((left + 28, top + 22), "Tree-based search expansion", font=FONT_H2, fill=INK)
    draw.text(
        (left + 28, top + 50),
        "Path ids match TreeNode.path. Colors show queued, running, and completed work.",
        font=FONT_BODY,
        fill=MUTED,
    )

    for current_depth in range(depth, 0, -1):
        paths = levels[current_depth]
        sample_box = positions[paths[0]]
        label_x = sample_box[0]
        label = f"current_depth={current_depth}"
        draw.text((label_x, top + 95), label, font=FONT_H3, fill=INK)
        if current_depth == depth:
            sub = f"{len(paths)} root queries"
        else:
            sub = f"{len(paths)} queries"
        draw.text((label_x, top + 120), sub, font=FONT_TINY, fill=MUTED)

    for current_depth in range(depth, 0, -1):
        for path in levels[current_depth]:
            parent = parent_path(path)
            if not parent:
                continue
            x0, y0, x1, y1 = positions[parent]
            cx0, cy0, cx1, cy1 = positions[path]
            start = (x1, (y0 + y1) // 2)
            end = (cx0, (cy0 + cy1) // 2)
            edge_color = "#ddcfba"
            if statuses.get(parent) != "future" and statuses.get(path) != "future":
                edge_color = "#b79c7b"
            draw.line((start[0], start[1], end[0], end[1]), fill=edge_color, width=2)

    for current_depth in range(depth, 0, -1):
        for path in levels[current_depth]:
            box = positions[path]
            fill, outline, text_fill = status_colors(statuses.get(path, "future"))
            width = 4 if statuses.get(path) == "active" else 2
            draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline, width=width)
            center_text(draw, box, path, FONT_SMALL, text_fill)

    legend_y = bottom - 42
    legend_items = [
        ("future", FUTURE, BORDER),
        ("queued", SEARCH_SOFT, SEARCH),
        ("running", ACTIVE_SOFT, ACTIVE),
        ("done", DONE_SOFT, DONE),
    ]
    cursor_x = left + 30
    for label, fill, outline in legend_items:
        swatch = (cursor_x, legend_y, cursor_x + 18, legend_y + 18)
        draw.rounded_rectangle(swatch, radius=5, fill=fill, outline=outline, width=2)
        draw.text((cursor_x + 28, legend_y - 1), label, font=FONT_TINY, fill=MUTED)
        cursor_x += 128


def draw_worker_panel(
    draw: ImageDraw.ImageDraw,
    stage: Stage,
    total_queries: int,
    completed_queries: int,
    breadth: int,
    depth: int,
    concurrency_limit: int,
) -> None:
    left, top, right, bottom = WORKER_PANEL
    draw.rounded_rectangle(WORKER_PANEL, radius=28, fill=PANEL, outline=BORDER, width=2)
    draw.text((left + 26, top + 22), stage.title, font=FONT_H2, fill=INK)
    current_y = draw_text_block(
        draw,
        (left + 26, top + 54),
        stage.subtitle,
        FONT_BODY,
        MUTED,
        max_width=right - left - 52,
    )

    current_y += 12
    metrics_box_height = 22 + len(lines := [
        f"breadth={breadth}, depth={depth}, concurrency_limit={concurrency_limit}",
        f"research tasks complete: {completed_queries}/{total_queries}",
        *stage.stat_lines,
    ]) * 20
    metrics_box = (left + 24, current_y, right - 24, current_y + metrics_box_height)
    draw.rounded_rectangle(metrics_box, radius=18, fill="#fcf4e7", outline=BORDER, width=2)
    line_y = metrics_box[1] + 14
    for line in lines:
        draw.text((metrics_box[0] + 16, line_y), line, font=FONT_SMALL, fill=INK)
        line_y += 20

    queue_box_top = metrics_box[3] + 14
    queue_box = (left + 24, queue_box_top, right - 24, bottom - 24)
    draw.rounded_rectangle(queue_box, radius=18, fill="#fdf8f0", outline=BORDER, width=2)
    draw.text((queue_box[0] + 16, queue_box[1] + 14), "Parallel workers", font=FONT_H3, fill=INK)

    slot_y = queue_box[1] + 50
    slot_gap = 8
    workers = list(stage.workers)
    slot_count = max(len(workers), 4)
    footer_reserved = 12
    available_slot_space = max(queue_box[3] - slot_y - footer_reserved, 120)
    raw_slot_h = (available_slot_space - (slot_count - 1) * slot_gap) // slot_count
    slot_h = max(36, min(56, raw_slot_h))
    for idx in range(slot_count):
        y0 = slot_y + idx * (slot_h + slot_gap)
        slot = (queue_box[0] + 16, y0, queue_box[2] - 16, y0 + slot_h)
        has_task = idx < len(workers)
        fill = ACTIVE_SOFT if has_task else "#f0e7d7"
        outline = ACTIVE if has_task else BORDER
        draw.rounded_rectangle(slot, radius=14, fill=fill, outline=outline, width=2)
        title = f"slot {idx + 1}"
        draw.text((slot[0] + 14, slot[1] + 8), title, font=FONT_TINY, fill=MUTED)
        if has_task:
            worker = workers[idx]
            line_y = slot[1] + max(20, slot_h // 2)
            draw.text((slot[0] + 14, line_y), worker.label, font=FONT_SMALL, fill=INK)
            meta = f"{worker.phase} | {worker.detail}"
            draw.text((slot[0] + 155, line_y + 1), meta, font=FONT_TINY, fill=MUTED)
        else:
            draw.text((slot[0] + 14, slot[1] + max(20, slot_h // 2)), "idle", font=FONT_SMALL, fill=FUTURE_TEXT)


def draw_footer(draw: ImageDraw.ImageDraw, stage_index: int, stage_count: int) -> None:
    y = HEIGHT - 26
    x = 50
    gap = 18
    radius = 6
    for idx in range(stage_count):
        fill = ACCENT if idx == stage_index else "#d9cdb8"
        draw.ellipse((x, y, x + radius * 2, y + radius * 2), fill=fill)
        x += radius * 2 + gap


def render_frame(
    stage: Stage,
    stage_index: int,
    stage_count: int,
    breadth: int,
    depth: int,
    concurrency_limit: int,
    levels: Dict[int, List[str]],
    positions: Dict[str, Tuple[int, int, int, int]],
    total_queries: int,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((42, 22), "deep_researcher_langgraph parallel search", font=FONT_H1, fill=INK)
    draw.text(
        (43, 58),
        "Animation for breadth=2 and depth=3, using the current level-by-level tree execution model.",
        font=FONT_BODY,
        fill=MUTED,
    )

    draw_structure_graph(draw, stage.active_node)

    completed_queries = sum(1 for status in stage.statuses.values() if status == "done")
    draw_tree_panel(draw, depth, levels, positions, stage.statuses)
    draw_worker_panel(
        draw,
        stage,
        total_queries=total_queries,
        completed_queries=completed_queries,
        breadth=breadth,
        depth=depth,
        concurrency_limit=concurrency_limit,
    )
    draw_footer(draw, stage_index, stage_count)

    return image


def build_statuses(levels: Dict[int, List[str]], done: Iterable[str], queued: Iterable[str], active: Iterable[str]) -> Dict[str, str]:
    statuses = {path: "future" for paths in levels.values() for path in paths}
    for path in done:
        statuses[path] = "done"
    for path in queued:
        statuses[path] = "queued"
    for path in active:
        statuses[path] = "active"
    return statuses


def build_stages(levels: Dict[int, List[str]], depth: int, concurrency_limit: int) -> List[Stage]:
    level_3 = sorted(levels[depth], key=tuple_sort_key)
    level_2 = sorted(levels[depth - 1], key=tuple_sort_key)
    level_1 = sorted(levels[1], key=tuple_sort_key)
    first_leaf_batch = level_1[:concurrency_limit]
    second_leaf_batch = level_1[concurrency_limit:]

    return [
        Stage(
            title="Overview",
            subtitle="The graph plans once, then loops through query generation and research one depth wave at a time.",
            active_node=None,
            statuses=build_statuses(levels, done=[], queued=[], active=[]),
            workers=[],
            queued_count=0,
            stat_lines=(
                "waves of research: 3",
                "query-generation calls by wave: 1 -> 2 -> 4",
                "research tasks by wave: 2 -> 4 -> 8",
            ),
            duration_ms=1200,
        ),
        Stage(
            title="1. generate_research_plan",
            subtitle="Initial search results and follow-up questions are merged into a single combined query for the tree search.",
            active_node="generate_research_plan",
            statuses=build_statuses(levels, done=[], queued=[], active=[]),
            workers=[WorkerTask("combined query", "planning", "initial search + follow-up questions")],
            queued_count=0,
            stat_lines=("tree not expanded yet", "pending branches: 0"),
            duration_ms=1050,
        ),
        Stage(
            title=f"2. generate_search_queries at current_depth={depth}",
            subtitle="The root call runs once and emits two top-level research queries, paths 0 and 1.",
            active_node="generate_search_queries",
            statuses=build_statuses(levels, done=[], queued=level_3, active=[]),
            workers=[WorkerTask("root query generator", "query gen", "1 call -> 2 research queries")],
            queued_count=0,
            stat_lines=("root wave width: 2", "parallel query-generation calls now: 1"),
            duration_ms=1050,
        ),
        Stage(
            title=f"3. execute_research at current_depth={depth}",
            subtitle="Both GPTResearcher tasks run together because the wave width is below the concurrency cap.",
            active_node="execute_research",
            statuses=build_statuses(levels, done=[], queued=[], active=level_3),
            workers=[WorkerTask(path, "research", "running") for path in level_3],
            queued_count=0,
            stat_lines=("running research tasks: 2", "queued research tasks: 0"),
            duration_ms=1150,
        ),
        Stage(
            title=f"4. fan_out_branches to current_depth={depth - 1}",
            subtitle="Each completed root result creates one pending branch seed. The tree has 2 parents ready for the next wave.",
            active_node="fan_out_branches",
            statuses=build_statuses(levels, done=level_3, queued=[], active=[]),
            workers=[WorkerTask("branch 0", "fan-out", "seed next wave"), WorkerTask("branch 1", "fan-out", "seed next wave")],
            queued_count=0,
            stat_lines=("pending branch seeds: 2", "new current_depth: 2"),
            duration_ms=1050,
        ),
        Stage(
            title="5. generate_search_queries at current_depth=2",
            subtitle="Query generation now runs once per pending branch in parallel. Two branch prompts yield four child research queries.",
            active_node="generate_search_queries",
            statuses=build_statuses(levels, done=level_3, queued=level_2, active=[]),
            workers=[
                WorkerTask("branch 0", "query gen", "2 child queries"),
                WorkerTask("branch 1", "query gen", "2 child queries"),
            ],
            queued_count=0,
            stat_lines=("parallel query-generation calls now: 2", "wave width after expansion: 4"),
            duration_ms=1150,
        ),
        Stage(
            title="6. execute_research at current_depth=2",
            subtitle="All four child queries fit under the semaphore limit, so the entire second wave runs at once.",
            active_node="execute_research",
            statuses=build_statuses(levels, done=level_3, queued=[], active=level_2),
            workers=[WorkerTask(path, "research", "running") for path in level_2],
            queued_count=0,
            stat_lines=("running research tasks: 4", "queued research tasks: 0"),
            duration_ms=1150,
        ),
        Stage(
            title="7. fan_out_branches to current_depth=1",
            subtitle="Four completed parents become four pending seeds for the deepest wave.",
            active_node="fan_out_branches",
            statuses=build_statuses(levels, done=[*level_3, *level_2], queued=[], active=[]),
            workers=[WorkerTask(path, "fan-out", "seed next wave") for path in level_2],
            queued_count=0,
            stat_lines=("pending branch seeds: 4", "new current_depth: 1"),
            duration_ms=1050,
        ),
        Stage(
            title="8. generate_search_queries at current_depth=1",
            subtitle="The deepest query-generation wave fans out across four parents in parallel, producing eight leaf research queries.",
            active_node="generate_search_queries",
            statuses=build_statuses(levels, done=[*level_3, *level_2], queued=level_1, active=[]),
            workers=[WorkerTask(path, "query gen", "2 leaf queries") for path in level_2],
            queued_count=0,
            stat_lines=("parallel query-generation calls now: 4", "wave width after expansion: 8"),
            duration_ms=1200,
        ),
        Stage(
            title="9. execute_research at current_depth=1 | batch 1",
            subtitle="The tree is now wider than the semaphore limit, so only four leaf tasks run while the rest wait in queue.",
            active_node="execute_research",
            statuses=build_statuses(levels, done=[*level_3, *level_2], queued=second_leaf_batch, active=first_leaf_batch),
            workers=[WorkerTask(path, "research", "running") for path in first_leaf_batch],
            queued_count=len(second_leaf_batch),
            stat_lines=("running research tasks: 4", "queued research tasks: 4"),
            duration_ms=1250,
        ),
        Stage(
            title="10. execute_research at current_depth=1 | batch 2",
            subtitle="As worker slots free up, the remaining four leaf tasks start. Peak tree width is 8, but runtime parallelism stays capped at 4.",
            active_node="execute_research",
            statuses=build_statuses(levels, done=[*level_3, *level_2, *first_leaf_batch], queued=[], active=second_leaf_batch),
            workers=[WorkerTask(path, "research", "running") for path in second_leaf_batch],
            queued_count=0,
            stat_lines=("running research tasks: 4", "queued research tasks: 0"),
            duration_ms=1250,
        ),
        Stage(
            title="11. assemble_final_context -> generate_report",
            subtitle="Once current_depth reaches 1, the graph stops descending, assembles the hierarchical context from 14 tree nodes, and writes the report.",
            active_node="generate_report",
            statuses=build_statuses(levels, done=[*level_3, *level_2, *level_1], queued=[], active=[]),
            workers=[
                WorkerTask("assemble_final_context", "context", "sort + deduplicate"),
                WorkerTask("generate_report", "report", "final markdown"),
            ],
            queued_count=0,
            stat_lines=("total research tasks executed: 14", "peak parallel research: min(8, concurrency_limit=4) = 4"),
            duration_ms=1600,
        ),
    ]


def save_gif(frames: Sequence[Image.Image], durations: Sequence[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=list(durations),
        loop=0,
        optimize=False,
        disposal=2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deep research concurrency GIF.")
    parser.add_argument("--breadth", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.breadth < 1 or args.depth < 1 or args.concurrency < 1:
        raise SystemExit("breadth, depth, and concurrency must all be positive integers")
    if args.depth != 3 or args.breadth != 2:
        raise SystemExit("This renderer currently targets the requested breadth=2, depth=3 animation.")

    levels = build_tree_levels(args.breadth, args.depth)
    positions = compute_node_positions(levels, args.depth)
    total_queries = sum(len(paths) for paths in levels.values())
    stages = build_stages(levels, args.depth, args.concurrency)
    frames = [
        render_frame(
            stage,
            index,
            len(stages),
            breadth=args.breadth,
            depth=args.depth,
            concurrency_limit=args.concurrency,
            levels=levels,
            positions=positions,
            total_queries=total_queries,
        )
        for index, stage in enumerate(stages)
    ]
    save_gif(frames, [stage.duration_ms for stage in stages], args.output)
    print(f"Saved GIF to {args.output}")


if __name__ == "__main__":
    main()
