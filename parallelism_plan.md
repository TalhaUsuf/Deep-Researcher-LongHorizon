# Plan: Parallel Deep Research with Global Semaphore + Hierarchical Report

## Context
Two problems in the LangGraph deep research module (`deep_researcher_langgraph/` on branch `feature/langgraph-deep-research-checkpointer`):

1. **Sequential execution**: DFS stack processes ONE branch at a time — O(total_branches) wall time
2. **Flat context**: Results are dumped into flat lists, losing the tree structure needed for hierarchical report headings (coarse → fine detail)

The goal: level-by-level parallel execution with a global semaphore, AND path-tagged results that produce a properly structured report where depth levels map to heading levels.

**Branch:** `feature/langgraph-deep-research-checkpointer`

---

## Current Architecture (sequential — DFS stack)

```
pick_next_branch → generate_search_queries → execute_research → fan_out_branches
     ↑                    (1 branch)           (1 batch)              │
     └────────────────────────────────────────────────────────────────┘
                        (pop next item from stack, repeat)
```

Each loop iteration handles ONE branch. With depth=2, breadth=4: 4 level-1 branches processed one at a time, then 4×2=8 level-2 branches one at a time = 12 sequential iterations.

## Target Architecture (parallel by level)

```
generate_research_plan
         │
         v
generate_search_queries  ◄──────────┐
  (all branches at level,            │
   each query tagged with path)      │
         │                           │
         v                           │
execute_research                     │
  (ALL queries concurrent,           │
   global semaphore, results         │
   carry path + topic + depth)       │
         │                           │
         v                           │
should_continue_deeper?              │
    /           \                    │
"go_deeper"   "done"                │
    │              │                 │
    v              v                 │
fan_out_branches   assemble_final   │
  (children inherit    context      │
   parent paths)    (sort by path,  │
    │                render tree)   │
    └────────────────────────────────┘
                         │
                         v
                  generate_report
                  (hierarchical prompt) → END
```

Each loop iteration handles ALL branches at one depth level concurrently. With depth=2, breadth=4: 1 iteration at level 1 (4 concurrent), 1 iteration at level 2 (8 concurrent, semaphore-limited) = 2 iterations.

---

## Timeline Comparison

### Old (DFS stack — depth=3, breadth=2)

```
TIME ─────────────────────────────────────────────────────────────────────────►
│▓ A,B ▓│▓ A' ▓│▓A.1.a│▓A.1.b│▓A.2.a│▓A.2.b│▓ B' ▓│▓B.1.a│▓B.1.b│▓B.2.a│▓B.2.b│▓report▓│
←─────────────── 7 sequential execute_research calls ──────────────────────────→
```

### New (level-by-level parallel — depth=3, breadth=2)

```
TIME ───────────────────────────────────────────────►
│▓▓ A,B ▓▓│▓▓ A.1,A.2,B.1,B.2 ▓▓│▓▓ all 8 leaf queries ▓▓│▓report▓│
←──── 3 execute_research calls ────→
```

### Full Execution Trace (new — depth=3, breadth=2, concurrency=8)

```
 PHASE 1: generate_research_plan
 │
 │  → web search → follow-up questions → combined_query
 │  → global_semaphore(8) created
 │  → pending_branches = []
 │
 PHASE 2: generate_search_queries (combined_query, breadth=2)
 │
 │  → 2 queries: [A, B] with paths ["0", "1"]
 │
 PHASE 3: execute_research ── all through global semaphore(8)
 │
 │  ╔══════════════════════════════════╗
 │  ║  slot: ▓▓▓▓▓ A ▓▓▓▓▓│          ║  2 concurrent researchers
 │  ║  slot: ▓▓▓▓▓▓ B ▓▓▓▓▓▓│        ║  (both fit in semaphore=8)
 │  ╚══════════════════════════════════╝
 │
 │  → 2 results + 2 TreeNodes (path="0" depth=3, path="1" depth=3)
 │  → depth=3 > 1 → "go_deeper"
 │
 PHASE 4: fan_out_branches
 │
 │  → pending_branches = [
 │      {path="0", parent_topic="Hardware"},
 │      {path="1", parent_topic="Algorithms"}
 │    ]  at depth=2
 │
 PHASE 5: generate_search_queries (for A' AND B' concurrently, breadth=2 each)
 │
 │  ╔══════════════════════════════════════════════╗
 │  ║  A' → 2 queries: paths ["0.0", "0.1"]       ║  query generation
 │  ║  B' → 2 queries: paths ["1.0", "1.1"]       ║  runs concurrently
 │  ╚══════════════════════════════════════════════╝
 │
 │  → flattened: [A.1, A.2, B.1, B.2]  (4 queries)
 │
 PHASE 6: execute_research ── all through global semaphore(8)
 │
 │  ╔══════════════════════════════════════════════╗
 │  ║  slot: ▓▓▓▓ A.1 ▓▓▓▓│                       ║
 │  ║  slot: ▓▓▓▓▓ A.2 ▓▓▓▓▓│                     ║  4 concurrent
 │  ║  slot: ▓▓▓▓ B.1 ▓▓▓▓│                       ║  researchers
 │  ║  slot: ▓▓▓▓▓▓ B.2 ▓▓▓▓▓▓│                   ║  (all fit in 8)
 │  ╚══════════════════════════════════════════════╝
 │
 │  → 4 results + 4 TreeNodes (paths "0.0","0.1","1.0","1.1", depth=2)
 │  → depth=2 > 1 → "go_deeper"
 │
 PHASE 7: fan_out_branches
 │
 │  → pending_branches = [
 │      {path="0.0", parent_topic="Superconducting"},
 │      {path="0.1", parent_topic="Ion Traps"},
 │      {path="1.0", parent_topic="Shor's"},
 │      {path="1.1", parent_topic="Grover's"}
 │    ]  at depth=1
 │
 PHASE 8: generate_search_queries (for ALL 4 branches, breadth=2 each)
 │
 │  → 8 queries with paths:
 │    "0.0.0", "0.0.1", "0.1.0", "0.1.1",
 │    "1.0.0", "1.0.1", "1.1.0", "1.1.1"
 │
 PHASE 9: execute_research ── all through global semaphore(8)
 │
 │  ╔══════════════════════════════════════════════════════════════╗
 │  ║  slot: ▓▓▓▓ A.1.a ▓▓▓▓│                                    ║
 │  ║  slot: ▓▓▓▓▓ A.1.b ▓▓▓▓▓│                                  ║
 │  ║  slot: ▓▓▓ A.2.a ▓▓▓│                                      ║  8 concurrent
 │  ║  slot: ▓▓▓▓▓ A.2.b ▓▓▓▓▓│                                  ║  researchers
 │  ║  slot: ▓▓▓▓ B.1.a ▓▓▓▓│                                    ║  (all fit in 8)
 │  ║  slot: ▓▓▓▓▓▓ B.1.b ▓▓▓▓▓▓│                                ║
 │  ║  slot: ▓▓▓ B.2.a ▓▓▓│                                      ║
 │  ║  slot: ▓▓▓▓ B.2.b ▓▓▓▓│                                    ║
 │  ╚══════════════════════════════════════════════════════════════╝
 │
 │  → 8 results + 8 TreeNodes (depth=1)
 │  → depth=1, not > 1 → "done"
 │
 PHASE 10: assemble_final_context
 │  → sort research_tree by path → render hierarchical markdown
 │
 PHASE 11: generate_report → END
```

---

## Files to Modify

All under `deep_researcher_langgraph/`:

| File | Changes |
|------|---------|
| `state.py` | Add `path`/`parent_topic` to `BranchItem`; add `TreeNode`; add `research_tree`, `pending_branches`, `global_semaphore` to state; remove `branch_stack` |
| `schemas.py` | Add `path` field to `SearchQueryItem` |
| `nodes.py` | Rewrite 5 nodes for parallel + path tracking; remove `pick_next_branch` and `has_more_work` |
| `prompts.py` | Update `GENERATE_REPORT_PROMPT` for hierarchical context |
| `graph.py` | Simplify topology (remove stack-based loop) |

---

## Detailed Changes

### 1. `state.py`

**Update `BranchItem`:**
```python
class BranchItem(TypedDict):
    query: str
    depth: int
    path: str            # NEW: tree path e.g. "0", "0.1", "1.0.2"
    parent_topic: str    # NEW: research_goal of parent, becomes section heading
```

**Add `TreeNode`:**
```python
class TreeNode(TypedDict):
    """A single node in the research tree, carrying its position and findings."""
    path: str              # "0", "0.1", "1.0.2" etc.
    depth_level: int       # which depth this was researched at (3=shallowest, 1=deepest)
    topic: str             # research_goal — becomes section heading
    learnings: List[str]
    citations: Dict[str, str]
    context: str
```

**Add reducer:**
```python
def _replace(a, b):
    """Last-writer-wins reducer for fields that should be overwritten, not accumulated."""
    return b if b is not None else a
```

**Update `DeepResearchState`:**
```python
# REMOVE:
#   branch_stack: List[BranchItem]

# ADD:
pending_branches: List[BranchItem]                    # replaced each level (no Annotated — last writer wins)
global_semaphore: Any                                  # set once, never merged
research_tree: Annotated[List[TreeNode], operator.add] # accumulated across all levels
```

Keep existing flat accumulators (`all_learnings`, `all_citations`, etc.) — they're still used for dedup in `assemble_final_context`. The `research_tree` is the NEW hierarchical structure used for report generation.

---

### 2. `schemas.py`

**Update `SearchQueryItem`:**
```python
class SearchQueryItem(BaseModel):
    query: str = Field(description="A specific search query")
    research_goal: str = Field(description="The research goal this query aims to achieve")
    path: str = Field(default="", description="Tree path for hierarchical ordering")
    parent_topic: str = Field(default="", description="Parent section topic")
```

The `path` and `parent_topic` are NOT populated by the LLM — they're set programmatically in `generate_search_queries` after the LLM returns.

---

### 3. `nodes.py`

#### `generate_research_plan` — add global semaphore + init tree

```python
return {
    ...existing fields...,
    "global_semaphore": asyncio.Semaphore(state.get("concurrency_limit", 4)),
    "pending_branches": [],
    "research_tree": [],
}
```

#### `generate_search_queries` — handle multiple branches + assign paths

Two cases:

**First call** (pending_branches empty): generate queries from `combined_query`, assign root paths:
```python
# After LLM returns queries:
for i, q in enumerate(search_queries):
    q["path"] = str(i)                     # "0", "1", "2", ...
    q["parent_topic"] = state["query"]     # root topic
```

**Subsequent calls** (pending_branches non-empty): generate queries for EACH branch concurrently, assign child paths:
```python
semaphore = state["global_semaphore"]
all_search_queries = []

async def gen_for_branch(branch, branch_idx):
    async with semaphore:
        # Call LLM to generate queries for this branch
        messages = GENERATE_SEARCH_QUERIES_PROMPT.format_messages(
            query=branch["query"], num_queries=breadth
        )
        response = await llm_service.invoke_structured(strategic_llm, SearchQueriesResponse, messages)
        queries = []
        for j, q in enumerate(response.queries[:breadth]):
            queries.append({
                "query": q.query,
                "research_goal": q.research_goal,
                "path": f"{branch['path']}.{j}",           # "0.0", "0.1", "1.0", ...
                "parent_topic": branch["parent_topic"],
            })
        return queries

tasks = [gen_for_branch(b, i) for i, b in enumerate(pending)]
results = await asyncio.gather(*tasks)
all_search_queries = [q for batch in results for q in batch]
```

#### `execute_research` — global semaphore + tag results with tree info

Replace local semaphore:
```python
semaphore = state["global_semaphore"]   # instead of asyncio.Semaphore(concurrency_limit)
```

Each result now carries tree metadata from its query:
```python
return {
    ...existing fields...,
    "path": serp_query["path"],
    "parent_topic": serp_query.get("parent_topic", ""),
}
```

After gathering all results, build `TreeNode` entries:
```python
tree_nodes = []
for r in results:
    tree_nodes.append(TreeNode(
        path=r["path"],
        depth_level=state["current_depth"],
        topic=r["research_goal"],          # becomes section heading
        learnings=r["learnings"],
        citations=r["citations"],
        context=r["context"],
    ))

return {
    ...existing flat accumulators...,
    "research_tree": tree_nodes,           # appended via operator.add reducer
    "research_results": results,
}
```

#### `fan_out_branches` — children inherit parent paths

```python
async def fan_out_branches(state, config):
    results = state.get("research_results", [])
    current_depth = state["current_depth"]
    new_depth = current_depth - 1

    new_branches = []
    for r in results:
        goal = r.get("research_goal", "")
        follow_ups = " ".join(r.get("follow_up_questions", []))
        branch_query = f"Previous research goal: {goal}\nFollow-up questions: {follow_ups}"
        new_branches.append(BranchItem(
            query=branch_query,
            depth=new_depth,
            path=r["path"],              # inherit parent's path
            parent_topic=r["research_goal"],  # parent heading text
        ))

    return {
        "pending_branches": new_branches,
        "current_depth": new_depth,
        "research_results": [],
    }
```

#### `assemble_final_context` — sort tree + render hierarchy

```python
async def assemble_final_context(state, config):
    tree = state.get("research_tree", [])

    # Sort by path as integer tuples for correct hierarchy ordering
    # "0" < "0.0" < "0.0.0" < "0.0.1" < "0.1" < "1" < "1.0"
    tree_sorted = sorted(tree, key=lambda n: tuple(int(x) for x in n["path"].split(".")))

    # Determine depth range for heading level mapping
    max_depth_level = state["depth"]  # e.g. 3

    # Render hierarchical context
    sections = []
    for node in tree_sorted:
        # Map depth_level to heading: deepest depth_level=1 → most nested
        # heading_level = (max_depth - depth_level + 1) gives: depth3→h1, depth2→h2, depth1→h3
        heading_level = max_depth_level - node["depth_level"] + 1
        heading_prefix = "#" * (heading_level + 1)  # +1 because # is reserved for title

        section = f"{heading_prefix} {node['topic']} [path={node['path']}]\n"
        if node["learnings"]:
            section += "Key findings:\n"
            for learning in node["learnings"]:
                citation = node["citations"].get(learning, "")
                if citation:
                    section += f"- {learning} [Source: {citation}]\n"
                else:
                    section += f"- {learning}\n"
        if node["context"]:
            section += f"\nDetailed context:\n{node['context'][:2000]}\n"
        sections.append(section)

    hierarchical_context = "\n\n".join(sections)

    # Trim to word limit — prioritize shallow nodes (broader context)
    final_context = hierarchical_context
    if _count_words(final_context) > MAX_CONTEXT_WORDS:
        trimmed_sections = []
        total_words = 0
        for section in sections:
            words = _count_words(section)
            if total_words + words <= MAX_CONTEXT_WORDS:
                trimmed_sections.append(section)
                total_words += words
        final_context = "\n\n".join(trimmed_sections)

    return {"final_context": final_context}
```

#### Remove `pick_next_branch` and `has_more_work`

Delete both functions entirely.

#### Update conditional edge

```python
def should_continue_deeper(state):
    if state.get("current_depth", 0) > 1:
        return "go_deeper"
    return "done"
```

---

### 4. `prompts.py`

**Update `GENERATE_REPORT_PROMPT`:**
```python
GENERATE_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research report writer. The research context below is "
        "organized hierarchically with markdown headings (## for broad topics, "
        "### for subtopics, #### for fine details). Preserve this heading hierarchy "
        "in your report — each heading level represents a deeper level of research. "
        "Write detailed, factual, and unbiased reports with citations."
    ),
    (
        "human",
        "Based on the following research query and hierarchically organized context, "
        "write a comprehensive research report.\n\n"
        "Query: {query}\n\n"
        "Hierarchical Research Context:\n{context}\n\n"
        "Tone: {tone}\n\n"
        "Instructions:\n"
        "- Preserve the heading hierarchy from the context (## → ### → ####)\n"
        "- Each top-level section (##) covers a broad aspect of the topic\n"
        "- Sub-sections (###) cover specific areas within each aspect\n"
        "- Detail sections (####) provide fine-grained findings\n"
        "- Include all relevant citations as markdown links\n"
        "- Add an introduction before the first section and a conclusion after the last\n"
        "- The report should be at least 1200 words and include references."
    ),
])
```

---

### 5. `graph.py`

```python
workflow = StateGraph(DeepResearchState)

# Nodes (6, down from 7)
workflow.add_node("generate_research_plan", generate_research_plan)
workflow.add_node("generate_search_queries", generate_search_queries)
workflow.add_node("execute_research", execute_research)
workflow.add_node("fan_out_branches", fan_out_branches)
workflow.add_node("assemble_final_context", assemble_final_context)
workflow.add_node("generate_report", generate_report)

# Entry
workflow.set_entry_point("generate_research_plan")

# Linear
workflow.add_edge("generate_research_plan", "generate_search_queries")
workflow.add_edge("generate_search_queries", "execute_research")

# Branch or finish
workflow.add_conditional_edges(
    "execute_research",
    should_continue_deeper,
    {"go_deeper": "fan_out_branches", "done": "assemble_final_context"},
)

# Level loop
workflow.add_edge("fan_out_branches", "generate_search_queries")

# Final
workflow.add_edge("assemble_final_context", "generate_report")
workflow.add_edge("generate_report", END)
```

**Removed:** `pick_next_branch` node, `has_more_work` conditional, all stack edges.

---

## Path Flow Example (depth=3, breadth=2)

```
Level 1 (depth=3):
  generate_search_queries → queries with paths:
    "0" (goal: "Hardware")
    "1" (goal: "Algorithms")

  execute_research → results:
    path="0", topic="Hardware",    learnings=[...], depth_level=3
    path="1", topic="Algorithms",  learnings=[...], depth_level=3

  fan_out_branches → pending_branches:
    {path="0", parent_topic="Hardware"}
    {path="1", parent_topic="Algorithms"}

Level 2 (depth=2):
  generate_search_queries → queries with paths:
    "0.0" (goal: "Superconducting", parent: "Hardware")
    "0.1" (goal: "Ion Traps",       parent: "Hardware")
    "1.0" (goal: "Shor's",          parent: "Algorithms")
    "1.1" (goal: "Grover's",        parent: "Algorithms")

  execute_research → 4 results, all concurrent, depth_level=2

  fan_out_branches → pending_branches:
    {path="0.0", parent_topic="Superconducting"}
    {path="0.1", parent_topic="Ion Traps"}
    {path="1.0", parent_topic="Shor's"}
    {path="1.1", parent_topic="Grover's"}

Level 3 (depth=1):
  generate_search_queries → 8 queries:
    "0.0.0", "0.0.1", "0.1.0", "0.1.1",
    "1.0.0", "1.0.1", "1.1.0", "1.1.1"

  execute_research → 8 results, all concurrent, depth_level=1

  should_continue_deeper? → depth=1, not > 1 → "done"

assemble_final_context sorts by path:
  "0"     → ## Hardware               (depth_level=3 → ##)
  "0.0"   → ### Superconducting       (depth_level=2 → ###)
  "0.0.0" → #### Josephson Junctions  (depth_level=1 → ####)
  "0.0.1" → #### Transmon Arch.       (depth_level=1 → ####)
  "0.1"   → ### Ion Traps             (depth_level=2 → ###)
  "0.1.0" → #### Gate Fidelity        (depth_level=1 → ####)
  "0.1.1" → #### Error Rates          (depth_level=1 → ####)
  "1"     → ## Algorithms             (depth_level=3 → ##)
  "1.0"   → ### Shor's Algorithm      (depth_level=2 → ###)
  ...
```

---

## Report Output Structure

```
# Quantum Computing: A Comprehensive Analysis    ← LLM-generated title

[Introduction paragraph]                          ← LLM-generated

## 1. Hardware Advances                           ← from path="0", depth_level=3
   Overview findings...

   ### 1.1 Superconducting Qubits                 ← from path="0.0", depth_level=2
      Mid-level findings...

      #### Josephson Junctions                    ← from path="0.0.0", depth_level=1
         Fine-grained detail...

      #### Transmon Architecture                  ← from path="0.0.1", depth_level=1
         Fine-grained detail...

   ### 1.2 Ion Traps                              ← from path="0.1", depth_level=2
      ...

## 2. Algorithms                                  ← from path="1", depth_level=3
   ...

## Conclusion                                     ← LLM-generated
## References                                     ← from accumulated citations
```

---

## Research Tree Visualization

```
                          ┌─── query ───┐
                          │             │
                          A             B
                        ┌─┴─┐         ┌─┴─┐
                       A.1  A.2      B.1  B.2        ← OLD: 1 at a time
                      ┌┴┐  ┌┴┐     ┌┴┐  ┌┴┐            NEW: entire level
                     a  b  a  b    a  b  a  b

  Total queries:  2 + 4 + 8 = 14 (same in both)
  Execute phases: OLD = 7,  NEW = 3
  Speedup:        ~2.3x (and grows with larger breadth/depth)
```

---

## Performance Impact

| Config | Old (DFS) phases | New (level) phases | Speedup |
|--------|------------------|--------------------|---------|
| depth=2, breadth=2 | 3 | 2 | 1.5x |
| depth=2, breadth=4 | 7 | 2 | 3.5x |
| depth=3, breadth=2 | 7 | 3 | 2.3x |
| depth=3, breadth=4 | 28+ | 3 | ~9x |

Global semaphore prevents overload: with concurrency_limit=4 and 8 leaf queries, they run in two batches of 4 — still only 3 phases, just slightly longer per phase.

---

## Tests to Update

| Test file | Changes |
|-----------|---------|
| `test_state_and_schemas.py` | `BranchItem` has `path`/`parent_topic`; `TreeNode` added; `branch_stack` → `pending_branches`; `research_tree` field |
| `test_nodes.py` | Remove `pick_next_branch`/`has_more_work` tests; update `generate_search_queries` for multi-branch + paths; update `execute_research` for global semaphore + tree nodes; update `fan_out_branches` for path inheritance; update `assemble_final_context` for hierarchical rendering |
| `test_graph.py` | Simplified topology (6 nodes, no stack loop) |
| `test_integration.py` | End-to-end flow with path verification |
| `test_prompts.py` | Updated `GENERATE_REPORT_PROMPT` |

---

## Verification
1. `pytest deep_researcher_langgraph/tests/ -v` — all tests pass
2. `python deep_researcher_langgraph/main.py "test query"` — full run with depth=2+
3. Verify log timestamps show concurrent execution across branches at same level
4. Verify final report has correct heading hierarchy (## → ### → ####)
5. Verify path ordering in assembled context matches tree structure
6. Test with `concurrency_limit=1` — sequential fallback, same report structure
7. Test with `depth=1` — no recursion, flat report (all ## headings)
8. Test checkpointer persistence with new state shape
