# Task 7.3: LangGraph Workflow Assembly & Routing Logic

Assemble the LangGraph `StateGraph(RAGState)` Corrective RAG (CRAG) state machine, implement deterministic conditional routing after document grading (`generate`, `rewrite_query`, `refuse`), support loop-back retrieval after query rewriting, and compile the graph with checkpointer support (`MemorySaver` / `BaseCheckpointSaver`).

## User Review & Critical Decisions

> [!IMPORTANT]
> Review the proposed workflow architecture and routing design below before proceeding to implementation:

- **Deterministic Conditional Edge (`route_after_grading`)**: Evaluates `state["relevance_status"]` and `state["transformation_count"]` against `settings.max_query_transformations` (default `1`):
  - `"correct"` $\rightarrow$ `"generate"`
  - `"ambiguous"` $\rightarrow$ `"rewrite_query"` if `transformation_count < max_query_transformations` (default `1`), otherwise `"refuse"` (defensive guard preventing infinite rewrite loops).
  - `"incorrect"` (or any unhandled state) $\rightarrow$ `"refuse"`
- **Bounded Corrective Loop**: `"rewrite_query"` connects back to `"retrieve"`, which automatically prioritizes `state["rewritten_query"]` (implemented in Task 7.2) and re-runs hybrid search, RRF fusion, and Cross-Encoder reranking before re-entering `"grade_documents"`.
- **Checkpointer Support**: `compile_crag_workflow` / `build_crag_workflow` accepts an optional `checkpointer: BaseCheckpointSaver[Any] | bool | None = None` (automatically instantiating `langgraph.checkpoint.memory.MemorySaver` when `checkpointer=True` or when a custom saver is provided), enabling stateful thread persistence and inspection across invocations.

## 1. Overview & Core Concept

- **What It Does**: Connects the atomic execution nodes from `src/graph/nodes.py` (`retrieve_node`, `grade_documents_node`, `rewrite_query_node`, `generate_node`, `refuse_node`) into a compiled LangGraph state machine (`src/graph/workflow.py`).
- **Target Audience / Persona**: Downstream FastAPI SSE streaming endpoint (`Task 8.3: /api/v1/query/stream`), Chainlit UI (`Task 9.2`), and Ragas evaluation runner (`Task 10.2`).
- **Key Value**: Enforces strict zero-hallucination quality gates and bounded query refinement (at most 1 rewrite cycle) before any LLM answer generation occurs.

## 2. User Experience & Execution Flow

- **Path A — Direct Grounded Generation (`CORRECT`)**:
  1. `START` $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents` (score $\ge 0.70$, status `"correct"`).
  2. Conditional edge routes to `generate` $\rightarrow$ `END`.
- **Path B — Ambiguous Query Refinement (`AMBIGUOUS` $\rightarrow$ Rewrite Loop)**:
  1. `START` $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents` ($0.35 \le \text{score} < 0.70$, `transformation_count == 0`, status `"ambiguous"`).
  2. Conditional edge routes to `rewrite_query` (increments `transformation_count` to `1` and sets `rewritten_query`).
  3. Edge routes from `rewrite_query` $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents`.
  4. On second grading pass, routes either to `generate` (if new score $\ge 0.70$) or `refuse` (if still $< 0.70$, since `transformation_count == 1`).
- **Path C — Immediate Refusal (`INCORRECT`)**:
  1. `START` $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents` (score $< 0.35$ or empty chunks, status `"incorrect"`).
  2. Conditional edge routes directly to `refuse` $\rightarrow$ `END` without invoking LLM generation.

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: Dual Guard on Query Transformation Limit**
  - *Chosen Approach*: Enforce `transformation_count < max_query_transformations` (default `1`) in both `grade_documents_node` and the conditional edge router `route_after_grading`.
  - *Why*: Guarantees loop termination at the graph routing layer even if a caller injects custom state or tests routing logic in isolation.
- **Decision 2: Dependency-Injected Workflow Builder & Checkpointer**
  - *Chosen Approach*: Provide `create_crag_state_graph(nodes, settings)` (returns uncompiled `StateGraph`) and `build_crag_workflow(nodes, settings, checkpointer)` (returns `CompiledStateGraph`), plus a `CRAGWorkflow` facade class with `.ainvoke()` and `.astream()`.
  - *Why*: Allows unit tests and downstream FastAPI/Chainlit services to inject mocked `CRAGNodes` and `MemorySaver` checkpointers without initializing real Qdrant/vLLM connections.

## 4. Technical Architecture & Data Strategy

```
                           ┌───────────────┐
                           │     START     │
                           └───────┬───────┘
                                   │
                                   ▼
                           ┌───────────────┐
              ┌───────────►│   retrieve    │
              │            └───────┬───────┘
              │                    │
              │                    ▼
     ┌────────┴────────┐   ┌───────────────┐
     │  rewrite_query  │◄──┤grade_documents│
     └─────────────────┘   └───┬───────┬───┘
       AMBIGUOUS               │       │
       (count < 1)     CORRECT │       │ INCORRECT
                               │       │ (or count >= 1)
                               ▼       ▼
                       ┌──────────┐ ┌──────────┐
                       │ generate │ │  refuse  │
                       └────┬─────┘ └────┬─────┘
                            │            │
                            ▼            ▼
                         ┌─────────────────┐
                         │       END       │
                         └─────────────────┘
```

### Target Files & Signatures
- `src/graph/workflow.py`:
  - `NodeName = Literal["retrieve", "grade_documents", "rewrite_query", "generate", "refuse"]`
  - `RouteDestination = Literal["generate", "rewrite_query", "refuse"]`
  - `route_after_grading(state: RAGState, max_transformations: int = 1) -> RouteDestination`
  - `create_crag_state_graph(nodes: CRAGNodes | None = None, settings: Settings | None = None) -> StateGraph`
  - `build_crag_workflow(nodes: CRAGNodes | None = None, settings: Settings | None = None, checkpointer: BaseCheckpointSaver[Any] | bool | None = None) -> CompiledStateGraph`
  - `CRAGWorkflow`: High-level runner wrapping the compiled graph with `ainvoke(query, trace_id, language, thread_id)` and `astream(...)`.
- `src/graph/__init__.py`:
  - Export `CRAGWorkflow`, `build_crag_workflow`, `create_crag_state_graph`, `route_after_grading`.
- `tests/unit/test_workflow.py`:
  - Unit tests for `route_after_grading` (`correct`, `ambiguous` with `transformation_count=0`, `ambiguous` with `transformation_count>=1`, `incorrect`, case-insensitivity).
  - End-to-end compiled workflow execution tests with mocked `CRAGNodes`:
    1. `CORRECT` path (`retrieve` $\rightarrow$ `grade_documents` $\rightarrow$ `generate`).
    2. `AMBIGUOUS` $\rightarrow$ `rewrite_query` $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents` $\rightarrow$ `generate` (successful recovery after rewrite).
    3. `AMBIGUOUS` $\rightarrow$ `rewrite_query` $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents` $\rightarrow$ `refuse` (exhausted transformation budget).
    4. `INCORRECT` path (`retrieve` $\rightarrow$ `grade_documents` $\rightarrow$ `refuse`).
    5. Checkpointer compilation & thread state persistence verification (`MemorySaver` with `configurable={"thread_id": "..."}`).

### Acceptance Criteria & Verification
1. **Conditional Routing**:
   - `CORRECT` routes to `generate`.
   - `AMBIGUOUS` routes to `rewrite_query` when `transformation_count < 1`, and to `refuse` when `transformation_count >= 1`.
   - `INCORRECT` routes to `refuse`.
2. **Loop & Termination**:
   - `rewrite_query` edges back to `retrieve`, and the graph terminates deterministically at `END` after `generate` or `refuse`.
3. **Checkpointer Support**:
   - Workflow compiles cleanly both with and without a `MemorySaver` / `BaseCheckpointSaver` checkpointer and persists state across thread runs when enabled.
4. **Verification Commands**:
   - `pytest tests/unit/test_workflow.py -v` passes with 0 failures.
   - `ruff check src tests` and `ruff format --check src tests` pass with 0 errors.
   - `mypy --strict src` passes with 0 errors.
