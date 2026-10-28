# Task 9.2: Interactive Chat Workflow, Step Collapsing & Citations Drawer

Connect the Chainlit bilingual interface (`app.py`) to the LangGraph CRAG streaming workflow (`CRAGWorkflow.astream`), rendering collapsible reasoning steps (`🔍 Retrieval (Dense + Sparse)`, `⚖️ Fusion & Reranking`, `🎯 Quality Gate (CRAG)`), interactive side-drawer citations (`display="side"`), query rewrite notices, and user feedback actions, backed by unit tests in `tests/unit/test_chainlit_helpers.py`.

## User Review & Critical Decisions

> [!IMPORTANT]
> Review the key architectural and UX decisions below before proceeding to implementation:

- **In-Process `CRAGWorkflow` Streaming with Sanitization Boundary**: `@cl.on_message` sanitizes incoming queries via `sanitize_query` (`src.common.security`) and streams directly from `get_crag_workflow().astream(...)` (reusing the singleton provider from `src.api.dependencies`), avoiding redundant HTTP serialization overhead while keeping a clean dependency-injection hook for unit tests.
- **Three Collapsible Reasoning Steps (`cl.Step`)**:
  - `🔍 Retrieval (Dense + Sparse)` — summarizes hybrid candidate count (`retrieved_chunks`) and active query.
  - `⚖️ Fusion & Reranking` — displays top reranked chunks (`reranked_chunks`) with source file, chunk ID, and calibrated Cross-Encoder score pills.
  - `🎯 Quality Gate (CRAG)` — displays CRAG relevance status (`CORRECT`, `AMBIGUOUS`, `INCORRECT`) and confidence score, plus an automatic rewrite notice (`"کوئری شما برای دقت بیشتر به عبارت تخصصی‌تر بازنویسی شد."`) if `rewrite_query` is triggered.
- **Interactive Citations Drawer & Feedback Actions**:
  - Citations are rendered both as inline LTR-isolated pills (`.citation-pill` in `public/custom.css`) and as `cl.Text(..., display="side")` elements so clicking a citation name opens Chainlit's side drawer with the source file, chunk ID / section heading, score, and excerpt.
  - Thumbs-up / thumbs-down `cl.Action` buttons (`feedback_positive`, `feedback_negative`) record structured feedback logs correlated with `trace_id`.

## 1. Overview & Core Concept

- **What It Does**: Upgrades `app.py` from static Task 9.1 scaffolding to a live, streaming Corrective RAG assistant with full transparency into retrieval, reranking, CRAG grading, source attribution, and response feedback.
- **Target Audience / Persona**: DevOps engineers, DBAs, and Linux administrators querying technical documentation in Persian and English.
- **Key Value**: Provides radical transparency (`docs/DESIGN_SYSTEM.md` §1) by exposing candidate scores, CRAG gating decisions, and verifiable document excerpts in a collapsible, RTL-safe interface.

## 2. User Experience & Visual Design

- **Flow 1 — Standard Query Execution (`docs/UX_FLOWS.md` Flow 1)**:
  1. User submits a Persian or English technical question.
  2. Collapsible `cl.Step` accordions (`🔍 Retrieval (Dense + Sparse)`, `⚖️ Fusion & Reranking`, `🎯 Quality Gate (CRAG)`) populate with real-time metrics from the workflow nodes.
  3. Answer tokens stream into the assistant message (`msg.stream_token`), followed by inline citation pills and clickable side-drawer references (`cl.Text(display="side")`).
- **Flow 2 — Ambiguous Query Rewrite (`docs/UX_FLOWS.md` Flow 2)**:
  - When `rewrite_query` executes, the UI displays the bilingual notice (`"کوئری شما برای دقت بیشتر به عبارت تخصصی‌تر بازنویسی شد."`) alongside the refined technical query before running secondary retrieval.
- **Flow 3 — Out-of-Domain Refusal & Feedback (`docs/UX_FLOWS.md` Flow 3)**:
  - When `refuse` executes (`INCORRECT` status), the polite bilingual refusal message streams with zero hallucinated citations, and feedback action buttons allow rating the response.

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: Pure Formatting & Builder Helpers in `app.py`**
  - *Chosen Approach*: Extract deterministic formatting and element-construction logic (`format_retrieval_step_output`, `format_rerank_step_output`, `format_crag_step_output`, `build_citation_elements`, `format_citation_pills_markdown`, `build_feedback_actions`) as pure, strictly-typed functions in `app.py`.
  - *Why*: Enables fast, deterministic unit testing in `tests/unit/test_chainlit_helpers.py` without requiring a live browser or external vector/LLM servers, while preserving full compatibility with `tests/unit/test_chainlit_theming.py`.
- **Decision 2: Side-Drawer Citation Linking via `cl.Text(display="side")`**
  - *Chosen Approach*: Create a `cl.Text` element per citation whose `name` matches the citation label referenced in the message footer (e.g., `[1] docker_networking_bridge.md#chunk-0`), including `source_file`, `chunk_id`, `section_heading`, `relevance_score`, and `snippet`.
  - *Why*: Native Chainlit pattern that automatically turns matching citation names in the message into clickable links opening the side drawer.

## 4. Technical Architecture & Data Strategy

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        User Message (@cl.on_message)                     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│             Input Sanitization (sanitize_query + trace_id)               │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 CRAGWorkflow.astream(query, trace_id, ...)               │
├──────────────────────────────────────────────────────────────────────────┤
│  ├── "retrieve" event:                                                   │
│  │     ├── cl.Step("🔍 Retrieval (Dense + Sparse)")                      │
│  │     └── cl.Step("⚖️ Fusion & Reranking")                              │
│  ├── "grade_documents" event:                                            │
│  │     └── cl.Step("🎯 Quality Gate (CRAG)")                             │
│  ├── "rewrite_query" event:                                              │
│  │     └── Rewrite Alert Banner + Secondary Retrieval Loop               │
│  └── "generate" / "refuse" event:                                        │
│        ├── Stream Answer Tokens (tokenize_generation_stream)             │
│        ├── Attach Side-Drawer Citations (cl.Text display="side")         │
│        └── Attach Feedback Buttons (cl.Action positive / negative)       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Target Files & Signatures
- **`app.py`**:
  - `STEP_RETRIEVAL_NAME = "🔍 Retrieval (Dense + Sparse)"`
  - `STEP_RERANK_NAME = "⚖️ Fusion & Reranking"`
  - `STEP_CRAG_NAME = "🎯 Quality Gate (CRAG)"`
  - `format_retrieval_step_output(retrieved_chunks: list[RetrievedChunk], query: str) -> str`
  - `format_rerank_step_output(reranked_chunks: list[RetrievedChunk]) -> str`
  - `format_crag_step_output(status: str, score: float, rewritten_query: str | None = None) -> str`
  - `format_citation_label(index: int, source_file: str, chunk_id: str | int | None = None) -> str`
  - `format_citation_pills_markdown(citations: list[CitationItem], chunks: list[RetrievedChunk] | None = None) -> str`
  - `build_citation_elements(citations: list[CitationItem], chunks: list[RetrievedChunk] | None = None) -> list[cl.Text]`
  - `build_feedback_actions(trace_id: str) -> list[cl.Action]`
  - `on_chat_start() -> None`
  - `on_message(message: cl.Message) -> None`
  - `on_feedback_action(action: cl.Action) -> None` (`@cl.action_callback`)
- **`tests/unit/test_chainlit_helpers.py`**:
  - Unit tests covering step formatting helpers, citation pill & drawer element builders, feedback action builders, input validation error handling, and end-to-end `@cl.on_message` workflow streaming (happy path, query rewrite path, refusal path) with mocked Chainlit primitives.
- **`docs/Plans.md`**:
  - Mark Task 9.2 as completed (`[x]`).

### Acceptance Criteria & Verification
1. **Collapsible Steps (`cl.Step`)**: Renders `🔍 Retrieval (Dense + Sparse)`, `⚖️ Fusion & Reranking`, and `🎯 Quality Gate (CRAG)` with candidate counts, scores, and CRAG decision states.
2. **Citations & Side Drawer**: Renders inline `.citation-pill` badges and attaches clickable `cl.Text(display="side")` elements containing `source_file`, `chunk_id`, `section_heading`, `relevance_score`, and `snippet`.
3. **Bilingual Rewrite & Refusal Support**: Handles `rewrite_query` notification and `refuse` zero-citation responses cleanly, plus thumbs-up/down feedback actions correlated by `trace_id`.
4. **Verification Commands**:
   - `pytest tests/unit/test_chainlit_helpers.py tests/unit/test_chainlit_theming.py -v` passes with 0 failures.
   - `ruff check app.py tests/unit/test_chainlit_helpers.py` and `mypy --strict app.py src` pass with 0 errors.
