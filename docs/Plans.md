# Master Implementation Plan (PLAN.md)

## Agent Execution Protocol
Before starting any task in this plan, the AI Coding Agent **MUST** adhere to the following workflow:
1. **Read First:** Open and read `AGENTS.md` (in project root) and the specific **Reference Docs** listed under the current task.
2. **One Task at a Time:** Execute strictly **one** atomic task per iteration. Never combine multiple tasks unless explicitly instructed.
3. **Test-Driven Verification:** Run the **Verification Command** specified at the end of the task before marking it as complete (`[x]`).
4. **Strict Typing & Formatting:** Run `ruff check` and `mypy` on all touched files before concluding any task.

---

## Phase 1: Project Scaffolding, Configuration & Infrastructure

### [x] Task 1.1: Initialize Directory Structure & Tooling Config
- **Objective:** Create the complete Clean Architecture folder tree and configure `pyproject.toml` for `ruff`, `mypy`, and `pytest`.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 6: Project Structure), `docs/CODE_STYLE.md` (Section 8: Tooling & Enforcement).
- **Target Files:**
  - `pyproject.toml`
  - `requirements.txt`
  - `.gitignore`
  - `src/__init__.py`, `src/common/__init__.py`, `src/ingestion/__init__.py`, `src/retrieval/__init__.py`, `src/graph/__init__.py`, `src/services/__init__.py`, `src/api/__init__.py`
  - `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `evals/__init__.py`
- **Implementation Details:**
  - Set `line-length = 100`, target Python 3.11+, enable isort (`I001`) in `pyproject.toml`.
  - Block `.env*`, `*.pkl`, `data/processed/*`, and `__pycache__/` in `.gitignore`.
- **Verification:** `ruff check . && mypy src`

### [x] Task 1.2: Docker Compose Infrastructure Setup
- **Objective:** Configure containerized services for Qdrant, Redis, PostgreSQL, and Langfuse.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 3 & 5), `docs/DATABASE.md` (Section 1).
- **Target Files:**
  - `docker-compose.yml`
  - `Makefile`
- **Implementation Details:**
  - Define services: `qdrant` (ports 6333, 6334), `redis` (port 6379), `postgres` (port 5432), and `langfuse` (port 3000).
  - Create a `Makefile` with targets: `up`, `down`, `lint`, `test`, `eval`.
- **Verification:** `docker compose config` (validates YAML syntax without errors).

### [x] Task 1.3: Environment & YAML Configuration Loader
- **Objective:** Implement strongly-typed configuration management using `pydantic-settings` and `configs/config.yaml`.
- **Reference Docs:** `docs/SECURITY.md` (Section 4: Secrets Management), `docs/CODE_STYLE.md` (Section 5: Typing & Pydantic V2).
- **Target Files:**
  - `.env.example`
  - `configs/config.yaml`
  - `src/common/config.py`
  - `tests/unit/test_config.py`
- **Implementation Details:**
  - Implement `Settings(BaseSettings)` in `src/common/config.py` loading environment variables (`LLM_BASE_URL`, `QDRANT_HOST`, `REDIS_HOST`, `API_KEY`, etc.) and retrieval hyperparameters (`top_k_dense=20`, `top_k_sparse=20`, `rrf_k=60`, `rerank_top_n=5`, `crag_correct_threshold=0.70`, `crag_ambiguous_threshold=0.35`).
- **Verification:** `pytest tests/unit/test_config.py -v`

### [x] Task 1.4: Custom Exception Hierarchy & Structured Logger
- **Objective:** Build the domain exception classes and JSON/context-aware logger.
- **Reference Docs:** `docs/CODE_STYLE.md` (Section 6: Error Handling, Section 7: Logging), `docs/API.md` (Section 3: RFC 7807).
- **Target Files:**
  - `src/common/exceptions.py`
  - `src/common/logger.py`
  - `tests/unit/test_exceptions.py`
- **Implementation Details:**
  - Create `BaseRAGException(message, code, status_code)` and subclasses: `DocumentParsingError`, `VectorStoreTimeoutError`, `InferenceEngineUnavailableError`, `InvalidInputSecurityError`.
  - Implement structured logger supporting `trace_id` injection via `contextvars`.
- **Verification:** `pytest tests/unit/test_exceptions.py -v`

---

## Phase 2: Core Data Schemas & Security Sanitization

### [x] Task 2.1: Core Domain Pydantic V2 Models
- **Objective:** Define strict data contracts for document chunks, retrieval candidates, and citations.
- **Reference Docs:** `docs/DATABASE.md` (Section 3.1: Point Payload Schema), `docs/CODE_STYLE.md` (Section 5).
- **Target Files:**
  - `src/common/schemas.py`
  - `tests/unit/test_schemas.py`
- **Implementation Details:**
  - Create `DocumentChunkPayload`, `RetrievedChunk`, `CitationItem`, and `EvaluationRecord` with `ConfigDict(strict=True)`.
  - Implement deterministic UUIDv4/MD5 point ID generator helper from `source_file + ":" + str(chunk_id)`.
- **Verification:** `pytest tests/unit/test_schemas.py -v`

### [x] Task 2.2: Bilingual Input Sanitizer & Security Normalizer
- **Objective:** Sanitize user queries and normalize Persian/English characters against prompt injection and unicode spoofing.
- **Reference Docs:** `docs/SECURITY.md` (Section 3.2: Input Sanitization & Normalization), `docs/DATABASE.md` (Section 4).
- **Target Files:**
  - `src/common/security.py`
  - `tests/unit/test_security.py`
- **Implementation Details:**
  - Unicode NFC normalization, Arabic to Persian character mapping (ي -> ی, ك -> ک), ZWNJ preservation.
  - Directional override stripping (bidi attacks) and length validation (1 to 1000 characters).
- **Verification:** `pytest tests/unit/test_security.py -v`

---

## Phase 3: Data Ingestion & Technical Markdown Chunker

### [x] Task 3.1: Offline & Live Documentation Seed Pipeline
- **Objective:** Download official documentation subsets (Docker, PostgreSQL, Linux) and create offline fixture docs.
- **Reference Docs:** `docs/PRD.md` (Section 3: Target Technical Domains), `docs/ARCHITECTURE.md` (Section 4.1).
- **Target Files:**
  - `scripts/fetch_docs.py`
  - `data/raw/` (seed directory containing Docker, Postgres, Linux documentation)
- **Implementation Details:**
  - Fetch raw Markdown guides from GitHub repositories with offline fallback fixtures.
  - Store files under `data/raw/docker/`, `data/raw/postgres/`, `data/raw/linux/`.
- **Verification:** `pytest tests/unit/test_fetch_docs.py -v`

### [x] Task 3.2: Markdown Semantic Chunker with Code Block & Table Integrity
- **Objective:** Build a domain-specific Markdown chunker that preserves code blocks, tables, and hierarchy context.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.1 & ADR-001), `docs/DATABASE.md` (Section 2: Chunking Strategy).
- **Target Files:**
  - `src/ingestion/chunker.py`
  - `tests/unit/test_chunker.py`
- **Implementation Details:**
  - Parse `#`, `##`, `###` headings and associate closest parent heading with every chunk.
  - Guard fenced code blocks (```) and Markdown tables (|---|) from being split across chunk boundaries.
  - Soft limit: 512 tokens (using `tiktoken` cl100k_base), 10% overlap (50 tokens) for long prose.
  - Detect primary chunk language (`"fa"` vs `"en"`) based on character script ratio.
  - Calculate token count using `tiktoken`.
- **Verification:** `pytest tests/unit/test_chunker.py -v`

---

## Phase 4: Dual Indexing (BGE-M3 Dense + BM25 Sparse)

### [x] Task 4.1: Local BGE-M3 Embedding Wrapper
- **Objective:** Implement the local multilingual embedding generator using `BAAI/bge-m3`.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.1 & ADR-002), `docs/CODE_STYLE.md` (Section 4: Async Rules).
- **Target Files:**
  - `src/ingestion/embedder.py`
  - `tests/unit/test_embedder.py`
- **Implementation Details:**
  - Implement `BGEEmbedder` wrapping `SentenceTransformer("BAAI/bge-m3")`.
  - Provide both synchronous batch embedding (`embed_documents`) and async query embedding (`aembed_query` using `asyncio.to_thread`).
  - Mock model weights in unit tests to run in <1 second.
- **Verification:** `pytest tests/unit/test_embedder.py -v`

### [ ] Task 4.2: Qdrant Vector Store Manager & Upsert Logic
- **Objective:** Manage the `tech_docs_bilingual` Qdrant collection and perform idempotent batched upserts.
- **Reference Docs:** `docs/DATABASE.md` (Section 3.1: Qdrant Vector Collection & Section 4: Deduplication).
- **Target Files:**
  - `src/ingestion/qdrant_store.py`
  - `tests/unit/test_qdrant_store.py`
- **Implementation Details:**
  - Create collection with 1024-dim Cosine distance and HNSW config (`m=16`, `ef_construct=100`).
  - Create payload indexes on `doc_id`, `source_file`, and `language`.
  - Implement batched upsert using deterministic UUIDv4 keys.
- **Verification:** `pytest tests/unit/test_qdrant_store.py -v`

### [ ] Task 4.3: Persistent BM25 Lexical Indexer
- **Objective:** Build and persist the bilingual BM25 inverted index with atomic file writes.
- **Reference Docs:** `docs/DATABASE.md` (Section 1 & Section 5: BM25 Persistence), `docs/ARCHITECTURE.md` (Section 4.1).
- **Target Files:**
  - `src/ingestion/bm25_store.py`
  - `tests/unit/test_bm25_store.py`
- **Implementation Details:**
  - Implement bilingual tokenizer (preserving English error codes/flags like `--ipc=host` and Persian words).
  - Build `BM25Okapi` index over chunks and save atomically via temporary file + `os.replace` to `data/processed/bm25_index.pkl`.
- **Verification:** `pytest tests/unit/test_bm25_store.py -v`

### [ ] Task 4.4: Unified Ingestion Coordinator CLI & Service
- **Objective:** Connect chunker, embedder, Qdrant, and BM25 into a single ingestion service.
- **Reference Docs:** `docs/UX_FLOWS.md` (Flow 4), `docs/ARCHITECTURE.md` (Section 4.1).
- **Target Files:**
  - `src/ingestion/indexer.py`
  - `tests/unit/test_indexer.py`
- **Implementation Details:**
  - Implement `IngestionPipeline.ingest_directory(path)` and `ingest_file(filepath)`.
  - Return summary metrics (total chunks, total tokens, duration in seconds).
- **Verification:** `pytest tests/unit/test_indexer.py -v`

---

## Phase 5: Hybrid Retrieval, RRF Fusion & Cross-Encoder Reranking

### [ ] Task 5.1: Dense & Sparse Retrievers with Degraded Fallback
- **Objective:** Implement async Dense (Qdrant) and Sparse (BM25) retrieval modules with fault tolerance.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.2 & Section 9: Failure Scenarios), `docs/PRD.md` (Feature 1).
- **Target Files:**
  - `src/retrieval/dense.py`
  - `src/retrieval/sparse.py`
  - `tests/unit/test_retrievers.py`
- **Implementation Details:**
  - `DenseRetriever.asearch(query, top_k=20)` queries Qdrant with a 2-second timeout.
  - `SparseRetriever.asearch(query, top_k=20)` queries the loaded BM25 index via `asyncio.to_thread`.
  - If Qdrant times out, log a warning and allow the pipeline to proceed with BM25 results (degraded mode).
- **Verification:** `pytest tests/unit/test_retrievers.py -v`

### [ ] Task 5.2: Reciprocal Rank Fusion (RRF) Engine
- **Objective:** Implement deterministic Reciprocal Rank Fusion to merge Dense and Sparse rankings.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.2: RRF formula), `docs/TESTING.md` (Section 3.1).
- **Target Files:**
  - `src/retrieval/fusion.py`
  - `tests/unit/test_fusion.py`
- **Implementation Details:**
  - Implement:
    $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + r_m(d)}$$
    with default $k = 60$.
  - Deduplicate chunks by `chunk_id` and return Top-15 sorted candidates.
- **Verification:** `pytest tests/unit/test_fusion.py -v`

### [ ] Task 5.3: Cross-Encoder Reranker (`bge-reranker-v2-m3`)
- **Objective:** Rescore the Top-15 RRF candidates using the bilingual Cross-Encoder model and calibrate scores to $[0.0, 1.0]$.
- **Reference Docs:** `docs/PRD.md` (Feature 2), `docs/CODE_STYLE.md` (Section 4: CPU-bound Offloading).
- **Target Files:**
  - `src/retrieval/reranker.py`
  - `tests/unit/test_reranker.py`
- **Implementation Details:**
  - Wrap `CrossEncoder("BAAI/bge-reranker-v2-m3")` with sigmoid activation so scores strictly lie in $[0.0, 1.0]$.
  - Offload inference using `asyncio.to_thread` and return the Top-5 sorted `RetrievedChunk` instances.
- **Verification:** `pytest tests/unit/test_reranker.py -v`

---

## Phase 6: Local vLLM Service, Security Prompts & Redis Cache

### [ ] Task 6.1: Strict XML Prompt Templates & Guardrails
- **Objective:** Create prompt templates with `<retrieved_context>` XML wrappers for generation, CRAG grading, and query rewriting.
- **Reference Docs:** `docs/SECURITY.md` (Section 3.1: Strict Context XML Enclosure, Section 3.3: Output Guardrails).
- **Target Files:**
  - `configs/prompts.yaml`
  - `src/services/prompt_builder.py`
  - `tests/unit/test_prompt_builder.py`
- **Implementation Details:**
  - Enclose context with `<retrieved_context>` and strict instruction: "Answer only based on the context above. If not found, say you don't know."
  - Enforce zero-hallucination refusal rules in bilingual prompts.
- **Verification:** `pytest tests/unit/test_prompt_builder.py -v`

### [ ] Task 6.2: OpenAI-Compatible Local vLLM Client
- **Objective:** Implement asynchronous client to communicate with local vLLM OpenAI-compatible endpoint.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.3 & ADR-003).
- **Target Files:**
  - `src/services/llm_client.py`
  - `tests/unit/test_llm_client.py`
- **Implementation Details:**
  - Use `httpx.AsyncClient` targeting `LLM_BASE_URL`.
  - Support streaming responses (`yield token`) and complete text generation.
  - Handle retry logic (up to 3 retries with exponential backoff) and timeout handling.
- **Verification:** `pytest tests/unit/test_llm_client.py -v`

### [ ] Task 6.3: Redis Exact & Semantic Query Cache
- **Objective:** Implement 2-layer Redis caching (SHA-256 exact match + normalized query key).
- **Reference Docs:** `docs/DATABASE.md` (Section 2.1: Key Formats & Section 2.2: Semantic Cache), `docs/ARCHITECTURE.md` (Section 4.4).
- **Target Files:**
  - `src/services/cache.py`
  - `tests/unit/test_cache.py`
- **Implementation Details:**
  - Key format: `cache:response:{sha256(normalized_query)}`.
  - Default TTL: 86400 seconds (24 hours).
  - Implement `get(query)` and `set(query, response, citations)` methods.
- **Verification:** `pytest tests/unit/test_cache.py -v`

---

## Phase 7: LangGraph Corrective RAG (CRAG) Control Plane

### [ ] Task 7.1: LangGraph CRAG State Definition & Typed Contracts
- **Objective:** Define the complete typed dictionary state for the LangGraph workflow.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.3: CRAG Loop & Section 7: LangGraph State Definition).
- **Target Files:**
  - `src/graph/state.py`
  - `tests/unit/test_graph_state.py`
- **Implementation Details:**
  - Implement `RAGState(TypedDict)` tracking `query`, `trace_id`, `normalized_query`, `retrieved_chunks`, `reranked_chunks`, `relevance_status`, `crag_score`, `rewritten_query`, `transformation_count`, `generation`, `citations`, `refusal_reason`.
- **Verification:** `pytest tests/unit/test_graph_state.py -v`

### [ ] Task 7.2: Control Plane Nodes Implementation
- **Objective:** Implement each atomic graph node: `retrieve_node`, `grade_documents_node`, `rewrite_query_node`, `generate_node`, `refuse_node`.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.3), `docs/SECURITY.md` (Section 3.1).
- **Target Files:**
  - `src/graph/nodes.py`
  - `tests/unit/test_graph_nodes.py`
- **Implementation Details:**
  - `grade_documents_node`: checks top rerank score against `crag_correct_threshold` (0.70) and `crag_ambiguous_threshold` (0.35).
  - `rewrite_query_node`: increments `transformation_count` and calls LLM rewriter.
  - `refuse_node`: sets standard polite bilingual refusal message without calling generator.
- **Verification:** `pytest tests/unit/test_graph_nodes.py -v`

### [ ] Task 7.3: LangGraph Workflow Assembly & Routing Logic
- **Objective:** Assemble `StateGraph`, define conditional edges, and compile the workflow.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.3), `docs/TESTING.md` (Section 3.2).
- **Target Files:**
  - `src/graph/workflow.py`
  - `tests/unit/test_workflow.py`
- **Implementation Details:**
  - Conditional edge from `grade_documents`:
    - `CORRECT` -> `generate`
    - `AMBIGUOUS` -> `rewrite_query` (if `transformation_count < 1`, else `refuse`)
    - `INCORRECT` -> `refuse`
  - Compile with checkpointer support.
- **Verification:** `pytest tests/unit/test_workflow.py -v`

---

## Phase 8: FastAPI REST API, SSE Streaming & Middleware

### [ ] Task 8.1: API Schemas & RFC 7807 Exception Handlers
- **Objective:** Create API request/response Pydantic models and global RFC 7807 error formatters.
- **Reference Docs:** `docs/API.md` (Section 2: Endpoints & Section 3: RFC 7807), `docs/SECURITY.md` (Section 4.2).
- **Target Files:**
  - `src/api/schemas.py`
  - `src/api/errors.py`
  - `tests/unit/test_api_schemas.py`
- **Implementation Details:**
  - Schemas: `QueryRequest`, `QueryResponse`, `FeedbackRequest`, `IngestionResponse`.
  - Custom exception handler mapping `BaseRAGException` subclasses to standard JSON `{ type, title, status, detail, instance, trace_id }`.
- **Verification:** `pytest tests/unit/test_api_schemas.py -v`

### [ ] Task 8.2: Security Middleware & Dependency Injection
- **Objective:** Build API Key authentication, trace ID context injection, and sliding-window rate limiter.
- **Reference Docs:** `docs/SECURITY.md` (Section 4.1: API Key Auth, Section 4.3: Rate Limiting), `docs/API.md` (Section 1).
- **Target Files:**
  - `src/api/middleware.py`
  - `src/api/dependencies.py`
  - `tests/unit/test_middleware.py`
- **Implementation Details:**
  - Middleware injecting `X-Trace-ID` into `contextvars`.
  - `verify_api_key` dependency verifying `Authorization: Bearer <API_KEY>`.
  - Redis-based sliding-window rate limiter (60 requests/minute per IP).
- **Verification:** `pytest tests/unit/test_middleware.py -v`

### [ ] Task 8.3: SSE Streaming Endpoint (`/api/v1/query/stream`)
- **Objective:** Implement real-time Server-Sent Events (SSE) streaming endpoint with LangGraph step updates.
- **Reference Docs:** `docs/API.md` (Section 2.2: Streaming Contract), `docs/UX_FLOWS.md` (Flow 1).
- **Target Files:**
  - `src/api/routes.py`
  - `src/main.py`
  - `tests/integration/test_api_streaming.py`
- **Implementation Details:**
  - Stream events: `step` (retrieval, rerank, evaluate), `token` (LLM answer stream), `citations` (final list), `done`.
  - Validate SSE headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`.
- **Verification:** `pytest tests/integration/test_api_streaming.py -v`

---

## Phase 9: Bilingual Chainlit Web UI & Bidirectional Layout

### [ ] Task 9.1: Chainlit Application Setup & Persian/RTL Theming
- **Objective:** Configure Chainlit with custom CSS supporting proper Persian right-to-left layout and bilingual fonts (Vazirmatn).
- **Reference Docs:** `docs/DESIGN_SYSTEM.md` (Section 3: RTL Rules, Section 4: Typography), `docs/UX_FLOWS.md` (Flow 1 & 2).
- **Target Files:**
  - `chainlit.md`
  - `.chainlit/config.toml`
  - `public/custom.css`
  - `app.py`
- **Implementation Details:**
  - Add Google Fonts / local font definition for `Vazirmatn`.
  - Set `dir="auto"` on message containers and define bidirectional styling for mixed Persian/English code spans.
- **Verification:** Manual UI inspection via `chainlit run app.py --headless`.

### [ ] Task 9.2: Interactive Chat Workflow, Step Collapsing & Citations Drawer
- **Objective:** Connect Chainlit frontend to LangGraph stream, displaying real-time reasoning steps, citations, and feedback buttons.
- **Reference Docs:** `docs/UX_FLOWS.md` (Flow 1: Query Execution & Flow 3: Feedback), `docs/DESIGN_SYSTEM.md` (Section 5: Components).
- **Target Files:**
  - `app.py`
  - `tests/unit/test_chainlit_helpers.py`
- **Implementation Details:**
  - Implement `@cl.on_message` streaming from FastAPI backend or internal graph runner.
  - Render collapsible `cl.Step` elements: `🔍 Retrieval (Dense + Sparse)`, `⚖️ Fusion & Reranking`, `🎯 Quality Gate (CRAG)`.
  - Render citation pills and clickable drawer sidebar with source file and chunk ID.
- **Verification:** `pytest tests/unit/test_chainlit_helpers.py -v`

---

## Phase 10: Ragas Metric Evaluation & CI/CD Quality Gates

### [ ] Task 10.1: Golden Evaluation Dataset Builder
- **Objective:** Assemble 30 curated bilingual technical query-context-ground_truth evaluation samples.
- **Reference Docs:** `docs/TESTING.md` (Section 4: Ragas Gating Suite), `docs/PRD.md` (Section 5: Evaluation Criteria).
- **Target Files:**
  - `evals/datasets/golden_bilingual_eval.json`
  - `evals/eval_runner.py`
- **Implementation Details:**
  - 15 Persian queries, 15 English queries covering Docker, Postgres, Linux.
  - Include tricky edge cases: English command flags in Persian queries, ambiguous error logs, out-of-domain refusal test cases.
- **Verification:** Python script verifying JSON schema of evaluation records.

### [ ] Task 10.2: Ragas Automated Metric Evaluation Runner
- **Objective:** Execute Ragas evaluation for Faithfulness, Answer Relevance, Context Precision, and Context Recall.
- **Reference Docs:** `docs/TESTING.md` (Section 4.2: Target Thresholds).
- **Target Files:**
  - `evals/eval_runner.py`
  - `tests/integration/test_ragas_gating.py`
- **Implementation Details:**
  - Compute scores using local vLLM judge model (`Qwen/Qwen2.5-7B-Instruct`).
  - Gating thresholds:
    - Faithfulness: $\ge 0.85$
    - Answer Relevance: $\ge 0.85$
    - Context Precision: $\ge 0.80$
    - Context Recall: $\ge 0.75$
  - Return exit code 1 if any metric fails gating threshold.
- **Verification:** `pytest tests/integration/test_ragas_gating.py -v -m eval`

### [ ] Task 10.3: GitHub Actions CI Pipeline Configuration
- **Objective:** Configure complete automated GitHub Actions CI pipeline running lint, type checks, unit tests, and integration suites.
- **Reference Docs:** `docs/TESTING.md` (Section 5: CI Pipeline), `docs/CODE_STYLE.md` (Section 8).
- **Target Files:**
  - `.github/workflows/ci.yml`
- **Implementation Details:**
  - Job 1: `lint-and-typecheck` (`ruff check .`, `ruff format --check .`, `mypy src`).
  - Job 2: `unit-tests` (`pytest tests/unit -v --cov=src`).
  - Job 3: `integration-tests` (with Qdrant and Redis service containers).
- **Verification:** Local validation via `make test` or `act`.
