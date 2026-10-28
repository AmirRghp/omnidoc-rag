# Master Implementation Plan (PLAN.md)

## Agent Execution Protocol
Before starting any task in this plan, the AI Coding Agent **MUST** adhere to the following workflow:
1. **Read First:** Open and read `AGENTS.md` (in project root) and the specific **Reference Docs** listed under the current task.
2. **One Task at a Time:** Execute strictly **one** atomic task per iteration. Never combine multiple tasks unless explicitly instructed.
3. **Test-Driven Verification:** Run the **Verification Command** specified at the end of the task before marking it as complete (`[x]`).
4. **Strict Typing & Formatting:** Run `ruff check` and `mypy` on all touched files before concluding any task.

---

## Phase 1: Project Scaffolding, Configuration & Infrastructure

### [ ] Task 1.1: Initialize Directory Structure & Tooling Config
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

### [ ] Task 1.2: Docker Compose Infrastructure Setup
- **Objective:** Configure containerized services for Qdrant, Redis, PostgreSQL, and Langfuse.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 3 & 5), `docs/DATABASE.md` (Section 1).
- **Target Files:**
  - `docker-compose.yml`
  - `Makefile`
- **Implementation Details:**
  - Define services: `qdrant` (ports 6333, 6334), `redis` (port 6379), `postgres` (port 5432), and `langfuse` (port 3000).
  - Create a `Makefile` with targets: `up`, `down`, `lint`, `test`, `eval`.
- **Verification:** `docker compose config` (validates YAML syntax without errors).

### [ ] Task 1.3: Environment & YAML Configuration Loader
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

### [ ] Task 1.4: Custom Exception Hierarchy & Structured Logger
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

### [ ] Task 2.1: Core Domain Pydantic V2 Models
- **Objective:** Define strict data contracts for document chunks, retrieval candidates, and citations.
- **Reference Docs:** `docs/DATABASE.md` (Section 3.1: Point Payload Schema), `docs/CODE_STYLE.md` (Section 5).
- **Target Files:**
  - `src/common/schemas.py`
  - `tests/unit/test_schemas.py`
- **Implementation Details:**
  - Create `DocumentChunkPayload`, `RetrievedChunk`, `CitationItem`, and `EvaluationRecord` with `ConfigDict(strict=True)`.
  - Implement deterministic UUIDv4/MD5 point ID generator helper from `source_file + ":" + str(chunk_id)`.
- **Verification:** `pytest tests/unit/test_schemas.py -v`

### [ ] Task 2.2: Bilingual Input Sanitizer & Security Normalizer
- **Objective:** Sanitize user queries and normalize Persian/English characters against prompt injection and unicode spoofing.
- **Reference Docs:** `docs/SECURITY.md` (Section 3.2: Input Sanitization & Normalization), `docs/DATABASE.md` (Section 4).
- **Target Files:**
  - `src/common/security.py`
  - `tests/unit/test_security.py`
- **Implementation Details:**
  - Normalize Arabic `ي` and `ك` to standard Persian `ی` and `ک`.
  - Strip directional override unicode characters (`U+202E`, `U+202B`, etc.).
  - Enforce max query length of 1,000 characters; raise `InvalidInputSecurityError` if exceeded or empty.
- **Verification:** `pytest tests/unit/test_security.py -v`

---

## Phase 3: Document Acquisition, Parsing & Semantic Chunking

### [ ] Task 3.1: Technical Documentation Fetcher Script
- **Objective:** Create an automated script to fetch official technical Markdown files (e.g., Docker Engine manuals) and add sample Persian technical guides.
- **Reference Docs:** `docs/PRD.md` (Section 2 & 10), `docs/UX_FLOWS.md` (Flow 4).
- **Target Files:**
  - `scripts/fetch_docs.py`
  - `data/raw/sample_persian_docker_guide.md`
- **Implementation Details:**
  - Download 25–30 Markdown files from GitHub API into `data/raw/`.
  - Create a rich bilingual Persian/English Markdown troubleshooting guide (`sample_persian_docker_guide.md`) containing common Docker/PostgreSQL error codes (`ERR_DB_TIMEOUT`, `502 Bad Gateway`, `--ipc=host`, `permission denied`) to ensure bilingual test coverage.
- **Verification:** `python scripts/fetch_docs.py` (verifies files exist in `data/raw/`).

### [ ] Task 3.2: Header-Aware & Code-Safe Semantic Chunker
- **Objective:** Implement a Markdown chunker that splits by headers while keeping code blocks and tables intact.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.1), `docs/TESTING.md` (Section 3.2: Chunker Structural Integrity).
- **Target Files:**
  - `src/ingestion/chunker.py`
  - `tests/unit/test_chunker.py`
- **Implementation Details:**
  - Implement `MarkdownSemanticChunker`:
    - Parse Markdown headers (`#`, `##`, `###`) and attach the nearest heading to `section_heading` metadata.
    - Never split inside triple-backtick code blocks (` ```...``` `) or Markdown tables.
    - Detect primary chunk language (`"fa"` vs `"en"`) based on character script ratio.
    - Calculate token count using `tiktoken`.
- **Verification:** `pytest tests/unit/test_chunker.py -v`

---

## Phase 4: Dual Indexing (BGE-M3 Dense + BM25 Sparse)

### [ ] Task 4.1: Local BGE-M3 Embedding Wrapper
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
  - Format chunks inside `<retrieved_context><doc id="1" source="...">...</doc></retrieved_context>`.
  - Enforce system instructions to ignore any imperative commands inside `<retrieved_context>` and answer in the user's query language (Persian or English) with inline `[Source: file#section]` citations.
- **Verification:** `pytest tests/unit/test_prompt_builder.py -v`

### [ ] Task 6.2: Async vLLM Inference Client (Streaming & Structured)
- **Objective:** Implement the OpenAI-compatible async client talking to local vLLM with exponential backoff and timeout handling.
- **Reference Docs:** `docs/ARCHITECTURE.md` (ADR-001 & Section 9), `docs/PRD.md` (Feature 4).
- **Target Files:**
  - `src/services/llm_service.py`
  - `tests/unit/test_llm_service.py`
- **Implementation Details:**
  - Implement `LLMService` using `AsyncOpenAI(base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY)`.
  - Keep system prompt prefix identical across calls to maximize vLLM Prefix Caching hits.
  - Raise `InferenceEngineUnavailableError` if vLLM connection fails after retries.
- **Verification:** `pytest tests/unit/test_llm_service.py -v`

### [ ] Task 6.3: Redis Semantic Cache & Rate Limiter Service
- **Objective:** Implement Redis query response caching (24h TTL) and sliding-window IP rate limiting (30 req/min).
- **Reference Docs:** `docs/DATABASE.md` (Section 3.2: Redis Key Schema), `docs/SECURITY.md` (Section 5).
- **Target Files:**
  - `src/services/cache_service.py`
  - `tests/unit/test_cache_service.py`
- **Implementation Details:**
  - Compute SHA-256 of normalized query for key `cache:query:<sha256_hash>`.
  - Implement `check_rate_limit(client_ip)` using `ratelimit:ip:<client_ip>`.
  - Fail open (log warning and continue) if Redis container is temporarily unreachable.
- **Verification:** `pytest tests/unit/test_cache_service.py -v`

---

## Phase 7: LangGraph Corrective RAG (CRAG) Workflow

### [ ] Task 7.1: LangGraph State Definition & CRAG Grader Node
- **Objective:** Define the typed state dictionary and the Corrective RAG evaluator logic.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.3: LangGraph Corrective Control Plane), `docs/PRD.md` (Feature 3).
- **Target Files:**
  - `src/graph/state.py`
  - `src/graph/grader.py`
  - `tests/unit/test_crag_grader.py`
- **Implementation Details:**
  - Define `RAGGraphState(TypedDict)` tracking `original_query`, `active_query`, `retrieved_chunks`, `reflection_status` (`CORRECT`, `AMBIGUOUS`, `INCORRECT`), `rewrite_count`, `answer`, and `citations`.
  - Evaluate top reranker score:
    - $\ge 0.70 \rightarrow$ `CORRECT`
    - $0.35 \le \text{score} < 0.70$ and `rewrite_count < 1` $\rightarrow$ `AMBIGUOUS`
    - $< 0.35$ (or second attempt below threshold) $\rightarrow$ `INCORRECT`
- **Verification:** `pytest tests/unit/test_crag_grader.py -v`

### [ ] Task 7.2: Query Rewriter & Refusal Fallback Nodes
- **Objective:** Implement the query rewriting node for ambiguous queries and the zero-hallucination refusal node for out-of-domain queries.
- **Reference Docs:** `docs/UX_FLOWS.md` (Flow 2 & Flow 3), `docs/ARCHITECTURE.md` (Section 4.3).
- **Target Files:**
  - `src/graph/nodes.py`
  - `tests/unit/test_graph_nodes.py`
- **Implementation Details:**
  - `rewrite_query_node`: Reformulates vague queries (e.g., "مشکل ارور ۵۰۲") into technical search keywords and increments `rewrite_count`.
  - `refuse_node`: Returns the deterministic bilingual refusal message without calling LLM generation when context is `INCORRECT`.
- **Verification:** `pytest tests/unit/test_graph_nodes.py -v`

### [ ] Task 7.3: Assemble & Compile LangGraph Workflow with Langfuse Tracing
- **Objective:** Wire all nodes (`retrieve_and_rerank`, `grade_context`, `rewrite_query`, `generate`, `refuse`) into a compiled `StateGraph`.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 4.3, 4.4 & Section 5 Diagram).
- **Target Files:**
  - `src/graph/workflow.py`
  - `tests/unit/test_workflow.py`
- **Implementation Details:**
  - Add conditional edges from `grade_context` to `generate`, `rewrite_query`, or `refuse`.
  - Add edge from `rewrite_query` back to `retrieve_and_rerank` (capped at 1 loop).
  - Integrate Langfuse trace spans around node executions.
- **Verification:** `pytest tests/unit/test_workflow.py -v`

---

## Phase 8: FastAPI Backend, SSE Streaming & Security Middleware

### [ ] Task 8.1: API Request/Response Schemas & RFC 7807 Exception Handler
- **Objective:** Implement Pydantic API models and RFC 7807 Problem Details error middleware.
- **Reference Docs:** `docs/API.md` (Section 2 & Section 3: Error Handling Contract), `docs/SECURITY.md` (Section 2).
- **Target Files:**
  - `src/api/schemas.py`
  - `src/api/dependencies.py`
  - `tests/unit/test_api_schemas.py`
- **Implementation Details:**
  - Create `QueryRequest`, `QueryResponse`, `IngestResponse`, `EvalRunRequest`, `EvalRunResponse`, and `ProblemDetail`.
  - Implement Bearer API Key validation dependency and rate-limit check dependency.
- **Verification:** `pytest tests/unit/test_api_schemas.py -v`

### [ ] Task 8.2: Synchronous & SSE Streaming Query Endpoints
- **Objective:** Implement `POST /api/v1/query` and `POST /api/v1/query/stream` in FastAPI.
- **Reference Docs:** `docs/API.md` (Section 2.1 & 2.2), `docs/UX_FLOWS.md` (Flow 1).
- **Target Files:**
  - `src/api/routes.py`
  - `main.py`
  - `tests/integration/test_api_routes.py`
- **Implementation Details:**
  - Emit structured SSE events in `/api/v1/query/stream`: `step_start`, `retrieval_complete`, `token`, and `done`.
  - Wire application lifespan in `main.py` to initialize shared Qdrant, Redis, and vLLM clients once at startup.
- **Verification:** `pytest tests/integration/test_api_routes.py -v`

### [ ] Task 8.3: Ingestion & Background Job Status Endpoints
- **Objective:** Implement `POST /api/v1/ingest` and `GET /api/v1/ingest/status/{job_id}` with file upload security validation.
- **Reference Docs:** `docs/API.md` (Section 2.3 & 2.4), `docs/SECURITY.md` (Section 6: File Upload & Ingestion Security).
- **Target Files:**
  - `src/api/routes.py`
  - `tests/integration/test_ingest_routes.py`
- **Implementation Details:**
  - Enforce allowed extensions (`.md`, `.markdown`, `.txt`), 25MB size cap, and strip path traversal sequences (`../`).
  - Run indexing in FastAPI `BackgroundTasks` and store job status in Redis.
- **Verification:** `pytest tests/integration/test_ingest_routes.py -v`

---

## Phase 9: Chainlit Bilingual UI & Step Visualization

### [ ] Task 9.1: Design System CSS & Bi-Directional (RTL/LTR) Styling
- **Objective:** Configure Chainlit theme, Vazirmatn + JetBrains Mono fonts, and automatic RTL/LTR styling.
- **Reference Docs:** `docs/DESIGN_SYSTEM.md` (Section 3, 4, 10, 13).
- **Target Files:**
  - `.chainlit/config.toml`
  - `public/custom.css`
- **Implementation Details:**
  - Configure dark/light palette (`#090D16` background, `#10B981` primary Emerald, `#131B2E` surface).
  - Load `Vazirmatn` for Persian text (`dir="auto"`) and force `dir="ltr"` on `<pre><code>` blocks.
- **Verification:** Verify `.chainlit/config.toml` syntax and `public/custom.css` presence.

### [ ] Task 9.2: Interactive Chainlit Application (`app.py`)
- **Objective:** Build the Chainlit chat interface with collapsible reasoning steps and interactive source citations.
- **Reference Docs:** `docs/UX_FLOWS.md` (Flows 1, 2, 3), `docs/DESIGN_SYSTEM.md` (Section 10 & 11).
- **Target Files:**
  - `app.py`
- **Implementation Details:**
  - Render starter prompts in Persian and English on empty state.
  - Display collapsible `cl.Step` accordions for:
    1. *Hybrid Retrieval (Qdrant + BM25)*
    2. *Reciprocal Rank Fusion & Cross-Encoder Reranking (with score pills)*
    3. *CRAG Reflection Gate (showing `CORRECT`, `AMBIGUOUS` rewrite badge, or `INCORRECT` refusal)*
  - Stream tokens smoothly and attach expandable source snippet elements at completion.
- **Verification:** `ruff check app.py && mypy app.py`

---

## Phase 10: Quantitative Evaluation Pipeline (Ragas + DeepEval) & Documentation

### [ ] Task 10.1: Curated Bilingual Golden Evaluation Dataset
- **Objective:** Create `data/golden_eval_dataset.json` with 50 bilingual technical QA pairs and reference contexts.
- **Reference Docs:** `docs/TESTING.md` (Section 4.1: Evaluation Dataset), `docs/PRD.md` (Feature 5).
- **Target Files:**
  - `data/golden_eval_dataset.json`
  - `evals/generate_synthetic_data.py`
- **Implementation Details:**
  - Include exact error code lookups, Docker/PostgreSQL flags, Persian troubleshooting questions, and out-of-domain refusal questions.
- **Verification:** `python -c "import json; data = json.load(open('data/golden_eval_dataset.json')); assert len(data) >= 30"`

### [ ] Task 10.2: Multi-Variant Benchmark Runner (`Naive` vs `Hybrid` vs `Hybrid+Rerank+CRAG`)
- **Objective:** Implement the automated Ragas evaluation runner comparing all 3 pipeline variants using local vLLM.
- **Reference Docs:** `docs/TESTING.md` (Section 4.2 & 4.3), `docs/PRD.md` (Section 9: Success Metrics).
- **Target Files:**
  - `evals/run_benchmark.py`
  - `evals/evaluate_metrics.py`
  - `tests/unit/test_eval_runner.py`
- **Implementation Details:**
  - Compute `context_recall`, `context_precision`, `faithfulness`, `answer_relevancy`, and `p95_latency_ms`.
  - Export JSON and Markdown comparison tables to `evals/results/latest_benchmark.md`.
  - Exit with status code `1` if `faithfulness < 0.85` or `context_recall < 0.80`.
- **Verification:** `pytest tests/unit/test_eval_runner.py -v`

### [ ] Task 10.3: GitHub Actions CI & Regression Gate Workflows
- **Objective:** Configure automated CI workflows for unit testing, linting, and evaluation gating.
- **Reference Docs:** `docs/ARCHITECTURE.md` (Section 6), `docs/TESTING.md` (Section 4.3 & 5).
- **Target Files:**
  - `.github/workflows/test.yml`
  - `.github/workflows/eval-benchmark.yml`
  - `Dockerfile`
- **Implementation Details:**
  - Configure non-root user in `Dockerfile` per `docs/SECURITY.md`.
  - Run `ruff`, `mypy`, and `pytest` on every push and PR in `test.yml`.
- **Verification:** `pytest tests/unit -v && ruff check src tests evals`

### [ ] Task 10.4: Portfolio-Grade `README.md` with Benchmarks & Failure Analysis
- **Objective:** Write the final engineering `README.md` highlighting architecture, quantitative benchmark tables, and honest edge-case failure modes.
- **Reference Docs:** `docs/PRD.md` (Section 12: DoD), `docs/ARCHITECTURE.md` (Section 5 & 8).
- **Target Files:**
  - `README.md`
- **Implementation Details:**
  - Include system architecture diagram, ADR summary (why vLLM over Ollama, why BGE-M3, why CRAG), quantitative Ragas comparison table (`Naive Dense` vs `Hybrid` vs `Hybrid + Reranker + CRAG`), one-command Docker setup instructions, and a dedicated **Limitations & Edge Cases** section.
- **Verification:** Verify all links in `README.md` point to valid files in `docs/`.