# System Architecture (ARCHITECTURE.md)

## 1. System Overview
OmniDoc RAG is an enterprise-grade, bilingual technical documentation assistant. It pairs a high-precision hybrid retrieval engine (Dense BGE-M3 + Sparse BM25 + Reciprocal Rank Fusion + Cross-Encoder Reranking) with a LangGraph-orchestrated Corrective RAG (CRAG) loop and high-throughput local LLM inference served via vLLM.

---

## 2. Architecture Pattern
- **Pattern:** Modular Monolith (Clean Architecture) with Decoupled High-Performance Serving Microservices.
- **Why this pattern?**
  - Keeps core business logic (retrieval, fusion, reranking, reflection) strictly separated from presentation and framework adapters.
  - Decoupling heavy GPU model serving (vLLM) from application code prevents slow reloading cycles during development and allows independent vertical scaling.

---

## 3. Technology Stack
- **Frontend & Interaction:** Chainlit (Async SSE streaming, collapsible step UI, bi-directional text support).
- **Application Backend:** FastAPI (Python 3.11+, Pydantic V2, Async I/O, OpenAPI 3.1).
- **State Machine & Control Flow:** LangGraph (StateGraph for deterministic routing, reflection, and retry limits).
- **Vector Database:** Qdrant (Rust-based, Dockerized, HNSW indexing, filtered payloads).
- **Lexical Search:** BM25 (`rank_bm25` / persistent serialized index).
- **Embedding Model:** `BAAI/bge-m3` (1024-dim dense vectors + sparse lexical weights, multilingual).
- **Cross-Encoder Reranker:** `BAAI/bge-reranker-v2-m3` (Hugging Face `sentence-transformers`).
- **Inference Engine:** vLLM (OpenAI-compatible server, PagedAttention, Continuous Batching, AWQ quantization).
- **Caching & Rate Limiting:** Redis 7 (In-memory cache, semantic query caching, sliding-window rate limit).
- **Observability & Tracing:** Langfuse (Self-hosted via Docker, OpenTelemetry conventions).
- **Evaluation & Testing:** Ragas (RAG triaging metrics), DeepEval, Pytest.

---

## 4. System Components

### 4.1 Ingestion & Parsing Subsystem
- **Markdown & Code Chunker:** Splits technical documents by header boundaries (`#`, `##`, `###`) and logical code blocks, maintaining parent header context in chunk metadata.
- **Embedding Generator:** Produces dense embeddings via BGE-M3 and extracts sparse token weights.
- **Index Synchronizer:** Upserts dense vectors and payloads to Qdrant, updates the BM25 inverse document frequency table, and invalidates affected Redis cache entries.

### 4.2 Hybrid Retrieval Subsystem
- **Dense Retriever:** Queries Qdrant HNSW vector index using Cosine similarity (Top-20).
- **Sparse Retriever:** Computes BM25 lexical scores over tokenized corpus (Top-20).
- **Reciprocal Rank Fusion (RRF):** Merges both ranked lists using the formula:
  $$	ext{RRF Score}(d) = \sum_{m \in \{	ext{dense}, 	ext{sparse}\}} rac{1}{60 + r_m(d)}$$
  Yields Top-15 balanced candidates.
- **Cross-Encoder Reranker:** Runs joint sequence scoring (`[CLS] Query [SEP] Chunk [SEP]`) to produce final calibrated relevance scores (Top-5).

### 4.3 LangGraph Corrective Control Plane (CRAG)
- **Node: Retrieve & Rerank:** Gathers top candidate documents.
- **Node: Evaluate Context (Grader):** Inspects max reranker score and content overlap:
  - If score $\ge 0.70$ $ightarrow$ Route to `generate` (Status: CORRECT).
  - If $0.35 \le 	ext{score} < 0.70$ and rewrite count $< 1$ $ightarrow$ Route to `rewrite_query` (Status: AMBIGUOUS).
  - If score $< 0.35$ $ightarrow$ Route to `refuse` (Status: INCORRECT / EMPTY).
- **Node: Rewrite Query:** Refines ambiguous technical queries into explicit search terms.
- **Node: Grounded Generation:** Feeds context into Qwen 2.5 with a strict zero-hallucination prompt.

### 4.4 Caching & Observability Subsystem
- **Redis Semantic Cache:** Computes SHA-256 hash of normalized queries; caches verified answers for 24 hours.
- **Langfuse Tracer:** Intercepts every retrieval step, reranker score, prompt latency, token count, and user feedback.

---

## 5. Architecture Diagram

```
                             ┌───────────────────────┐
                             │   Chainlit Web UI     │
                             │ (User Chat & Steps)   │
                             └───────────▲───────────┘
                                         │ SSE Streaming
                             ┌───────────▼───────────┐
                             │  FastAPI Backend API  │
                             └───────────▲───────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │          LangGraph Orchestration State        │
                 │                                               │
                 │  ┌─────────────────────────────────────────┐  │
                 │  │ 1. Redis Cache Check (Fast Return)       │  │
                 │  └────────────────────┬────────────────────┘  │
                 │                       ▼                       │
                 │  ┌─────────────────────────────────────────┐  │
                 │  │ 2. Dual Parallel Retrieval              │  │
                 │  │    ├── Qdrant (Dense Top-20)            │  │
                 │  │    └── BM25 Index (Sparse Top-20)       │  │
                 │  └────────────────────┬────────────────────┘  │
                 │                       ▼                       │
                 │  ┌─────────────────────────────────────────┐  │
                 │  │ 3. Reciprocal Rank Fusion (Top-15)      │  │
                 │  └────────────────────┬────────────────────┘  │
                 │                       ▼                       │
                 │  ┌─────────────────────────────────────────┐  │
                 │  │ 4. Cross-Encoder Reranker (Top-5)       │  │
                 │  └────────────────────┬────────────────────┘  │
                 │                       ▼                       │
                 │  ┌─────────────────────────────────────────┐  │
                 │  │ 5. CRAG Evaluator (Relevance Gate)      │  │
                 │  └────────┬───────────────────────┬────────┘  │
                 │           │                       │           │
                 │      Pass │                  Fail │           │
                 │           ▼                       ▼           │
                 │  ┌─────────────────┐    ┌──────────────────┐  │
                 │  │ 6. LLM Stream   │    │ 6b. Query Rewrite│  │
                 │  │    via vLLM     │    │     or Refusal   │  │
                 │  └─────────────────┘    └──────────────────┘  │
                 └───────────────────────┬───────────────────────┘
                                         │
                             ┌───────────▼───────────┐
                             │  Langfuse Monitoring  │
                             │  (Traces & Latency)   │
                             └───────────────────────┘
```

---

## 6. Project Structure

```text
omnidoc-rag/
├── .agents/
│   └── skills/                      # Local Curated Skills Suite (29 skills)
├── .github/
│   └── workflows/
│       ├── test.yml                 # Pytest, ruff, mypy CI
│       └── eval-benchmark.yml       # Ragas regression gating
├── configs/
│   ├── config.yaml                  # Model names, thresholds, ports, top_k
│   └── prompts.yaml                 # System prompts, grading prompts, rewrite prompts
├── data/
│   ├── raw/                         # Markdown technical docs
│   ├── processed/                   # Chunks cache & BM25 serialized index
│   └── golden_eval_dataset.json     # 50 verified test cases
├── docs/                            # Blueprint Documentation Suite
│   ├── PRD.md
│   ├── DESIGN_SYSTEM.md
│   ├── UX_FLOWS.md
│   ├── ARCHITECTURE.md
│   ├── DATABASE.md
│   ├── API.md
│   ├── SECURITY.md
│   ├── CODE_STYLE.md
│   ├── TESTING.md
│   └── Plans.md
├── evals/
│   ├── __init__.py
│   ├── generate_synthetic_data.py   # Ragas synthetic test set generator
│   ├── run_benchmark.py             # Evaluation runner
│   └── evaluate_metrics.py          # Metric calculations & table exporter
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── config.py                # Pydantic Settings
│   │   ├── exceptions.py            # Custom application exceptions
│   │   └── logger.py                # Structured logging configuration
│   ├── ingestion/
│   │   ├── chunker.py               # Section-aware markdown splitting
│   │   ├── embedder.py              # BGE-M3 local inference
│   │   └── indexer.py               # Qdrant upsert & BM25 builder
│   ├── retrieval/
│   │   ├── dense.py                 # Qdrant vector search
│   │   ├── sparse.py                # BM25 search
│   │   ├── fusion.py                # RRF algorithm
│   │   └── reranker.py              # BGE Cross-Encoder
│   ├── graph/
│   │   ├── state.py                 # LangGraph AgentState TypedDict
│   │   ├── nodes.py                 # Retrieve, Grade, Rewrite, Generate
│   │   └── workflow.py              # Graph compilation & entrypoint
│   ├── services/
│   │   ├── cache_service.py         # Redis semantic & token caching
│   │   └── llm_service.py           # vLLM async streaming client
│   └── api/
│       ├── routes.py                # FastAPI endpoints
│       └── schemas.py               # Request/Response Pydantic models
├── app.py                           # Chainlit frontend application
├── main.py                          # FastAPI application entrypoint
├── docker-compose.yml               # Qdrant, Redis, Postgres, Langfuse
├── Dockerfile                       # App container definition
├── Makefile                         # dev, test, eval, up, down targets
├── pyproject.toml                   # Ruff, mypy, pytest configuration
├── requirements.txt                 # Frozen production dependencies
├── AGENTS.md                        # Master AI Agent Guidelines & Skills Routing
├── CLAUDE.md                        # Claude harness instructions
├── GEMINI.md                        # Gemini harness instructions
└── README.md                        # Documentation, benchmarks, failure analysis
```

---

## 7. Dependency Rules
- **Layer 1 (Common / Entities):** No dependencies on external frameworks.
- **Layer 2 (Core Logic - Ingestion & Retrieval):** Depends only on Common. Does not depend on FastAPI or Chainlit.
- **Layer 3 (Workflow - LangGraph):** Depends on Core Logic. Manages application state.
- **Layer 4 (Delivery - API / UI):** FastAPI and Chainlit depend on Workflow and Services. Core business logic never imports UI or Web frameworks.

---

## 8. Key Technical Decisions (Architecture Decision Records)

### ADR-001: vLLM over Ollama
- **Decision:** Use standalone vLLM server with PagedAttention and AWQ quantization instead of Ollama.
- **Rationale:** Ollama serializes requests in a single-threaded FIFO queue. Evaluating 50 queries in Ragas requires hundreds of concurrent calls. vLLM continuous batching drops evaluation duration from 45 minutes to under 3 minutes.

### ADR-002: BGE-M3 for Dual Dense and Sparse Representation
- **Decision:** Adopt `BAAI/bge-m3` as the primary embedding model.
- **Rationale:** It supports 100+ languages (with native Persian and English tokenization), an 8192 context window, and out-of-the-box dense + sparse token weights.

### ADR-003: Qdrant as Dedicated Vector Database
- **Decision:** Use Qdrant over pgvector.
- **Rationale:** Written in Rust, lower memory footprint under high concurrency, fast payload filtering, and native support for sparse/dense multi-vector queries.

---

## 9. Failure Scenarios & Self-Healing
- **vLLM Inference Failure / OOM:** FastAPI falls back to a graceful 503 error message with exponential backoff retry.
- **Qdrant DB Timeout:** If vector search fails, the system executes sparse-only (BM25) retrieval as a degraded mode rather than returning a 500 error.
- **Redis Cache Unavailable:** System logs a warning and bypasses caching without halting query execution.
