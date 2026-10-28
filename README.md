# OmniDoc RAG — Robust Bilingual Technical RAG & Evaluation System

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0%2B-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-v1.11.0-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/Redis-7--alpine-red.svg?logo=redis&logoColor=white)](https://redis.io/)
[![vLLM](https://img.shields.io/badge/vLLM-OpenAI--Compatible-7C3AED.svg)](https://github.com/vllm-project/vllm)
[![BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3-10B981.svg)](https://huggingface.co/BAAI/bge-m3)
[![Ragas](https://img.shields.io/badge/Evaluation-Ragas%20Gating-purple.svg)](https://docs.ragas.io/)
[![CI Quality Gate](https://img.shields.io/badge/CI-Passing%20(3%20Jobs)-brightgreen.svg)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An enterprise-grade, bilingual (**Persian / English**) technical retrieval-augmented generation engine engineered for high-precision DevOps and infrastructure documentation (Docker, PostgreSQL, Linux). Pairs a **Dual-Channel Hybrid Retrieval** pipeline (Dense BGE-M3 + Sparse BM25 + Reciprocal Rank Fusion + Cross-Encoder Reranking) with a **LangGraph Corrective RAG (CRAG)** state machine, local **vLLM** inference, and an automated **Ragas continuous evaluation gate**.

---

## Table of Contents
- [1. The "Demo vs. Production" Gap](#1-the-demo-vs-production-gap)
- [2. System Architecture](#2-system-architecture)
- [3. Quantitative Benchmark Results](#3-quantitative-benchmark-results)
- [4. Architectural Decision Records (ADRs)](#4-architectural-decision-records-adrs)
- [5. Bilingual Persian/English UX & Security Engineering](#5-bilingual-persianenglish-ux--security-engineering)
- [6. One-Command Quickstart & Deployment](#6-one-command-quickstart--deployment)
- [7. Honest Limitations & Edge Cases (Failure Analysis)](#7-honest-limitations--edge-cases-failure-analysis)
- [8. Project Structure & Documentation Index](#8-project-structure--documentation-index)
- [9. Testing Pyramid & Quality Gates](#9-testing-pyramid--quality-gates)

---

## 1. The "Demo vs. Production" Gap

Most RAG tutorials fall apart in production when applied to technical infrastructure documentation:

| Production Failure Mode | Naive Vector RAG Behavior | OmniDoc RAG Defensive Solution |
| :--- | :--- | :--- |
| **Exact Error Codes** (e.g. `dial unix /var/run/docker.sock: connect: permission denied`) | Semantic search loses token specificity in high-dimensional embedding space, retrieving generic networking docs. | **Hybrid Dual-Channel Search**: BM25 lexical retriever preserves exact socket paths, error strings, and CLI flags, fused via Reciprocal Rank Fusion (RRF). |
| **Hallucination on Missing Context** | The generator creates plausible-sounding but fictional configuration directives or invalid flags. | **Corrective RAG (CRAG) Gate**: Cross-Encoder score thresholding evaluates context relevance ($\ge 0.70$ pass; $0.35 - 0.70$ rewrite; $< 0.35$ polite refusal). Zero-hallucination refusal rules enclosed in `<retrieved_context>` XML wrappers. |
| **Mixed Persian/English Queries** (e.g. `تنظیمات shared_buffers در postgres چیست؟`) | Monolingual tokenizers split English flags mid-word or fail on Persian Arabic-variant characters (`ي`, `ك`). | **Bilingual Sanitizer & BGE-M3**: Unicode NFC normalization, Arabic-to-Persian char conversion, ZWNJ preservation, and native 100+ language subword tokenization. |
| **Evaluation Blindness** | Teams modify prompts or chunk sizes without measuring regression impact. | **Automated Ragas CI Gate**: Automated evaluation suite asserting Faithfulness ($\ge 0.85$), Answer Relevance ($\ge 0.85$), Context Precision ($\ge 0.80$), and Context Recall ($\ge 0.75$). |
| **Cloud API Costs & Privacy** | Every query and chunk is leaked to third-party proprietary LLMs at steep per-token pricing. | **100% Local Inference**: Served via local vLLM OpenAI-compatible endpoint with AWQ quantization at **\$0.00 API cost**. |

---

## 2. System Architecture

OmniDoc RAG is built as a **Modular Monolith** adhering to Clean Architecture principles:

```
                               ┌────────────────────────────────────────────────────────┐
                               │               Chainlit Web Interface                   │
                               │  - Bi-directional RTL / LTR layout (Vazirmatn Font)     │
                               │  - Collapsible Reasoning Steps (Retrieval, CRAG, Gen)  │
                               │  - Interactive Source Citations Drawer                 │
                               └───────────────────────────▲────────────────────────────┘
                                                           │ SSE Streaming / WebSockets
                               ┌───────────────────────────▼────────────────────────────┐
                               │                  FastAPI Backend                       │
                               │  - RFC 7807 Problem Details Error Handlers             │
                               │  - Token-by-Token SSE Stream (/api/v1/query/stream)    │
                               │  - Sliding-Window Rate Limiting & Auth Middleware      │
                               └───────────────────────────▲────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴────────────────────────────┐
                               │           LangGraph CRAG Control Plane                 │
                               │                                                        │
                               │   ┌────────────────────────────────────────────────┐   │
                               │   │ 1. Redis Cache Query (Exact SHA-256 + Semantic) │   │
                               │   └───────────────────────┬────────────────────────┘   │
                               │                           ▼                            │
                               │   ┌────────────────────────────────────────────────┐   │
                               │   │ 2. Dual-Channel Parallel Retrieval             │   │
                               │   │    ├── Qdrant Vector DB (Dense BGE-M3 Top-20)  │   │
                               │   │    └── BM25 Okapi Index (Sparse Lexical Top-20)│   │
                               │   └───────────────────────┬────────────────────────┘   │
                               │                           ▼                            │
                               │   ┌────────────────────────────────────────────────┐   │
                               │   │ 3. Reciprocal Rank Fusion (RRF k=60, Top-15)   │   │
                               │   └───────────────────────┬────────────────────────┘   │
                               │                           ▼                            │
                               │   ┌────────────────────────────────────────────────┐   │
                               │   │ 4. Cross-Encoder Reranker (bge-reranker-v2-m3) │   │
                               │   │    └── Sigmoid Calibrated Relevance in [0, 1]  │   │
                               │   └───────────────────────┬────────────────────────┘   │
                               │                           ▼                            │
                               │   ┌────────────────────────────────────────────────┐   │
                               │   │ 5. CRAG Quality Gate (Relevance Status Check)  │   │
                               │   └──────┬─────────────────┬─────────────────┬─────┘   │
                               │          │                 │                 │         │
                   Max Score >= 0.70      │     0.35 <= Score < 0.70          │ Max Score < 0.35
                                          ▼                 ▼                 ▼
                                   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                                   │   Generate   │  │Rewrite Query │  │ Refusal Gate │
                                   │  (vLLM Qwen) │  │  (Limit: 2)  │  │(Zero Halluc.)│
                                   └──────────────┘  └──────────────┘  └──────────────┘
```

### Layer Boundaries
- **Layer 1 (Common / Schemas):** Strictly pure Pydantic V2 models (`DocumentChunkPayload`, `RetrievedChunk`, `CitationItem`, `EvaluationRecord`), custom exceptions, and Unicode sanitizers.
- **Layer 2 (Core Logic - Ingestion & Retrieval):** Independent semantic chunker, BGE-M3 embedding wrapper, Qdrant store manager, persistent BM25 indexer, and RRF fusion engine.
- **Layer 3 (Workflow - LangGraph CRAG):** Finite state machine orchestrating retrieve, grade, rewrite, generate, and refusal transitions.
- **Layer 4 (Delivery - API & UI):** FastAPI REST/SSE endpoints and Chainlit interactive web interface.

---

## 3. Quantitative Benchmark Results

Evaluated against the curated 30-sample bilingual technical dataset (`evals/datasets/golden_bilingual_eval.json`) covering Docker, PostgreSQL, and Linux troubleshooting scenarios. Benchmarks were computed using the automated Ragas metric runner (`evals/eval_runner.py`) with a local vLLM judge model (`Qwen/Qwen2.5-1.5B-Instruct-AWQ` / `Qwen2.5-7B-Instruct`). Stored execution artifact: [`evals/eval_report.json`](evals/eval_report.json).

### 3-Way Comparative Evaluation Matrix

| Evaluation Metric | Naive Dense Baseline (BGE-M3 only) | Hybrid Search (Dense + BM25 + RRF) | OmniDoc RAG (Verified Live Run) | Production CI Gate | Target Benchmark |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context Recall@5** | 0.68 | 0.84 | **0.959** (95.9%) | $\ge 0.75$ | $\ge 0.90$ |
| **Context Precision@5** | 0.62 | 0.76 | **0.953** (95.3%) | $\ge 0.80$ | $\ge 0.85$ |
| **Faithfulness (Zero-Hallucination)** | 0.74 | 0.82 | **0.985** (98.5%) | $\ge 0.85$ | $\ge 0.92$ |
| **Answer Relevancy** | 0.71 | 0.81 | **0.968** (96.8%) | $\ge 0.85$ | $\ge 0.88$ |
| **P95 Latency (End-to-End)** | 180 ms | 215 ms | **256 ms** | $\le 1000\text{ ms}$ | $\le 500\text{ ms}$ |
| **External API Inference Cost** | \$0.00 | \$0.00 | **\$0.00** | \$0.00 | \$0.00 |

### Key Takeaways
1. **Recall Jump (+28% vs Naive):** Combining BGE-M3 dense vectors with BM25 lexical search and Reciprocal Rank Fusion lifted recall to **0.959**, capturing exact CLI flags (`--ipc=host`, `--gpus`) and PostgreSQL error codes that pure vector search missed.
2. **Faithfulness Jump (+24.5% vs Naive):** The Cross-Encoder reranker and CRAG reflection state machine filtered out noisy or partially matching passages, achieving **0.985** Faithfulness with near-zero hallucinations.
3. **Refusal Accuracy (100%):** On out-of-domain queries, the CRAG relevance gate scored context $< 0.35$ and executed a polite refusal without generating ungrounded claims.
4. **Reproduce Locally:** Run the automated evaluation suite against the golden dataset:
   ```bash
   python -m evals.eval_runner --samples 30 --output evals/eval_report.json
   ```

---

## 4. Architectural Decision Records (ADRs)

Key architectural trade-offs documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md):

- **[ADR-001: vLLM over Ollama](docs/ARCHITECTURE.md#adr-001-vllm-over-ollama):**
  - *Decision:* Deploy vLLM with PagedAttention and AWQ quantization rather than Ollama.
  - *Rationale:* Ollama queues concurrent requests sequentially in a single-threaded FIFO queue. Evaluating benchmark queries or serving concurrent users under Ollama resulted in high latencies. vLLM continuous batching reduced 50-query evaluation runtime from ~45 minutes to under 3 minutes.
- **[ADR-002: BGE-M3 for Multilingual Dense & Sparse Representation](docs/ARCHITECTURE.md#adr-002-bge-m3-for-dual-dense-and-sparse-representation):**
  - *Decision:* Adopt `BAAI/bge-m3` as the primary embedding generator.
  - *Rationale:* Native multilingual support across 100+ languages, 8192-token context window, and out-of-the-box support for dense representations and sparse lexical token weights.
- **[ADR-003: Qdrant as Dedicated Vector Database](docs/ARCHITECTURE.md#adr-003-qdrant-as-dedicated-vector-database):**
  - *Decision:* Use Qdrant rather than pgvector or in-memory vector stores.
  - *Rationale:* Built in Rust, lower memory footprint under high concurrency, fast payload filtering on document metadata (`doc_id`, `language`), and native support for multi-vector hybrid search.
- **[ADR-004: Corrective RAG (CRAG) with LangGraph](docs/ARCHITECTURE.md#43-langgraph-corrective-control-plane-crag):**
  - *Decision:* Implement a deterministic state graph for retrieval evaluation, query rewriting, and refusal.
  - *Rationale:* Hard boundaries on query rewrite transformations ($N \le 2$) prevent infinite agentic recursion loops while guaranteeing zero-hallucination refusals when retrieved documentation is irrelevant.

---

## 5. Bilingual Persian/English UX & Security Engineering

Technical documentation environments regularly mix Latin commands, code blocks, and Persian explanations. OmniDoc RAG enforces bidirectional layout hygiene and defensive security filters:

### 1. Typography & Bidirectional Styling
- **Vazirmatn Font Integration:** Native typography for Persian prose with fallback to JetBrains Mono for code.
- **Automatic Container Directionality:** Containers use `dir="auto"` to automatically orient Persian paragraphs to Right-to-Left (RTL) and English paragraphs to Left-to-Right (LTR).
- **Code Block LTR Isolation:** Fenced code blocks (`<pre><code>`) and CLI commands are strictly forced to `dir="ltr"` and `text-align: left`, preventing reversed flags (`-p 8000:8000`), displaced brackets, or inverted IP addresses.

### 2. Input Sanitization & Security Normalizer ([`src/common/security.py`](src/common/security.py))
- **Unicode NFC Normalization:** Eliminates decomposed Unicode representation anomalies.
- **Arabic-to-Persian Mapping:** Automatically converts Arabic letter variants to Persian (`ي` $\rightarrow$ `ی`, `ك` $\rightarrow$ `ک`).
- **Zero-Width Non-Joiner (ZWNJ) Preservation:** Preserves legitimate Persian half-spaces (`می‌شود`, `دست‌انداز`) while stripping redundant control characters.
- **Directional Override Character Stripping:** Filters Unicode bidirectional control characters (`\u202A`–`\u202E`, `\u2066`–`\u2069`) to protect against Trojan Source attacks and visual spoofing.
- **Prompt Injection Defense:** User queries are never concatenated directly into prompt prose; retrieved chunks are wrapped within strict XML enclosures:
  ```xml
  <retrieved_context>
  ... verified documentation passages ...
  </retrieved_context>
  ```
  The LLM system prompt mandates: *"Answer only based on the verified context above. If the context is insufficient, state that you do not know."*

---

## 6. One-Command Quickstart & Deployment

### Prerequisites
- Docker & Docker Compose (v2.20+)
- Python 3.11+
- NVIDIA GPU with 12GB+ VRAM (optional, for local vLLM model serving; CPU mode supported for testing)

### Step 1: Clone and Configure Environment
```bash
git clone https://github.com/AmirRghp/omnidoc-rag.git
cd omnidoc-rag

# Copy environment template
cp .env.example .env
```

### Step 2: Start Infrastructure Microservices
Start Qdrant, Redis, PostgreSQL, and Langfuse in background containers:
```bash
make up
# Or: docker compose up -d
```
Service endpoints will become available:
- **Qdrant Vector DB:** `http://localhost:6333` (Dashboard: `http://localhost:6333/dashboard`)
- **Redis Cache:** `localhost:6379`
- **PostgreSQL:** `localhost:54321` (configurable via `POSTGRES_PORT`, mapped to internal `5432`)
- **Langfuse Observability:** `http://localhost:3000`

### Step 3: Install Python Virtual Environment
```bash
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install dependencies (with fast CPU-torch wheel for local dev)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 4: Ingest Documentation Corpus
Index seed documentation (Docker, PostgreSQL, Linux):
```bash
python scripts/fetch_docs.py

# Ingest into Qdrant & BM25 (using CPU to keep GPU memory free for your LLM):
# On Linux/macOS:
CUDA_VISIBLE_DEVICES="" python -m src.ingestion.indexer --path data/raw

# On Windows PowerShell:
$env:CUDA_VISIBLE_DEVICES=""; python -m src.ingestion.indexer --path data/raw
```

### Step 5: Start Your Local LLM (Choose Ollama or vLLM)
OmniDoc RAG uses an OpenAI-compatible interface, so you can connect either **Ollama** (easiest for laptops/low-VRAM) or **vLLM** (high-throughput batching for dedicated GPUs):

<details open>
<summary><b>Option A: Ollama (Recommended for laptops & GPUs with &lt;8GB VRAM)</b></summary>

1. **Pull and start model with Ollama:**
   ```bash
   ollama run qwen2.5:3b
   # Or for very light setups:
   # ollama run qwen2.5:1.5b
   ```
2. **Configure `.env`:**
   ```env
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_API_KEY=ollama
   LLM_MODEL=qwen2.5:3b
   ```
</details>

<details>
<summary><b>Option B: vLLM (Recommended for dedicated GPUs with PagedAttention batching)</b></summary>

> [!NOTE]
> **Windows Notice:** `vLLM` relies on Linux-only CUDA C++ kernels. On **native Windows**, running `pip install vllm` will fail with missing module errors. Windows users must run vLLM via the official **Docker container** (Method B1) or WSL2. On **Linux**, run via native Python (Method B2).

#### B1. Windows (Docker with NVIDIA GPU Passthrough)
Ensure Docker Desktop is running with WSL2 backend, then run in PowerShell:
```powershell
docker run --gpus all -p 8001:8000 --ipc=host `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen2.5-1.5B-Instruct-AWQ `
  --gpu-memory-utilization 0.6 `
  --max-model-len 2048
```

#### B2. Linux / WSL2 (Native Python)
1. **Install vLLM:**
   ```bash
   pip install vllm
   ```
2. **Launch the vLLM server on port 8001** (avoids colliding with FastAPI on port 8000):
   ```bash
   # For 4GB-8GB GPUs (AWQ quantized):
   python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
     --port 8001 \
     --gpu-memory-utilization 0.6 \
     --max-model-len 2048

   # For 12GB+ GPUs (Full 7B Instruct):
   # python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8001
   ```

#### B3. Configure `.env`:
```env
LLM_BASE_URL=http://localhost:8001/v1
LLM_API_KEY=dummy-local-key
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct-AWQ
```
</details>

### Step 6: Launch the Application
Run the FastAPI backend:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
# API Docs available at: http://localhost:8000/docs
```

Run the Chainlit Web Interface:
```bash
chainlit run app.py -w --port 8501
# UI available at: http://localhost:8501
```

### Developer Commands (`Makefile`)
| Command | Action |
| :--- | :--- |
| `make up` | Start Docker containers (Qdrant, Redis, Postgres, Langfuse) |
| `make down` | Stop containerized infrastructure |
| `make lint` | Run `ruff check .`, `ruff format --check .`, and `mypy src` |
| `make format` | Format code with `ruff format .` and `ruff check --fix .` |
| `make test` | Run full Pytest test suite (300+ unit and integration tests) |
| `make test-unit` | Run fast deterministic unit tests only (`pytest tests/unit -v`) |
| `make test-integration` | Run integration tests against service containers (`pytest tests/integration -v`) |
| `make eval` | Run Ragas metric evaluation benchmark against golden dataset |
| `make clean` | Remove temporary caches (`.pytest_cache`, `.mypy_cache`, `__pycache__`) |

---

## 7. Honest Limitations & Edge Cases (Failure Analysis)

In the interest of engineering transparency, the following edge cases and operational constraints have been observed and designed for:

1. **Self-Healing Degraded Mode (Qdrant Timeout):**
   - *Failure Scenario:* Qdrant container runs out of memory or latency exceeds 2000 ms.
   - *Mitigation:* The retrieval orchestrator catches `VectorStoreTimeoutError`, logs a warning, and falls back to **sparse-only (BM25)** retrieval rather than crashing the request lifecycle with a 500 error.
2. **Redis Outage Resilience:**
   - *Failure Scenario:* Redis service goes offline or reaches connection pool limits.
   - *Mitigation:* Cache reads and writes fail gracefully with an error log, allowing the CRAG pipeline to query Qdrant and vLLM directly without downtime.
3. **Complex Markdown Table Hierarchy:**
   - *Limitation:* Deeply nested Markdown tables spanning multiple subheaders can lose column context when chunk boundaries fall near table borders.
   - *Mitigation:* The semantic chunker detects table markers (`|---|`) and prevents splitting across chunk boundaries, but tables exceeding 512 tokens must be segmented with repeated header lines.
4. **Colloquial Persian Slang vs. Technical Precision:**
   - *Limitation:* Slang queries without Latin command anchors (e.g. `چرا کانتینرم خوابید؟`) may lack sufficient lexical overlap for BM25.
   - *Mitigation:* The CRAG evaluator routes ambiguous queries to the `rewrite_query` node, which converts colloquial phrases into explicit technical terms (`بررسی دلایل توقف کانتینر داکر و وضعیت Exit Code`).
5. **Local Hardware Requirements:**
   - *Limitation:* Running 7B/14B AWQ models with vLLM PagedAttention requires an NVIDIA GPU with at least 12GB VRAM.
   - *Mitigation:* Standard OpenAI-compatible API clients allow redirecting `LLM_BASE_URL` to remote inference servers or cloud endpoints if local GPU resources are unavailable.

---

## 8. Project Structure & Documentation Index

### Directory Tree
```text
omnidoc-rag/
├── .github/workflows/
│   └── ci.yml               # Automated 3-job CI pipeline (Lint, Unit, Integration)
├── configs/
│   ├── config.yaml          # Hyperparameters (top_k, rrf_k, CRAG thresholds)
│   └── prompts.yaml         # Guardrailed XML prompt templates
├── data/
│   ├── raw/                 # Seed documentation (Docker, Postgres, Linux)
│   └── processed/           # Persistent BM25 inverted index (.pkl)
├── docs/                    # Canonical Source of Truth documentation
│   ├── PRD.md               # Product requirements and acceptance criteria
│   ├── ARCHITECTURE.md      # Architecture, component flow, and ADRs
│   ├── DATABASE.md          # Qdrant collection schemas, payloads, Redis keys
│   ├── API.md               # OpenAPI contracts, SSE streaming specifications
│   ├── SECURITY.md          # Input sanitization, bidi defense, secrets
│   ├── CODE_STYLE.md        # Python typing, async rules, logging standards
│   ├── TESTING.md           # Testing pyramid, benchmarks, gating thresholds
│   ├── DESIGN_SYSTEM.md     # Chainlit palette, RTL typography, components
│   ├── UX_FLOWS.md          # User flows, reasoning drawer, feedback
│   └── Plans.md             # Master implementation tracking plan
├── evals/
│   ├── datasets/            # Curated 30-sample bilingual golden dataset
│   └── eval_runner.py       # Ragas evaluation runner and CI gating CLI
├── scripts/
│   └── fetch_docs.py        # Live documentation downloader and seed fixtures
├── src/
│   ├── common/              # Config, domain schemas, exceptions, security
│   ├── ingestion/           # Markdown chunker, BGE embedder, Qdrant/BM25 stores
│   ├── retrieval/           # Dense, sparse, RRF fusion, Cross-Encoder reranker
│   ├── graph/               # LangGraph CRAG state, nodes, and compiled workflow
│   ├── services/            # vLLM OpenAI-compatible client, 2-tier Redis cache
│   └── api/                 # FastAPI routes, middleware, RFC 7807 problem details
├── tests/
│   ├── unit/                # 290+ deterministic unit tests (math, schemas, graph)
│   └── integration/         # API streaming, live containers, Ragas gating
├── app.py                   # Bilingual Chainlit user interface
├── docker-compose.yml       # Qdrant, Redis, Postgres, Langfuse definitions
├── Makefile                 # Developer automation targets
├── pyproject.toml           # Ruff, MyPy, and Pytest configuration
├── requirements.txt         # Production dependencies
└── README.md                # System documentation and portfolio guide
```

### Documentation Cross-Reference
| Document | Focus Area |
| :--- | :--- |
| **[`docs/PRD.md`](docs/PRD.md)** | Product vision, user stories, success metrics, and MVP scope. |
| **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** | System layers, data flow diagrams, ADRs, and self-healing failure modes. |
| **[`docs/DATABASE.md`](docs/DATABASE.md)** | Qdrant vector collection payload schemas, Redis caching keys, and BM25 index. |
| **[`docs/API.md`](docs/API.md)** | FastAPI REST endpoints, SSE streaming contract, and RFC 7807 error models. |
| **[`docs/SECURITY.md`](docs/SECURITY.md)** | Unicode bidi defenses, indirect prompt injection guardrails, and secrets rules. |
| **[`docs/CODE_STYLE.md`](docs/CODE_STYLE.md)** | Strict typing rules (`mypy --strict`), async boundary rules, and logging. |
| **[`docs/TESTING.md`](docs/TESTING.md)** | Testing pyramid strategy, Ragas gating thresholds, and benchmark execution. |
| **[`docs/DESIGN_SYSTEM.md`](docs/DESIGN_SYSTEM.md)** | Chainlit visual palette, RTL layout guidelines, and typography. |
| **[`docs/UX_FLOWS.md`](docs/UX_FLOWS.md)** | User interaction flows, step collapsing, and citations drawer. |
| **[`docs/Plans.md`](docs/Plans.md)** | Master implementation tracker and completion protocol. |

---

## 9. Testing Pyramid & Quality Gates

OmniDoc RAG enforces quality via a 3-tier testing pyramid and automated CI checks:

```
                  ┌───────────────────────────────┐
                  │    Quantitative Ragas Evals   │  Continuous Evaluation Gate
                  │  (Faithfulness, Relevancy)    │  (tests/integration/test_ragas_gating.py)
                  └───────────────▲───────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │    Integration Test Suite     │  Service Containers
                  │(FastAPI SSE, Qdrant, Redis)   │  (tests/integration/)
                  └───────────────▲───────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │     Deterministic Unit Tests  │  Fast Mocked Execution (<10s)
                  │(Chunker, Fusion, CRAG, Schemas│  (tests/unit/)
                  └───────────────────────────────┘
```

### CI Pipeline (`.github/workflows/ci.yml`)
On every `push` and `pull_request` targeting `main`:
1. **`lint-and-typecheck`**: Runs `ruff check .`, `ruff format --check .`, and `mypy src`.
2. **`unit-tests`**: Runs all 290 unit tests with coverage (`pytest tests/unit -v --cov=src`), enforcing $\ge 85\%$ total coverage.
3. **`integration-tests`**: Runs integration suites against live containerized Qdrant and Redis services.

---

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
