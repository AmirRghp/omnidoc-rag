# Testing Strategy & Benchmark Verification (TESTING.md)

## 1. Testing Philosophy
In AI engineering, code syntax testing is only half the battle. A system can have 100% unit test coverage and still fail catastrophically in production due to retrieval degradation or hallucination.

OmniDoc RAG employs a **3-Tier Testing Pyramid**:
1. **Deterministic Unit & Component Tests** (Pytest): Fast execution (<5 seconds), testing chunking math, RRF fusion logic, Pydantic schemas, and error handlers.
2. **Integration Tests** (Testcontainers / Local Services): Testing FastAPI endpoints, Redis cache invalidation, and Qdrant payload filters.
3. **Quantitative LLM Evaluation Benchmarks** (Ragas + DeepEval): Automated measurement of `Context Recall`, `Context Precision`, `Faithfulness`, and `Answer Relevancy` against a curated Golden Dataset.

---

## 2. Test File Structure
```text
tests/
├── conftest.py
├── unit/
│   ├── test_app.py               # Chainlit bilingual UI, RTL & XSS sanitization
│   ├── test_bm25_store.py        # BM25s persistence & SHA-256 integrity verification
│   ├── test_cache.py             # Redis query cache & fallback behavior
│   ├── test_chunker.py           # Header-aware splitting & code block boundaries
│   ├── test_config.py            # Settings & env override precedence
│   ├── test_crag_workflow.py     # LangGraph CRAG nodes, reflection & streaming
│   ├── test_dense_retriever.py   # Qdrant vector search & degraded mode
│   ├── test_embedder.py          # BGE-M3 dense embedding wrapper
│   ├── test_eval_runner.py       # Ragas benchmark evaluation & gating logic
│   ├── test_exceptions.py        # Domain error hierarchy & RFC 7807 mapping
│   ├── test_fusion.py            # RRF reciprocal rank fusion correctness
│   ├── test_indexer.py           # End-to-end ingestion pipeline
│   ├── test_llm_client.py        # Async vLLM client & exponential backoff
│   ├── test_logger.py            # JSON structured logging & trace_id contextvar
│   ├── test_middleware.py        # TraceID cleanup & sliding-window rate limiter
│   ├── test_prompt_builder.py    # Bilingual prompt building & language detection
│   ├── test_qdrant_store.py      # Qdrant collection & payload index creation
│   ├── test_reranker.py          # Cross-Encoder reranking via asyncio.to_thread
│   ├── test_schemas.py           # Pydantic V2 validation & serialization
│   ├── test_security.py          # NFC normalization, prompt injection & path safety
│   └── test_sparse_retriever.py  # BM25s sparse retrieval
└── integration/
    └── test_api.py               # FastAPI TestClient endpoints, RBAC & SSE streams
```

---

## 3. Unit Testing Key Scenarios

### 3.1 Reciprocal Rank Fusion Verification
```python
# tests/unit/test_fusion.py
from src.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_scoring_order():
    list_a = [{"doc_id": "doc_1", "content": "a"}, {"doc_id": "doc_2", "content": "b"}]
    list_b = [{"doc_id": "doc_2", "content": "b"}, {"doc_id": "doc_3", "content": "c"}]

    # doc_2 is present in both lists and must rank #1
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert fused[0]["doc_id"] == "doc_2"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]
```

### 3.2 Chunker Structural Integrity
- Verify that markdown tables (`| col1 | col2 |`) are never broken mid-table.
- Verify that triple-backtick code blocks (` ```python ... ``` `) remain intact inside a single chunk.

---

## 4. Quantitative RAG Evaluation (Ragas Pipeline)

### 4.1 Evaluation Dataset (`evals/datasets/golden_bilingual_eval.json`)
The benchmark runs against manually curated bilingual technical QA pairs covering:
- Exact error codes (e.g., `dial unix /var/run/docker.sock: connect: permission denied`).
- Configuration directives (e.g., `max_connections`, `shared_buffers`).
- Out-of-domain questions to verify refusal behavior.

### 4.2 Gating Thresholds
| Metric | Minimum Acceptable (CI Gate) | Target Benchmark |
| :--- | :--- | :--- |
| **Context Recall** | $\ge 0.80$ | $\mathbf{0.91}$ |
| **Context Precision** | $\ge 0.75$ | $\mathbf{0.86}$ |
| **Faithfulness** | $\ge 0.85$ | $\mathbf{0.94}$ |
| **Answer Relevancy** | $\ge 0.80$ | $\mathbf{0.89}$ |

### 4.3 Automated Regression Rule
If a PR or prompt modification causes `Faithfulness` to regress by more than 3% compared to `main`, the CI workflow exits with code `1` and blocks merging.

---

## 5. Execution Commands

```bash
# Run unit tests only (fast)
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v

# Run full test suite with coverage report
pytest --cov=src --cov-report=term-missing tests/

# Execute Ragas evaluation benchmark
python -m evals.eval_runner --dataset evals/datasets/golden_bilingual_eval.json --output evals/results/latest.json

# Run linting and type checking
ruff check .
ruff format --check .
mypy .
```
