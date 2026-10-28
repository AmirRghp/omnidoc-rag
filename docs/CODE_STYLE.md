# Code Style & Engineering Standards (CODE_STYLE.md)

## 1. General Principles
- **Readability Over Cleverness:** Write transparent, explicit code. The next developer or AI agent should immediately understand intent without reverse-engineering complex one-liners.
- **Fail Fast & Explicitly:** Do not swallow exceptions silently. Raise domain-specific exceptions.
- **Strict Typing:** All function signatures and module interfaces must have complete Python type annotations.
- **Pure Core Logic:** Keep business logic (chunking, scoring, ranking) free from side-effects and external I/O frameworks whenever possible.

---

## 2. Naming Conventions
- **Classes & Types:** `PascalCase` (e.g., `HybridRetriever`, `DocumentChunk`, `RerankResult`).
- **Functions & Methods:** `snake_case` (e.g., `calculate_rrf_score()`, `extract_sections()`).
- **Variables & Attributes:** `snake_case` (e.g., `candidate_chunks`, `relevance_threshold`).
- **Constants & Enums:** `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_EMBEDDING_DIM`).
- **Filenames:** `snake_case.py` (e.g., `chunker.py`, `fusion.py`).
- **Pydantic Models:** Suffix with purpose (e.g., `QueryRequest`, `QueryResponse`, `ChunkMetadata`).

---

## 3. Imports Organization
Enforced strictly by `ruff` (isort rule `I001`):
1. **Standard Library Imports** (e.g., `os`, `sys`, `typing`, `dataclasses`, `hashlib`)
2. **Third-Party Imports** (e.g., `fastapi`, `pydantic`, `torch`, `qdrant_client`)
3. **Local Application Imports** (e.g., `from src.common.config import settings`)

```python
# Correct import structure
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.common.exceptions import RetrievalTimeoutError
from src.retrieval.dense import DenseRetriever
```

---

## 4. Async & Concurrency Rules
- **Non-blocking Event Loop:** Never call synchronous I/O or sleep inside async FastAPI route handlers.
- **CPU-bound Offloading:** Heavy CPU operations (such as Cross-Encoder reranking or tokenizer encoding) must be executed in a threadpool executor via `asyncio.to_thread()` or `concurrent.futures.ThreadPoolExecutor`:
  ```python
  # Offload CPU-bound reranking to prevent freezing FastAPI event loop
  results = await asyncio.to_thread(self.cross_encoder.predict, pairs)
  ```
- **Async Client Reuse:** Create HTTP and DB clients once during lifespan startup, never re-instantiate clients on every request.

---

## 5. Typing & Pydantic V2 Usage
- Use `Pydantic V2` models for all cross-module data boundaries.
- Configure models with `ConfigDict(strict=True, frozen=True)` where immutability is desired.
- Avoid passing raw untyped dictionaries (`Dict[str, Any]`) across module boundaries.

```python
from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(strict=True)

    chunk_id: str
    source_file: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, str] = Field(default_factory=dict)
```

---

## 6. Error Handling Pattern
- Never use bare `except:` or `except Exception: pass`.
- Define an application exception hierarchy inheriting from `BaseRAGException`:

```python
# src/common/exceptions.py
class BaseRAGException(Exception):
    """Base exception for all OmniDoc RAG domain errors."""

    def __init__(self, message: str, code: str):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DocumentParsingError(BaseRAGException):
    pass


class InferenceEngineUnavailableError(BaseRAGException):
    pass
```

---

## 7. Logging & Telemetry
- Use structured JSON logging via `structlog` or standard `logging` configured with context variables (Correlation ID / Trace ID).
- Include `trace_id` in all log messages triggered within a request lifecycle.
- Log levels:
  - `DEBUG`: Detailed vector similarity scores and chunk payloads.
  - `INFO`: Query lifecycle steps, latencies, cache hit/miss status.
  - `WARNING`: CRAG query rewrites, degraded retrieval fallbacks, cache misses.
  - `ERROR`: Unhandled exceptions, vLLM disconnects, Qdrant timeouts.

---

## 8. Tooling & Enforcement
- **Linter & Formatter:** `ruff` with line-length = 100.
- **Type Checker:** `mypy --strict --ignore-missing-imports`.
- **Pre-commit configuration:**
  ```yaml
  repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.6.9
      hooks:
        - id: ruff
          args: [--fix]
        - id: ruff-format
  ```
