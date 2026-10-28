# Task 8.1: API Schemas & RFC 7807 Exception Handlers

Implement strict Pydantic V2 request/response data contracts (`QueryRequest`, `QueryResponse`, `FeedbackRequest`, `IngestionResponse`, `ProblemDetails`) in `src/api/schemas.py` and global RFC 7807 Problem Details exception handlers in `src/api/errors.py`.

## User Review & Critical Decisions

> [!IMPORTANT]
> Review the proposed API contract and error-handling design below before proceeding to implementation:

- **Input Security Validation at Boundary (`QueryRequest`)**: Validates `query` through `sanitize_query` from `src.common.security` (normalizing Persian characters `ی`/`ک`, stripping invisible bidirectional overrides, rejecting empty/whitespace-only inputs, and enforcing `1..1000` characters) alongside `session_id`, `temperature` ($[0.0, 2.0]$), and `enable_reflection`.
- **Unified RFC 7807 Error Format (`src/api/errors.py`)**: Maps all `BaseRAGException` subclasses (`DocumentParsingError` $\rightarrow 422$, `InvalidInputSecurityError` $\rightarrow 400$, `InferenceEngineUnavailableError` $\rightarrow 503$, `VectorStoreTimeoutError` $\rightarrow 504$, `CacheServiceUnavailableError` $\rightarrow 503$, `ConfigurationError` $\rightarrow 500$), FastAPI `RequestValidationError` ($\rightarrow 422$), and unhandled exceptions ($\rightarrow 500$) into a consistent RFC 7807 payload containing `{ type, title, status, detail, instance, trace_id, code, timestamp }`.
- **Trace ID Propagation**: Automatically resolves `trace_id` from `get_trace_id()` (`contextvars`), `request.state.trace_id`, or `X-Trace-ID` request header (with UUIDv4 fallback) so every error response can be correlated in logs.

## 1. Overview & Core Concept

- **What It Does**: Defines the external REST & SSE API boundary contracts in `src/api/schemas.py` and centralizes RFC 7807 exception formatting and FastAPI handler registration in `src/api/errors.py`.
- **Target Audience / Persona**: Downstream FastAPI middleware (`Task 8.2`), streaming/REST endpoints (`Task 8.3`), and the Chainlit frontend (`Task 9.2`).
- **Key Value**: Enforces strict input validation at the HTTP edge and guarantees that no internal stack traces or unformatted errors leak to clients.

## 2. User Experience & API Behavior

- **Valid Query Request (`POST /api/v1/query` & `/api/v1/query/stream`)**:
  - Client sends `{ "query": "...", "session_id": "...", "temperature": 0.2, "enable_reflection": true }`.
  - `QueryRequest` normalizes Persian/English text and validates parameters.
  - Synchronous endpoint returns `QueryResponse` with `query`, `answer`, `citations` (`list[CitationItem]`), `reflection_status` (`"CORRECT" | "AMBIGUOUS" | "INCORRECT"`), `latency_ms`, and `trace_id`.
- **Domain Exception Flow (RFC 7807)**:
  - If any route or service raises a `BaseRAGException` subclass (e.g., `InferenceEngineUnavailableError`), `rag_exception_handler` catches it, logs the event with `trace_id`, and returns a structured HTTP 503 JSON response matching RFC 7807.

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: Strict Pydantic V2 Models with Normalized Reflection Status**
  - *Chosen Approach*: Configure API schemas with `ConfigDict(strict=True)` while allowing `QueryResponse.reflection_status` to accept either lowercase (`"correct"`) from `RAGState` or uppercase (`"CORRECT"`) per `docs/API.md` (§2.2), normalizing to uppercase `"CORRECT" | "AMBIGUOUS" | "INCORRECT"`.
  - *Why*: Bridges internal `RAGState` literals (`"correct"`, `"ambiguous"`, `"incorrect"`) and the public REST specification (`"CORRECT"`) without manual string conversion in route handlers.
- **Decision 2: Combined RFC 7807 Payload Fields**
  - *Chosen Approach*: Include `type`, `title`, `status`, `detail`, `instance`, `trace_id`, `code`, and `timestamp` in `ProblemDetails` and `build_problem_details`.
  - *Why*: Satisfies both `docs/Plans.md` Task 8.1 (`{ type, title, status, detail, instance, trace_id }`) and `docs/API.md` §3 (`code`, `timestamp`).

## 4. Technical Architecture & Data Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           src/api/schemas.py                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ├── QueryRequest(query, session_id, temperature, enable_reflection)    │
│  ├── QueryResponse(query, answer, citations, reflection_status,         │
│  │                 latency_ms, trace_id)                                │
│  ├── FeedbackRequest(trace_id, rating, score, comment, session_id)      │
│  ├── FeedbackResponse(status, trace_id, message)                        │
│  ├── IngestionResponse(job_id, filename, status, message,               │
│  │                     chunks_created, total_tokens, duration_seconds,  │
│  │                     error)                                           │
│  └── ProblemDetails(type, title, status, detail, instance, trace_id,    │
│                     code, timestamp)                                    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            src/api/errors.py                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ├── resolve_trace_id(request: Request) -> str                          │
│  ├── build_problem_details(exc: BaseRAGException, instance: str,        │
│  │                         trace_id: str) -> dict[str, Any]             │
│  ├── rag_exception_handler(request: Request, exc: BaseRAGException)     │
│  ├── validation_exception_handler(request: Request,                     │
│  │                                exc: RequestValidationError)          │
│  ├── unhandled_exception_handler(request: Request, exc: Exception)      │
│  └── register_exception_handlers(app: FastAPI) -> None                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Target Files
- `src/api/schemas.py`
- `src/api/errors.py`
- `src/api/__init__.py`
- `tests/unit/test_api_schemas.py`
- `docs/Plans.md` (mark Task 8.1 `[x]`)

### Acceptance Criteria & Verification
1. **API Request & Response Schemas (`src/api/schemas.py`)**:
   - `QueryRequest`, `QueryResponse`, `FeedbackRequest`, `IngestionResponse`, and `ProblemDetails` validate valid inputs and reject invalid types/ranges under Pydantic V2.
   - `QueryRequest` sanitizes Persian/English text and rejects empty or $>1000$-char queries.
2. **RFC 7807 Exception Handlers (`src/api/errors.py`)**:
   - All `BaseRAGException` subclasses map to their exact HTTP status code and return `{ type, title, status, detail, instance, trace_id, code, timestamp }`.
   - FastAPI `RequestValidationError` and unexpected `Exception` are also formatted as RFC 7807 JSON responses without leaking internal stack traces.
3. **Verification Commands**:
   - `.venv/bin/pytest tests/unit/test_api_schemas.py -v` passes with 0 failures.
   - `.venv/bin/ruff check src tests` and `.venv/bin/ruff format --check src/api tests/unit/test_api_schemas.py` pass with 0 errors.
   - `.venv/bin/mypy --strict src` passes with 0 errors.
