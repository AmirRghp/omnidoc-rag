"""Unit tests for FastAPI request/response schemas and RFC 7807 error handlers (Task 8.1)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.errors import (
    build_problem_details,
    rag_exception_handler,
    register_exception_handlers,
    resolve_trace_id,
)
from src.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    IngestionResponse,
    ProblemDetails,
    QueryRequest,
    QueryResponse,
)
from src.common.exceptions import (
    BaseRAGException,
    CacheServiceUnavailableError,
    ConfigurationError,
    DocumentParsingError,
    InferenceEngineUnavailableError,
    InvalidInputSecurityError,
    VectorStoreTimeoutError,
)
from src.common.logger import clear_trace_id, set_trace_id
from src.common.schemas import CitationItem


class TestQueryRequestSchema:
    """Tests for QueryRequest validation and security normalization."""

    def test_valid_query_request_with_defaults(self) -> None:
        """Valid bilingual query populates default session, temperature, and reflection fields."""
        req = QueryRequest(query="چگونه پورت‌های کانتینر داکر را نگاشت کنیم؟")
        assert req.query == "چگونه پورت‌های کانتینر داکر را نگاشت کنیم؟"
        assert req.session_id is None
        assert req.temperature == 0.2
        assert req.enable_reflection is True

    def test_query_request_normalizes_arabic_variants_and_strips_bidi(self) -> None:
        """QueryRequest normalizes Arabic Yeh/Kaf to Persian and strips bidi overrides."""
        raw = "  \u202eمشكل در تنظيمات داكر ي\u202c  "
        req = QueryRequest(
            query=raw,
            session_id="sess_123",
            temperature=0.0,
            enable_reflection=False,
        )
        assert req.query == "مشکل در تنظیمات داکر ی"
        assert req.session_id == "sess_123"
        assert req.temperature == 0.0
        assert req.enable_reflection is False

    def test_query_request_rejects_empty_or_whitespace_query(self) -> None:
        """Empty or whitespace-only query raises validation/security error."""
        with pytest.raises((ValidationError, InvalidInputSecurityError)):
            QueryRequest(query="   \u202e   ")

    def test_query_request_rejects_query_exceeding_max_length(self) -> None:
        """Query longer than 1,000 characters is rejected."""
        with pytest.raises((ValidationError, InvalidInputSecurityError)):
            QueryRequest(query="a" * 1001)

    def test_query_request_rejects_out_of_bounds_temperature(self) -> None:
        """Temperature outside [0.0, 2.0] raises ValidationError."""
        with pytest.raises(ValidationError):
            QueryRequest(query="docker run", temperature=-0.1)

        with pytest.raises(ValidationError):
            QueryRequest(query="docker run", temperature=2.5)


class TestQueryResponseSchema:
    """Tests for QueryResponse schema and status normalization."""

    def test_valid_query_response_normalizes_reflection_status(self) -> None:
        """Lowercase reflection_status from RAGState is normalized to uppercase API enum."""
        citation = CitationItem(
            source_file="manuals/docker_run.md",
            section_heading="## Exposing and publishing ports",
            relevance_score=0.892,
            snippet="Use the -P or --publish-all flag to publish all exposed ports.",
        )
        resp = QueryResponse(
            query="How to publish all ports?",
            answer="Use `-P` or `--publish-all`.",
            citations=[citation],
            reflection_status="correct",  # type: ignore[arg-type]
            latency_ms=412,
            trace_id="tr_6a81d02c",
        )
        assert resp.reflection_status == "CORRECT"
        assert resp.latency_ms == 412
        assert resp.trace_id == "tr_6a81d02c"
        assert len(resp.citations) == 1
        assert resp.citations[0].source_file == "manuals/docker_run.md"

    def test_query_response_rejects_negative_latency(self) -> None:
        """Negative latency_ms raises ValidationError."""
        with pytest.raises(ValidationError):
            QueryResponse(
                query="test",
                answer="answer",
                citations=[],
                reflection_status="CORRECT",
                latency_ms=-5,
                trace_id="tr_1",
            )


class TestFeedbackAndIngestionSchemas:
    """Tests for FeedbackRequest, FeedbackResponse, IngestionResponse, and ProblemDetails."""

    def test_valid_feedback_request_and_response(self) -> None:
        """FeedbackRequest validates trace_id, rating, score, and optional comment."""
        fb = FeedbackRequest(
            trace_id="tr_6a81d02c",
            rating="up",
            score=1,
            comment="پاسخ بسیار دقیق و کاربردی بود.",
            session_id="sess_8f3a9e12",
        )
        assert fb.trace_id == "tr_6a81d02c"
        assert fb.rating == "up"
        assert fb.score == 1

        fb_resp = FeedbackResponse(trace_id=fb.trace_id)
        assert fb_resp.status == "recorded"
        assert fb_resp.trace_id == "tr_6a81d02c"

    def test_feedback_request_rejects_invalid_score(self) -> None:
        """Score outside [-1, 5] raises ValidationError."""
        with pytest.raises(ValidationError):
            FeedbackRequest(trace_id="tr_1", score=10)

    def test_valid_ingestion_response(self) -> None:
        """IngestionResponse supports both queued (PROCESSING) and finished (COMPLETED) states."""
        ing = IngestionResponse(
            job_id="ingest_9a12c4b8-2026",
            filename="postgresql_connection_pooling.md",
            status="PROCESSING",
            message="File received and queued for semantic chunking and dual-indexing.",
        )
        assert ing.job_id == "ingest_9a12c4b8-2026"
        assert ing.status == "PROCESSING"
        assert ing.chunks_created == 0
        assert ing.error is None

    def test_problem_details_schema_validation(self) -> None:
        """ProblemDetails validates standard RFC 7807 fields."""
        problem = ProblemDetails(
            type="https://api.omnidoc.local/errors/VLLM_CONNECTION_FAILED",
            title="Inference Engine Unreachable",
            status=503,
            detail="Failed to connect to local vLLM server.",
            instance="/api/v1/query/stream",
            code="ERR_VLLM_CONNECTION_FAILED",
            trace_id="tr_6a81d02c",
        )
        assert problem.status == 503
        assert problem.trace_id == "tr_6a81d02c"
        assert problem.timestamp.endswith("Z") or "+" in problem.timestamp


class TestRFC7807ExceptionHandlers:
    """Tests for RFC 7807 formatting and FastAPI global exception handlers."""

    def teardown_method(self) -> None:
        """Reset trace_id context variable after each test."""
        clear_trace_id()

    @pytest.mark.parametrize(
        ("exc_instance", "expected_status", "expected_code"),
        [
            (DocumentParsingError("Bad markdown table"), 422, "ERR_DOCUMENT_PARSING"),
            (VectorStoreTimeoutError("Qdrant timed out"), 504, "ERR_QDRANT_TIMEOUT"),
            (
                InferenceEngineUnavailableError("vLLM offline"),
                503,
                "ERR_VLLM_CONNECTION_FAILED",
            ),
            (
                InvalidInputSecurityError("Prompt injection detected"),
                400,
                "ERR_SECURITY_INPUT_REJECTED",
            ),
            (CacheServiceUnavailableError("Redis down"), 503, "ERR_REDIS_UNAVAILABLE"),
            (ConfigurationError("Missing config"), 500, "ERR_CONFIGURATION_INVALID"),
        ],
    )
    def test_build_problem_details_maps_all_domain_exceptions(
        self,
        exc_instance: BaseRAGException,
        expected_status: int,
        expected_code: str,
    ) -> None:
        """All BaseRAGException subclasses format to RFC 7807 with required keys."""
        payload = build_problem_details(
            exc=exc_instance,
            instance="/api/v1/query",
            trace_id="tr_test_999",
        )
        required_keys = {
            "type",
            "title",
            "status",
            "detail",
            "instance",
            "trace_id",
            "code",
            "timestamp",
        }
        assert required_keys.issubset(payload.keys())
        assert payload["status"] == expected_status
        assert payload["code"] == expected_code
        assert payload["instance"] == "/api/v1/query"
        assert payload["trace_id"] == "tr_test_999"

    def test_fastapi_exception_handlers_end_to_end(self) -> None:
        """Registered handlers return RFC 7807 JSON for domain, 422, and 500 errors."""
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/api/v1/fail-vllm")
        async def fail_vllm() -> dict[str, str]:
            raise InferenceEngineUnavailableError(
                detail="Failed to connect to local vLLM server at http://localhost:8000/v1."
            )

        @app.post("/api/v1/echo-query")
        async def echo_query(body: QueryRequest) -> dict[str, str]:
            return {"query": body.query}

        @app.get("/api/v1/fail-unhandled")
        async def fail_unhandled() -> dict[str, str]:
            raise RuntimeError("sensitive internal database path /var/lib/secret")

        client = TestClient(app, raise_server_exceptions=False)

        # 1. Domain exception (503) with X-Trace-ID header propagation
        res_vllm = client.get(
            "/api/v1/fail-vllm",
            headers={"X-Trace-ID": "tr_header_001"},
        )
        assert res_vllm.status_code == 503
        data_vllm = res_vllm.json()
        assert data_vllm["title"] == "Inference Engine Unreachable"
        assert data_vllm["status"] == 503
        assert data_vllm["instance"] == "/api/v1/fail-vllm"
        assert data_vllm["trace_id"] == "tr_header_001"
        assert data_vllm["code"] == "ERR_VLLM_CONNECTION_FAILED"

        # 2. Request validation error (422) for invalid schema payload
        set_trace_id("tr_ctx_422")
        res_val = client.post("/api/v1/echo-query", json={"temperature": 9.9})
        assert res_val.status_code == 422
        data_val = res_val.json()
        assert data_val["status"] == 422
        assert data_val["instance"] == "/api/v1/echo-query"
        assert data_val["trace_id"] == "tr_ctx_422"
        assert "type" in data_val and "title" in data_val and "detail" in data_val
        clear_trace_id()

        # 3. Unhandled generic exception (500) masks internal details
        res_500 = client.get(
            "/api/v1/fail-unhandled",
            headers={"X-Trace-ID": "tr_500_mask"},
        )
        assert res_500.status_code == 500
        data_500 = res_500.json()
        assert data_500["status"] == 500
        assert data_500["trace_id"] == "tr_500_mask"
        assert data_500["instance"] == "/api/v1/fail-unhandled"
        assert "/var/lib/secret" not in data_500["detail"]

    @pytest.mark.asyncio
    async def test_direct_rag_exception_handler_and_trace_resolution(self) -> None:
        """rag_exception_handler and resolve_trace_id use contextvar when header is absent."""
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/query/stream",
            "headers": [],
        }
        request = Request(scope)
        set_trace_id("tr_from_contextvar")

        assert resolve_trace_id(request) == "tr_from_contextvar"

        exc = InvalidInputSecurityError("Input rejected")
        response = await rag_exception_handler(request, exc)
        assert response.status_code == 400
