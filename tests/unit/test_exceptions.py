"""Unit tests for domain exceptions and structured logging."""

import json
from typing import Any

from src.common.exceptions import (
    BaseRAGException,
    CacheServiceUnavailableError,
    ConfigurationError,
    DocumentParsingError,
    InferenceEngineUnavailableError,
    InvalidInputSecurityError,
    VectorStoreTimeoutError,
)
from src.common.logger import (
    clear_trace_id,
    configure_logger,
    get_logger,
    get_trace_id,
    inject_trace_id,
    set_trace_id,
)


def test_base_rag_exception_problem_details() -> None:
    """Verify BaseRAGException produces valid RFC 7807 dictionary."""
    exc = BaseRAGException(
        message="Custom error message",
        code="ERR_CUSTOM_TEST",
        status_code=400,
        title="Custom Test Error",
        detail="Detailed explanation",
    )
    problem = exc.to_problem_details(instance="/api/v1/test")

    assert problem["type"] == "https://api.omnidoc.local/errors/CUSTOM_TEST"
    assert problem["title"] == "Custom Test Error"
    assert problem["status"] == 400
    assert problem["detail"] == "Detailed explanation"
    assert problem["instance"] == "/api/v1/test"
    assert problem["code"] == "ERR_CUSTOM_TEST"
    assert "timestamp" in problem


def test_exception_subclasses_hierarchy() -> None:
    """Verify specific exception subclasses inherit correctly and have proper status codes."""
    exceptions_to_check = [
        (DocumentParsingError(), 422, "ERR_DOCUMENT_PARSING"),
        (VectorStoreTimeoutError(), 504, "ERR_QDRANT_TIMEOUT"),
        (InferenceEngineUnavailableError(), 503, "ERR_VLLM_CONNECTION_FAILED"),
        (InvalidInputSecurityError(), 400, "ERR_SECURITY_INPUT_REJECTED"),
        (CacheServiceUnavailableError(), 503, "ERR_REDIS_UNAVAILABLE"),
        (ConfigurationError(), 500, "ERR_CONFIGURATION_INVALID"),
    ]

    for exc, expected_status, expected_code in exceptions_to_check:
        assert isinstance(exc, BaseRAGException)
        assert exc.status_code == expected_status
        assert exc.code == expected_code
        details = exc.to_problem_details()
        assert details["status"] == expected_status
        assert details["code"] == expected_code


def test_trace_id_contextvars() -> None:
    """Verify trace_id context variable getters, setters, and clean up."""
    clear_trace_id()
    assert get_trace_id() is None

    test_trace = "trace_abc123_xyz"
    set_trace_id(test_trace)
    assert get_trace_id() == test_trace

    clear_trace_id()
    assert get_trace_id() is None


def test_inject_trace_id_processor() -> None:
    """Verify structlog processor correctly injects trace_id when set."""
    clear_trace_id()
    event_dict = {"event": "test_event"}
    processed = inject_trace_id(None, "info", event_dict)
    assert "trace_id" not in processed

    set_trace_id("req_trace_98765")
    processed = inject_trace_id(None, "info", event_dict)
    assert processed["trace_id"] == "req_trace_98765"
    clear_trace_id()


def test_logger_json_output(capsys: Any) -> None:
    """Verify structured logger formats events as valid JSON containing trace_id."""
    configure_logger(log_level="DEBUG", json_format=True)
    logger = get_logger("test_module")

    set_trace_id("trace_session_456")
    logger.info("Hybrid retrieval started", top_k=20)

    captured = capsys.readouterr()
    log_line = captured.out.strip().split("\n")[-1]
    parsed = json.loads(log_line)

    assert parsed["event"] == "Hybrid retrieval started"
    assert parsed["top_k"] == 20
    assert parsed["level"] == "info"
    assert parsed["trace_id"] == "trace_session_456"
    assert "timestamp" in parsed

    clear_trace_id()
