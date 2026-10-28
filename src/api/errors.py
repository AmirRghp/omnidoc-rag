"""RFC 7807 Problem Details exception formatters and FastAPI global error handlers."""

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import ProblemDetails
from src.common.exceptions import BaseRAGException
from src.common.logger import get_logger, get_trace_id

logger = get_logger(__name__)


def resolve_trace_id(request: Request | None = None) -> str:
    """Resolve the active correlation trace_id from contextvars, request state, or headers.

    Args:
        request: Optional incoming FastAPI/Starlette Request.

    Returns:
        Non-empty trace_id string.
    """
    ctx_trace = get_trace_id()
    if ctx_trace:
        return ctx_trace

    if request is not None:
        state_trace = getattr(request.state, "trace_id", None)
        if isinstance(state_trace, str) and state_trace.strip():
            return state_trace.strip()

        header_trace = request.headers.get("X-Trace-ID") or request.headers.get("x-trace-id")
        if header_trace and header_trace.strip():
            return header_trace.strip()

    return f"tr_{uuid.uuid4().hex[:12]}"


def build_problem_details(
    exc: BaseRAGException,
    instance: str = "",
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Convert a BaseRAGException into a validated RFC 7807 ProblemDetails dictionary.

    Args:
        exc: Domain exception instance.
        instance: Request path where the error occurred.
        trace_id: Optional correlation trace ID.

    Returns:
        Dictionary with keys {type, title, status, detail, instance, trace_id, code, timestamp}.
    """
    raw = exc.to_problem_details(instance=instance)
    resolved_trace = trace_id or get_trace_id() or f"tr_{uuid.uuid4().hex[:12]}"

    problem = ProblemDetails(
        type=str(raw["type"]),
        title=str(raw["title"]),
        status=int(raw["status"]),
        detail=str(raw["detail"]),
        instance=str(raw["instance"]),
        trace_id=resolved_trace,
        code=str(raw["code"]),
        timestamp=str(raw["timestamp"]),
    )
    return problem.model_dump()


async def rag_exception_handler(
    request: Request,
    exc: BaseRAGException,
) -> JSONResponse:
    """FastAPI exception handler for BaseRAGException and all domain subclasses."""
    trace_id = resolve_trace_id(request)
    instance = str(request.url.path)
    payload = build_problem_details(exc=exc, instance=instance, trace_id=trace_id)

    logger.warning(
        "rag_domain_exception",
        code=exc.code,
        status=exc.status_code,
        instance=instance,
        trace_id=trace_id,
        detail=exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload,
        headers={"X-Trace-ID": trace_id},
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """FastAPI exception handler converting Pydantic/RequestValidationError to RFC 7807."""
    trace_id = resolve_trace_id(request)
    instance = str(request.url.path)
    errors = exc.errors()
    first_msg = "Request payload failed schema validation."
    if errors:
        first_err = errors[0]
        loc = ".".join(str(part) for part in first_err.get("loc", []))
        msg = str(first_err.get("msg", "Invalid value"))
        first_msg = f"{loc}: {msg}" if loc else msg

    problem = ProblemDetails(
        type="https://api.omnidoc.local/errors/VALIDATION_ERROR",
        title="Request Validation Failed",
        status=422,
        detail=first_msg,
        instance=instance,
        trace_id=trace_id,
        code="ERR_VALIDATION_FAILED",
        timestamp=datetime.now(UTC).isoformat(),
    )

    logger.info(
        "request_validation_failed",
        instance=instance,
        trace_id=trace_id,
        detail=first_msg,
    )
    return JSONResponse(
        status_code=422,
        content=problem.model_dump(),
        headers={"X-Trace-ID": trace_id},
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Fallback exception handler formatting unexpected server errors as safe RFC 7807 JSON."""
    trace_id = resolve_trace_id(request)
    instance = str(request.url.path)

    logger.error(
        "unhandled_server_exception",
        instance=instance,
        trace_id=trace_id,
        error_type=type(exc).__name__,
    )

    problem = ProblemDetails(
        type="https://api.omnidoc.local/errors/INTERNAL_ERROR",
        title="Internal Server Error",
        status=500,
        detail="An unexpected internal error occurred. Please reference trace_id for support.",
        instance=instance,
        trace_id=trace_id,
        code="ERR_INTERNAL_ERROR",
        timestamp=datetime.now(UTC).isoformat(),
    )
    return JSONResponse(
        status_code=500,
        content=problem.model_dump(),
        headers={"X-Trace-ID": trace_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all RFC 7807 exception handlers on a FastAPI application instance."""
    app.add_exception_handler(
        BaseRAGException,
        rag_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )
