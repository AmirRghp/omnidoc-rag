"""FastAPI application endpoints, schemas, and RFC 7807 error handlers."""

from src.api.dependencies import get_crag_workflow, get_settings, verify_api_key
from src.api.errors import (
    build_problem_details,
    rag_exception_handler,
    register_exception_handlers,
    resolve_trace_id,
    unhandled_exception_handler,
    validation_exception_handler,
)
from src.api.routes import format_sse, router, tokenize_generation_stream
from src.api.schemas import (
    FeedbackRating,
    FeedbackRequest,
    FeedbackResponse,
    IngestionResponse,
    IngestionStatus,
    ProblemDetails,
    QueryRequest,
    QueryResponse,
    ReflectionStatus,
)

__all__ = [
    "FeedbackRating",
    "FeedbackRequest",
    "FeedbackResponse",
    "IngestionResponse",
    "IngestionStatus",
    "ProblemDetails",
    "QueryRequest",
    "QueryResponse",
    "ReflectionStatus",
    "build_problem_details",
    "format_sse",
    "get_crag_workflow",
    "get_settings",
    "rag_exception_handler",
    "register_exception_handlers",
    "resolve_trace_id",
    "router",
    "tokenize_generation_stream",
    "unhandled_exception_handler",
    "validation_exception_handler",
    "verify_api_key",
]
