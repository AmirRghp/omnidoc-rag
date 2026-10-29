"""FastAPI request and response Pydantic V2 schemas."""

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common.schemas import CitationItem
from src.common.security import sanitize_query

UTC = timezone.utc

ReflectionStatus = Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]
FeedbackRating = Literal["up", "down", "positive", "negative"]
IngestionStatus = Literal["PROCESSING", "COMPLETED", "FAILED"]


class QueryRequest(BaseModel):
    """Schema for synchronous and streaming RAG query requests."""

    model_config = ConfigDict(strict=True)

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User technical query in Persian or English.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional conversation session identifier.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )
    enable_reflection: bool = Field(
        default=True,
        description="Whether to run CRAG grading and query reflection.",
    )

    @field_validator("query", mode="before")
    @classmethod
    def _validate_and_sanitize_query(cls, value: Any) -> str:
        """Sanitize and normalize user query at the API boundary."""
        if not isinstance(value, str):
            raise ValueError("Field 'query' must be a string.")
        return sanitize_query(value)


class QueryResponse(BaseModel):
    """Schema for synchronous grounded RAG query response."""

    model_config = ConfigDict(strict=True)

    query: str = Field(..., min_length=1, description="Sanitized user query.")
    answer: str = Field(..., min_length=1, description="Grounded answer or polite refusal.")
    citations: list[CitationItem] = Field(
        default_factory=list,
        description="List of grounded source citations.",
    )
    reflection_status: ReflectionStatus = Field(
        default="CORRECT",
        description="CRAG evaluation status (CORRECT, AMBIGUOUS, or INCORRECT).",
    )
    latency_ms: int = Field(..., ge=0, description="Total query execution latency in milliseconds.")
    trace_id: str = Field(..., min_length=1, description="Request correlation trace ID.")

    @field_validator("reflection_status", mode="before")
    @classmethod
    def _normalize_reflection_status(cls, value: Any) -> str:
        """Accept lowercase RAGState relevance status and normalize to uppercase API contract."""
        if isinstance(value, str):
            upper_val = value.strip().upper()
            if upper_val in {"CORRECT", "AMBIGUOUS", "INCORRECT"}:
                return upper_val
        raise ValueError(
            "Field 'reflection_status' must be one of 'CORRECT', 'AMBIGUOUS', or 'INCORRECT'."
        )


class FeedbackRequest(BaseModel):
    """Schema for user feedback on a generated RAG response."""

    model_config = ConfigDict(strict=True)

    trace_id: str = Field(
        ...,
        min_length=1,
        description="Correlation trace ID of the evaluated response.",
    )
    rating: FeedbackRating | None = Field(
        default="up",
        description="Categorical thumbs-up / thumbs-down rating.",
    )
    score: int = Field(
        default=1,
        ge=-1,
        le=5,
        description="Numeric feedback score (-1 to 5).",
    )
    comment: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional user comment explaining the rating.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional conversation session identifier.",
    )


class FeedbackResponse(BaseModel):
    """Acknowledgement schema for recorded user feedback."""

    model_config = ConfigDict(strict=True)

    status: Literal["recorded"] = Field(
        default="recorded",
        description="Feedback recording status.",
    )
    trace_id: str = Field(..., min_length=1, description="Associated query trace ID.")
    message: str = Field(
        default="Feedback recorded successfully.",
        description="Human-readable confirmation message.",
    )


class IngestionResponse(BaseModel):
    """Schema for document ingestion job trigger and status responses."""

    model_config = ConfigDict(strict=True)

    job_id: str = Field(
        default_factory=lambda: f"ingest_{uuid.uuid4().hex[:12]}",
        min_length=1,
        description="Unique ingestion job identifier.",
    )
    filename: str = Field(
        default="",
        description="Ingested filename or directory path.",
    )
    status: IngestionStatus = Field(
        default="COMPLETED",
        description="Current ingestion job status (PROCESSING, COMPLETED, FAILED).",
    )
    message: str = Field(
        default="File received and queued for semantic chunking and dual-indexing.",
        description="Human-readable status description.",
    )
    chunks_created: int = Field(
        default=0,
        ge=0,
        description="Number of semantic chunks created and indexed.",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens processed across all chunks.",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Elapsed ingestion duration in seconds.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if status is FAILED.",
    )


class ProblemDetails(BaseModel):
    """RFC 7807 Problem Details error response schema."""

    model_config = ConfigDict(strict=True)

    type: str = Field(
        ...,
        min_length=1,
        description="URI reference identifying the problem type.",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Short, human-readable summary of the problem type.",
    )
    status: int = Field(
        ...,
        ge=100,
        le=599,
        description="HTTP status code.",
    )
    detail: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation specific to this occurrence.",
    )
    instance: str = Field(
        ...,
        description="URI reference identifying the specific request path.",
    )
    trace_id: str = Field(
        default_factory=lambda: f"tr_{uuid.uuid4().hex[:12]}",
        min_length=1,
        description="Correlation trace ID for server log inspection.",
    )
    code: str = Field(
        default="ERR_INTERNAL_ERROR",
        min_length=1,
        description="Machine-readable domain error code.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp of the error occurrence.",
    )


class EvalRunRequest(BaseModel):
    """Schema for triggering an evaluation benchmark run (POST /api/v1/eval/run)."""

    model_config = ConfigDict(strict=True)

    sample_limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of golden dataset samples to evaluate.",
    )
    compare_with_baseline: bool = Field(
        default=True,
        description="Whether to compare metrics against baseline floors.",
    )


class EvalMetricsPayload(BaseModel):
    """Aggregate Ragas evaluation metrics summary."""

    model_config = ConfigDict(strict=True)

    context_recall: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)


class EvalRunResponse(BaseModel):
    """Schema for evaluation benchmark run results (docs/API.md §2.5)."""

    model_config = ConfigDict(strict=True)

    run_id: str = Field(
        default_factory=lambda: (
            f"eval_{datetime.now(UTC).strftime('%Y%m%d')}_{uuid.uuid4().hex[:4]}"
        ),
        min_length=1,
        description="Unique evaluation run identifier.",
    )
    metrics: EvalMetricsPayload = Field(..., description="Aggregated Ragas metric scores.")
    p95_latency_ms: int = Field(..., ge=0, description="95th percentile query latency in ms.")
    pass_threshold: bool = Field(..., description="Whether all quality gates passed.")


class HealthResponse(BaseModel):
    """Schema for service health check endpoints."""

    model_config = ConfigDict(strict=True)

    status: Literal["healthy", "degraded"] = Field(default="healthy")
    environment: str = Field(default="development")
    version: str = Field(default="0.1.0")
