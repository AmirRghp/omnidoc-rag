"""Core domain Pydantic V2 data contracts and schema definitions."""

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def generate_point_id(source_file: str, chunk_id: int) -> str:
    """Generate a deterministic UUID string from source_file and chunk_id using MD5."""
    key = f"{source_file}:{chunk_id}".encode()
    md5_hash = hashlib.md5(key).hexdigest()
    return str(uuid.UUID(hex=md5_hash))


class DocumentChunkPayload(BaseModel):
    """Schema for Qdrant document chunk point payload."""

    model_config = ConfigDict(strict=True)

    point_id: str = Field(
        default="",
        description="Deterministic UUIDv4/MD5 point identifier.",
    )
    doc_id: str = Field(..., min_length=1, description="Identifier of the parent document.")
    chunk_id: int = Field(
        ..., ge=0, description="0-indexed sequential chunk index within document."
    )
    content: str = Field(..., min_length=1, description="Raw text content of the chunk (UTF-8).")
    source_file: str = Field(..., min_length=1, description="Relative file path.")
    section_heading: str = Field(..., min_length=1, description="Nearest Markdown header.")
    language: Literal["fa", "en"] = Field(..., description="Primary language detected in chunk.")
    token_count: int = Field(..., ge=0, description="Token count using tokenizer.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description="Ingestion timestamp in ISO 8601 UTC format.",
    )

    @model_validator(mode="before")
    @classmethod
    def _populate_point_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            point_id = data.get("point_id")
            if not point_id and "source_file" in data and "chunk_id" in data:
                data["point_id"] = generate_point_id(
                    str(data["source_file"]), int(data["chunk_id"])
                )
        return data


class RetrievedChunk(BaseModel):
    """Schema representing a retrieved candidate chunk after dense, sparse, or reranking steps."""

    model_config = ConfigDict(strict=True)

    chunk_id: str | int = Field(..., description="Unique chunk or point identifier.")
    source_file: str = Field(..., min_length=1, description="Source document file path.")
    content: str = Field(..., min_length=1, description="Text snippet of the chunk.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated similarity or relevance score."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context such as RRF rank, original score, or headers.",
    )


class CitationItem(BaseModel):
    """Schema representing an individual grounded citation in API responses."""

    model_config = ConfigDict(strict=True)

    source_file: str = Field(..., min_length=1, description="Relative path of cited source file.")
    section_heading: str = Field(..., min_length=1, description="Section heading of cited content.")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score of citation.")
    snippet: str = Field(..., min_length=1, description="Exact cited text excerpt.")


class EvaluationRecord(BaseModel):
    """Schema representing a benchmark evaluation run record stored in PostgreSQL or reports."""

    model_config = ConfigDict(strict=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evaluation run identifier.",
    )
    run_name: str = Field(..., min_length=1, description="Name of the benchmark execution run.")
    model_name: str = Field(..., min_length=1, description="LLM evaluated.")
    embedding_model: str = Field(..., min_length=1, description="Embedding model evaluated.")
    retrieval_strategy: Literal["dense_only", "hybrid", "hybrid_rerank"] = Field(
        ..., description="Retrieval configuration evaluated."
    )
    context_recall: float = Field(..., ge=0.0, le=1.0, description="Ragas context recall score.")
    context_precision: float = Field(
        ..., ge=0.0, le=1.0, description="Ragas context precision score."
    )
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Ragas faithfulness score.")
    answer_relevancy: float = Field(
        ..., ge=0.0, le=1.0, description="Ragas answer relevancy score."
    )
    p95_latency_ms: int = Field(..., ge=0, description="P95 latency in milliseconds.")
    total_samples: int = Field(..., ge=1, description="Number of evaluation test cases evaluated.")
    pass_threshold: bool = Field(
        default=True, description="Whether the run met the CI gating threshold."
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description="Evaluation timestamp in ISO 8601 UTC format.",
    )


class IngestionSummary(BaseModel):
    """Summary metrics and audit record of an ingestion execution run."""

    model_config = ConfigDict(strict=True)

    total_files: int = Field(..., ge=0, description="Total number of files processed.")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks produced and indexed.")
    total_tokens: int = Field(..., ge=0, description="Total token volume across all chunks.")
    duration_seconds: float = Field(
        ..., ge=0.0, description="Elapsed indexing wall-clock time in seconds."
    )
    processed_files: list[str] = Field(
        default_factory=list, description="Relative paths of processed files."
    )
    errors: list[str] = Field(
        default_factory=list, description="Error messages encountered, if any."
    )

