"""Unit tests for core domain Pydantic V2 schemas and point ID generator."""

import uuid

import pytest
from pydantic import ValidationError

from src.common.schemas import (
    CitationItem,
    DocumentChunkPayload,
    EvaluationRecord,
    IngestionSummary,
    RetrievedChunk,
    generate_point_id,
)


class TestPointIdGenerator:
    """Tests for deterministic point ID generation."""

    def test_generate_point_id_idempotency(self) -> None:
        """Generating ID twice with identical parameters must yield identical UUID."""
        id1 = generate_point_id("manuals/docker_daemon.md", 0)
        id2 = generate_point_id("manuals/docker_daemon.md", 0)
        assert id1 == id2

    def test_generate_point_id_valid_uuid_format(self) -> None:
        """Generated ID must parse into a valid RFC 4122 UUID."""
        point_id = generate_point_id("manuals/docker_daemon.md", 1)
        parsed_uuid = uuid.UUID(point_id)
        assert str(parsed_uuid) == point_id

    def test_generate_point_id_distinct_inputs(self) -> None:
        """Different files or different chunk indices must produce distinct IDs."""
        id_chunk_0 = generate_point_id("manuals/docker.md", 0)
        id_chunk_1 = generate_point_id("manuals/docker.md", 1)
        id_other_file = generate_point_id("manuals/postgres.md", 0)

        assert id_chunk_0 != id_chunk_1
        assert id_chunk_0 != id_other_file


class TestDocumentChunkPayload:
    """Tests for DocumentChunkPayload validation and schema constraints."""

    def test_valid_payload_with_auto_point_id(self) -> None:
        """Payload without explicit point_id should auto-generate deterministic UUID."""
        payload = DocumentChunkPayload(
            doc_id="doc_docker_01",
            chunk_id=0,
            content="Docker daemon socket configuration guide.",
            source_file="manuals/docker_daemon.md",
            section_heading="## Daemon socket option",
            language="en",
            token_count=120,
        )
        expected_point_id = generate_point_id("manuals/docker_daemon.md", 0)
        assert payload.point_id == expected_point_id
        assert payload.doc_id == "doc_docker_01"
        assert payload.chunk_id == 0
        assert payload.language == "en"
        assert payload.token_count == 120
        assert "T" in payload.created_at

    def test_valid_payload_with_explicit_point_id(self) -> None:
        """Payload with explicitly passed point_id should preserve it."""
        explicit_id = str(uuid.uuid4())
        payload = DocumentChunkPayload(
            point_id=explicit_id,
            doc_id="doc_fa_01",
            chunk_id=2,
            content="تنظیمات پورت‌های کانتینر داکر",
            source_file="manuals/docker_fa.md",
            section_heading="## تنظیم پورت",
            language="fa",
            token_count=85,
        )
        assert payload.point_id == explicit_id
        assert payload.language == "fa"

    def test_strict_mode_rejects_type_coercion(self) -> None:
        """Strict mode must reject string-coerced integer fields."""
        with pytest.raises(ValidationError):
            DocumentChunkPayload(
                doc_id="doc_01",
                chunk_id="0",  # type: ignore[arg-type]
                content="Some text",
                source_file="manuals/docker.md",
                section_heading="## Intro",
                language="en",
                token_count=10,
            )

    def test_negative_chunk_id_and_token_count_rejected(self) -> None:
        """Negative indices or token counts must fail validation."""
        with pytest.raises(ValidationError):
            DocumentChunkPayload(
                doc_id="doc_01",
                chunk_id=-1,
                content="Some text",
                source_file="manuals/docker.md",
                section_heading="## Intro",
                language="en",
                token_count=10,
            )

        with pytest.raises(ValidationError):
            DocumentChunkPayload(
                doc_id="doc_01",
                chunk_id=0,
                content="Some text",
                source_file="manuals/docker.md",
                section_heading="## Intro",
                language="en",
                token_count=-5,
            )

    def test_empty_content_rejected(self) -> None:
        """Empty text content must be rejected."""
        with pytest.raises(ValidationError):
            DocumentChunkPayload(
                doc_id="doc_01",
                chunk_id=0,
                content="",
                source_file="manuals/docker.md",
                section_heading="## Intro",
                language="en",
                token_count=0,
            )

    def test_invalid_language_code_rejected(self) -> None:
        """Unsupported languages outside 'fa' and 'en' must fail validation."""
        with pytest.raises(ValidationError):
            DocumentChunkPayload(
                doc_id="doc_01",
                chunk_id=0,
                content="Bonjour le monde",
                source_file="manuals/docker.md",
                section_heading="## Intro",
                language="fr",  # type: ignore[arg-type]
                token_count=10,
            )


class TestRetrievedChunk:
    """Tests for RetrievedChunk validation and scoring bounds."""

    def test_valid_retrieved_chunk(self) -> None:
        """Valid candidate chunk creation with string chunk_id and score."""
        chunk = RetrievedChunk(
            chunk_id="doc_docker_01:0",
            source_file="manuals/docker.md",
            content="Sample retrieved text",
            score=0.875,
            metadata={"rrf_rank": 1},
        )
        assert chunk.chunk_id == "doc_docker_01:0"
        assert chunk.score == 0.875
        assert chunk.metadata == {"rrf_rank": 1}

    def test_integer_chunk_id_allowed(self) -> None:
        """Integer chunk_id is explicitly supported."""
        chunk = RetrievedChunk(
            chunk_id=42,
            source_file="manuals/docker.md",
            content="Sample text",
            score=0.5,
        )
        assert chunk.chunk_id == 42
        assert chunk.metadata == {}

    def test_score_boundaries(self) -> None:
        """Score must be within [0.0, 1.0]."""
        RetrievedChunk(
            chunk_id="1",
            source_file="f.md",
            content="text",
            score=0.0,
        )
        RetrievedChunk(
            chunk_id="1",
            source_file="f.md",
            content="text",
            score=1.0,
        )

        with pytest.raises(ValidationError):
            RetrievedChunk(
                chunk_id="1",
                source_file="f.md",
                content="text",
                score=-0.01,
            )

        with pytest.raises(ValidationError):
            RetrievedChunk(
                chunk_id="1",
                source_file="f.md",
                content="text",
                score=1.001,
            )

    def test_strict_mode_rejects_invalid_types(self) -> None:
        """Passing non-dict to metadata or invalid types must raise ValidationError."""
        with pytest.raises(ValidationError):
            RetrievedChunk(
                chunk_id="1",
                source_file="f.md",
                content="text",
                score=0.5,
                metadata="not_a_dict",  # type: ignore[arg-type]
            )


class TestCitationItem:
    """Tests for CitationItem validation and serialization."""

    def test_valid_citation_item(self) -> None:
        """Valid citation creation and dictionary serialization."""
        citation = CitationItem(
            source_file="manuals/docker_run.md",
            section_heading="## Exposing and publishing ports",
            relevance_score=0.892,
            snippet="Use the -P or --publish-all flag to publish all exposed ports.",
        )
        data = citation.model_dump()
        assert data["source_file"] == "manuals/docker_run.md"
        assert data["section_heading"] == "## Exposing and publishing ports"
        assert data["relevance_score"] == 0.892
        assert "publish-all" in data["snippet"]

    def test_invalid_score_and_empty_snippet(self) -> None:
        """Score out of range or empty snippet must fail."""
        with pytest.raises(ValidationError):
            CitationItem(
                source_file="manuals/docker_run.md",
                section_heading="## Header",
                relevance_score=1.5,
                snippet="Valid snippet",
            )

        with pytest.raises(ValidationError):
            CitationItem(
                source_file="manuals/docker_run.md",
                section_heading="## Header",
                relevance_score=0.5,
                snippet="",
            )


class TestEvaluationRecord:
    """Tests for EvaluationRecord validation and metric boundaries."""

    def test_valid_evaluation_record(self) -> None:
        """Valid evaluation record initialization with default ID and timestamp."""
        record = EvaluationRecord(
            run_name="bilingual_eval_baseline",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            embedding_model="BAAI/bge-m3",
            retrieval_strategy="hybrid_rerank",
            context_recall=0.912,
            context_precision=0.864,
            faithfulness=0.941,
            answer_relevancy=0.895,
            p95_latency_ms=420,
            total_samples=50,
            pass_threshold=True,
        )
        assert record.run_name == "bilingual_eval_baseline"
        assert record.retrieval_strategy == "hybrid_rerank"
        assert record.faithfulness == 0.941
        assert uuid.UUID(record.id)  # valid UUID
        assert "T" in record.created_at

    def test_invalid_strategy_rejected(self) -> None:
        """Unsupported retrieval strategy must fail validation."""
        with pytest.raises(ValidationError):
            EvaluationRecord(
                run_name="eval_test",
                model_name="model",
                embedding_model="emb",
                retrieval_strategy="unsupported_strategy",  # type: ignore[arg-type]
                context_recall=0.9,
                context_precision=0.8,
                faithfulness=0.9,
                answer_relevancy=0.8,
                p95_latency_ms=100,
                total_samples=10,
            )

    def test_metric_out_of_bounds_rejected(self) -> None:
        """Scores outside [0.0, 1.0] must raise ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationRecord(
                run_name="eval_test",
                model_name="model",
                embedding_model="emb",
                retrieval_strategy="hybrid",
                context_recall=1.2,  # > 1.0
                context_precision=0.8,
                faithfulness=0.9,
                answer_relevancy=0.8,
                p95_latency_ms=100,
                total_samples=10,
            )

    def test_zero_samples_rejected(self) -> None:
        """Total samples must be >= 1."""
        with pytest.raises(ValidationError):
            EvaluationRecord(
                run_name="eval_test",
                model_name="model",
                embedding_model="emb",
                retrieval_strategy="dense_only",
                context_recall=0.8,
                context_precision=0.8,
                faithfulness=0.8,
                answer_relevancy=0.8,
                p95_latency_ms=100,
                total_samples=0,  # ge=1
            )


class TestIngestionSummary:
    """Tests for IngestionSummary schema validation."""

    def test_valid_ingestion_summary(self) -> None:
        """Valid ingestion summary attributes must pass validation."""
        summary = IngestionSummary(
            total_files=2,
            total_chunks=15,
            total_tokens=3450,
            duration_seconds=1.234,
            processed_files=["docs/guide.md", "docs/intro.md"],
            errors=[],
        )
        assert summary.total_files == 2
        assert summary.total_chunks == 15
        assert summary.total_tokens == 3450
        assert summary.duration_seconds == 1.234
        assert len(summary.processed_files) == 2
        assert summary.errors == []

    def test_negative_values_rejected(self) -> None:
        """Negative file, chunk, token, or duration values must be rejected."""
        with pytest.raises(ValidationError):
            IngestionSummary(
                total_files=-1,
                total_chunks=10,
                total_tokens=100,
                duration_seconds=1.0,
            )

        with pytest.raises(ValidationError):
            IngestionSummary(
                total_files=1,
                total_chunks=-1,
                total_tokens=100,
                duration_seconds=1.0,
            )

        with pytest.raises(ValidationError):
            IngestionSummary(
                total_files=1,
                total_chunks=10,
                total_tokens=-1,
                duration_seconds=1.0,
            )

        with pytest.raises(ValidationError):
            IngestionSummary(
                total_files=1,
                total_chunks=10,
                total_tokens=100,
                duration_seconds=-0.5,
            )

    def test_strict_mode_rejects_coercion(self) -> None:
        """String numbers in integer fields must be rejected in strict mode."""
        with pytest.raises(ValidationError):
            IngestionSummary(
                total_files="1",  # type: ignore[arg-type]
                total_chunks=10,
                total_tokens=100,
                duration_seconds=1.0,
            )

