"""Unit tests for the Unified Ingestion Coordinator CLI and Service."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.common.exceptions import DocumentParsingError
from src.common.schemas import DocumentChunkPayload, IngestionSummary
from src.ingestion.indexer import IngestionPipeline, build_arg_parser, main


def _create_sample_payload(
    doc_id: str,
    chunk_id: int,
    source_file: str,
    content: str,
    token_count: int = 50,
) -> DocumentChunkPayload:
    """Helper to construct a valid DocumentChunkPayload."""
    return DocumentChunkPayload(
        doc_id=doc_id,
        chunk_id=chunk_id,
        content=content,
        source_file=source_file,
        section_heading="# Section",
        language="en",
        token_count=token_count,
    )


class TestIngestionPipelineInit:
    """Tests for pipeline initialization and dependency wiring."""

    def test_lazy_initialization_defaults(self) -> None:
        """Pipeline initializes components lazily on first access."""
        pipeline = IngestionPipeline()
        assert pipeline._chunker is None
        assert pipeline._embedder is None
        assert pipeline._qdrant_store is None
        assert pipeline._bm25_store is None

        # Access property triggers lazy instantiation
        assert pipeline.chunker is not None
        assert pipeline._chunker is not None

    def test_injected_components_retained(self) -> None:
        """Injected mocks or instances must be preserved."""
        mock_chunker = MagicMock()
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()
        mock_bm25 = MagicMock()

        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            qdrant_store=mock_qdrant,
            bm25_store=mock_bm25,
        )

        assert pipeline.chunker is mock_chunker
        assert pipeline.embedder is mock_embedder
        assert pipeline.qdrant_store is mock_qdrant
        assert pipeline.bm25_store is mock_bm25


class TestIngestFile:
    """Tests for single file ingestion."""

    def test_ingest_file_success(self, tmp_path: Path) -> None:
        """Valid file is chunked, embedded, and indexed into both stores."""
        test_file = tmp_path / "docker_guide.md"
        test_file.write_text("# Docker Guide\n\nContent here.", encoding="utf-8")

        mock_chunker = MagicMock()
        chunks = [
            _create_sample_payload("doc1", 0, str(test_file), "Chunk 1 content", token_count=40),
            _create_sample_payload("doc1", 1, str(test_file), "Chunk 2 content", token_count=60),
        ]
        mock_chunker.chunk_file.return_value = chunks

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]

        mock_qdrant = MagicMock()
        mock_bm25 = MagicMock()

        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            qdrant_store=mock_qdrant,
            bm25_store=mock_bm25,
        )

        summary = pipeline.ingest_file(file_path=test_file, doc_id="custom_doc_id")

        mock_qdrant.ensure_collection_exists.assert_called_once()
        mock_chunker.chunk_file.assert_called_once_with(
            file_path=test_file.resolve(), doc_id="custom_doc_id"
        )
        mock_embedder.embed_documents.assert_called_once_with(
            ["Chunk 1 content", "Chunk 2 content"]
        )
        mock_qdrant.upsert_chunks.assert_called_once_with(
            chunks=chunks, vectors=[[0.1] * 1024, [0.2] * 1024]
        )
        mock_bm25.upsert_chunks.assert_called_once_with(chunks=chunks)
        mock_bm25.save.assert_called_once()

        assert isinstance(summary, IngestionSummary)
        assert summary.total_files == 1
        assert summary.total_chunks == 2
        assert summary.total_tokens == 100
        assert summary.duration_seconds >= 0.0
        assert summary.processed_files == [str(test_file.resolve())]
        assert summary.errors == []

    def test_ingest_file_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Non-existent file raises DocumentParsingError."""
        missing_file = tmp_path / "non_existent.md"
        pipeline = IngestionPipeline()

        with pytest.raises(DocumentParsingError) as exc_info:
            pipeline.ingest_file(file_path=missing_file)
        assert "not found" in str(exc_info.value)

    def test_ingest_file_unsupported_extension_raises_error(self, tmp_path: Path) -> None:
        """File with unsupported extension raises DocumentParsingError."""
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("Hello world", encoding="utf-8")
        pipeline = IngestionPipeline()

        with pytest.raises(DocumentParsingError) as exc_info:
            pipeline.ingest_file(file_path=txt_file)
        assert "Unsupported document format" in str(exc_info.value)

    def test_ingest_file_empty_chunks_returns_zero_summary(self, tmp_path: Path) -> None:
        """When chunker returns no chunks, empty summary is returned without store upserts."""
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("   ", encoding="utf-8")

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = []
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()
        mock_bm25 = MagicMock()

        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            qdrant_store=mock_qdrant,
            bm25_store=mock_bm25,
        )

        summary = pipeline.ingest_file(file_path=empty_file)

        mock_qdrant.ensure_collection_exists.assert_called_once()
        mock_embedder.embed_documents.assert_not_called()
        mock_qdrant.upsert_chunks.assert_not_called()
        mock_bm25.upsert_chunks.assert_not_called()

        assert summary.total_files == 1
        assert summary.total_chunks == 0
        assert summary.total_tokens == 0


class TestIngestDirectory:
    """Tests for directory batch ingestion."""

    def test_ingest_directory_success(self, tmp_path: Path) -> None:
        """Directory with multiple Markdown files is ingested with bulk upserts."""
        sub_dir = tmp_path / "docs"
        sub_dir.mkdir()
        file1 = sub_dir / "guide1.md"
        file1.write_text("# Guide 1", encoding="utf-8")
        file2 = sub_dir / "guide2.md"
        file2.write_text("# Guide 2", encoding="utf-8")

        chunk1 = _create_sample_payload("doc1", 0, str(file1), "Chunk 1", token_count=50)
        chunk2 = _create_sample_payload("doc2", 0, str(file2), "Chunk 2", token_count=70)

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.side_effect = [[chunk1], [chunk2]]

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]

        mock_qdrant = MagicMock()
        mock_bm25 = MagicMock()

        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            qdrant_store=mock_qdrant,
            bm25_store=mock_bm25,
        )

        summary = pipeline.ingest_directory(dir_path=sub_dir)

        # Ensure collection exists called once
        mock_qdrant.ensure_collection_exists.assert_called_once()
        # Chunker called for both files
        assert mock_chunker.chunk_file.call_count == 2
        # Embedder called once with all aggregated chunks
        mock_embedder.embed_documents.assert_called_once_with(["Chunk 1", "Chunk 2"])
        # Qdrant upsert called once with all chunks
        mock_qdrant.upsert_chunks.assert_called_once_with(
            chunks=[chunk1, chunk2], vectors=[[0.1] * 1024, [0.2] * 1024]
        )
        # BM25 upsert and save called once
        mock_bm25.upsert_chunks.assert_called_once_with(chunks=[chunk1, chunk2])
        mock_bm25.save.assert_called_once()

        assert summary.total_files == 2
        assert summary.total_chunks == 2
        assert summary.total_tokens == 120
        assert summary.errors == []

    def test_ingest_directory_missing_dir_raises_error(self, tmp_path: Path) -> None:
        """Non-existent directory raises DocumentParsingError."""
        missing_dir = tmp_path / "missing_dir"
        pipeline = IngestionPipeline()

        with pytest.raises(DocumentParsingError) as exc_info:
            pipeline.ingest_directory(dir_path=missing_dir)
        assert "not found" in str(exc_info.value)

    def test_ingest_directory_empty_dir_returns_zero(self, tmp_path: Path) -> None:
        """Directory with no matching Markdown files returns 0 summary."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        pipeline = IngestionPipeline()

        summary = pipeline.ingest_directory(dir_path=empty_dir)
        assert summary.total_files == 0
        assert summary.total_chunks == 0
        assert summary.total_tokens == 0

    def test_ingest_directory_ignore_errors_flag(self, tmp_path: Path) -> None:
        """When ignore_errors=True, failed file is recorded in errors and others succeed."""
        dir_path = tmp_path / "mixed_docs"
        dir_path.mkdir()
        file_ok = dir_path / "a_ok.md"
        file_ok.write_text("# OK", encoding="utf-8")
        file_bad = dir_path / "b_bad.md"
        file_bad.write_text("# Bad", encoding="utf-8")

        chunk_ok = _create_sample_payload("doc_ok", 0, str(file_ok), "Content OK", token_count=30)

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.side_effect = [
            [chunk_ok],
            RuntimeError("Parsing corrupted syntax"),
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1] * 1024]

        mock_qdrant = MagicMock()
        mock_bm25 = MagicMock()

        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            qdrant_store=mock_qdrant,
            bm25_store=mock_bm25,
        )

        summary = pipeline.ingest_directory(dir_path=dir_path, ignore_errors=True)

        assert summary.total_files == 1
        assert summary.total_chunks == 1
        assert summary.total_tokens == 30
        assert len(summary.errors) == 1
        assert "Parsing corrupted syntax" in summary.errors[0]

    def test_ingest_directory_fail_fast_on_error(self, tmp_path: Path) -> None:
        """When ignore_errors=False, exception from chunker is re-raised immediately."""
        dir_path = tmp_path / "failing_docs"
        dir_path.mkdir()
        file1 = dir_path / "fail.md"
        file1.write_text("# Fail", encoding="utf-8")

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.side_effect = ValueError("Corrupted markdown structure")

        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            qdrant_store=MagicMock(),
            bm25_store=MagicMock(),
        )

        with pytest.raises(ValueError) as exc_info:
            pipeline.ingest_directory(dir_path=dir_path, ignore_errors=False)
        assert "Corrupted markdown structure" in str(exc_info.value)


class TestAsyncIngestion:
    """Tests for async ingestion wrappers."""

    @pytest.mark.asyncio
    async def test_aingest_file(self, tmp_path: Path) -> None:
        """aingest_file executes asynchronously via threadpool."""
        test_file = tmp_path / "async_doc.md"
        test_file.write_text("# Async Doc\n\nContent.", encoding="utf-8")

        chunk = _create_sample_payload("doc1", 0, str(test_file), "Async content", token_count=25)
        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = [chunk]

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.5] * 1024]

        mock_qdrant = MagicMock()
        mock_bm25 = MagicMock()

        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            qdrant_store=mock_qdrant,
            bm25_store=mock_bm25,
        )

        summary = await pipeline.aingest_file(file_path=test_file)
        assert summary.total_files == 1
        assert summary.total_chunks == 1
        assert summary.total_tokens == 25

    @pytest.mark.asyncio
    async def test_aingest_directory(self, tmp_path: Path) -> None:
        """aingest_directory executes asynchronously via threadpool."""
        dir_path = tmp_path / "async_dir"
        dir_path.mkdir()
        test_file = dir_path / "doc.md"
        test_file.write_text("# Async Doc", encoding="utf-8")

        chunk = _create_sample_payload(
            "doc1", 0, str(test_file), "Async dir content", token_count=45
        )
        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = [chunk]

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.5] * 1024]

        mock_qdrant = MagicMock()
        mock_bm25 = MagicMock()

        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            qdrant_store=mock_qdrant,
            bm25_store=mock_bm25,
        )

        summary = await pipeline.aingest_directory(dir_path=dir_path)
        assert summary.total_files == 1
        assert summary.total_chunks == 1
        assert summary.total_tokens == 45


class TestCLIEntrypoint:
    """Tests for the Ingestion Coordinator CLI runner."""

    def test_build_arg_parser(self) -> None:
        """Parser correctly parses required and optional CLI flags."""
        parser = build_arg_parser()
        args = parser.parse_args(["--path", "data/raw", "--pattern", "*.md", "--ignore-errors"])
        assert args.path == "data/raw"
        assert args.pattern == "*.md"
        assert args.ignore_errors is True

    def test_cli_missing_path_exits_with_error(self) -> None:
        """Calling CLI with non-existent path exits with code 1."""
        ret = main(["--path", "completely_non_existent_folder_or_file_12345"])
        assert ret == 1

    def test_cli_single_file_success(self, tmp_path: Path) -> None:
        """Calling CLI with existing file invokes ingest_file and returns 0."""
        test_file = tmp_path / "cli_test.md"
        test_file.write_text("# CLI Test", encoding="utf-8")

        dummy_summary = IngestionSummary(
            total_files=1,
            total_chunks=3,
            total_tokens=150,
            duration_seconds=0.42,
            processed_files=[str(test_file)],
            errors=[],
        )

        with patch.object(
            IngestionPipeline, "ingest_file", return_value=dummy_summary
        ) as mock_ingest:
            ret = main(["--path", str(test_file), "--doc-id", "test_doc"])
            assert ret == 0
            mock_ingest.assert_called_once_with(file_path=test_file.resolve(), doc_id="test_doc")

    def test_cli_directory_success(self, tmp_path: Path) -> None:
        """Calling CLI with directory invokes ingest_directory and returns 0."""
        test_dir = tmp_path / "cli_dir"
        test_dir.mkdir()

        dummy_summary = IngestionSummary(
            total_files=2,
            total_chunks=5,
            total_tokens=250,
            duration_seconds=0.88,
            processed_files=["f1.md", "f2.md"],
            errors=["Warning: small file"],
        )

        with patch.object(
            IngestionPipeline, "ingest_directory", return_value=dummy_summary
        ) as mock_ingest:
            ret = main(["--path", str(test_dir), "--pattern", "*.md", "--ignore-errors"])
            assert ret == 0
            mock_ingest.assert_called_once_with(
                dir_path=test_dir.resolve(),
                pattern="*.md",
                ignore_errors=True,
            )

    def test_cli_handles_unhandled_exception(self, tmp_path: Path) -> None:
        """Unhandled exception in pipeline returns exit code 1."""
        test_file = tmp_path / "broken.md"
        test_file.write_text("# Broken", encoding="utf-8")

        with patch.object(
            IngestionPipeline, "ingest_file", side_effect=RuntimeError("Fatal error")
        ):
            ret = main(["--path", str(test_file)])
            assert ret == 1
