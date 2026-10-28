"""Unified Ingestion Coordinator CLI and Service for OmniDoc RAG.

Coordinates MarkdownSemanticChunker, BGEEmbedder, QdrantStoreManager,
and BM25StoreManager into an end-to-end idempotent dual-indexing pipeline.
"""

import argparse
import asyncio
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from src.common.config import Settings, create_settings
from src.common.exceptions import DocumentParsingError
from src.common.logger import get_logger
from src.common.schemas import DocumentChunkPayload, IngestionSummary
from src.ingestion.bm25_store import BM25StoreManager
from src.ingestion.chunker import MarkdownSemanticChunker
from src.ingestion.embedder import BGEEmbedder
from src.ingestion.qdrant_store import QdrantStoreManager

logger = get_logger(__name__)

SUPPORTED_MARKDOWN_EXTENSIONS: set[str] = {".md", ".markdown"}

__all__ = [
    "IngestionPipeline",
    "IngestionSummary",
    "build_arg_parser",
    "main",
]


class IngestionPipeline:
    """Coordinates parsing, chunking, dense embedding, and dual-store indexing."""

    def __init__(
        self,
        chunker: MarkdownSemanticChunker | None = None,
        embedder: BGEEmbedder | None = None,
        qdrant_store: QdrantStoreManager | None = None,
        bm25_store: BM25StoreManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the IngestionPipeline with optional injected components.

        Unprovided components are lazily initialized from configuration settings.
        """
        self.settings = settings or create_settings()
        self._chunker = chunker
        self._embedder = embedder
        self._qdrant_store = qdrant_store
        self._bm25_store = bm25_store

    @property
    def chunker(self) -> MarkdownSemanticChunker:
        """Get or lazily instantiate MarkdownSemanticChunker."""
        if self._chunker is None:
            self._chunker = MarkdownSemanticChunker()
        return self._chunker

    @property
    def embedder(self) -> BGEEmbedder:
        """Get or lazily instantiate BGEEmbedder."""
        if self._embedder is None:
            self._embedder = BGEEmbedder(settings=self.settings)
        return self._embedder

    @property
    def qdrant_store(self) -> QdrantStoreManager:
        """Get or lazily instantiate QdrantStoreManager."""
        if self._qdrant_store is None:
            self._qdrant_store = QdrantStoreManager(settings=self.settings)
        return self._qdrant_store

    @property
    def bm25_store(self) -> BM25StoreManager:
        """Get or lazily instantiate BM25StoreManager."""
        if self._bm25_store is None:
            self._bm25_store = BM25StoreManager()
        return self._bm25_store

    def ingest_file(
        self,
        file_path: Path | str,
        doc_id: str | None = None,
    ) -> IngestionSummary:
        """Parse, embed, and index a single Markdown document into Qdrant and BM25.

        Args:
            file_path: Path to the Markdown file.
            doc_id: Optional explicit document identifier.

        Returns:
            IngestionSummary containing chunk count, token volume, and duration.

        Raises:
            DocumentParsingError: If file is missing, unreadable, or not a Markdown file.
        """
        start_time = time.perf_counter()
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise DocumentParsingError(
                message=f"Document file not found: {path}",
                detail=f"Path '{path}' does not exist or is not a regular file.",
            )

        if path.suffix.lower() not in SUPPORTED_MARKDOWN_EXTENSIONS:
            raise DocumentParsingError(
                message=f"Unsupported document format: {path.suffix}",
                detail=(
                    f"File '{path.name}' has extension '{path.suffix}'. "
                    f"Only Markdown files ({', '.join(sorted(SUPPORTED_MARKDOWN_EXTENSIONS))}) "
                    "are supported."
                ),
            )

        logger.info("ingesting_file_start", file_path=str(path), doc_id=doc_id)

        # 1. Ensure target Qdrant collection exists
        self.qdrant_store.ensure_collection_exists()

        # 2. Chunk Markdown into validated DocumentChunkPayload objects
        chunks = self.chunker.chunk_file(file_path=path, doc_id=doc_id)

        if not chunks:
            duration = round(time.perf_counter() - start_time, 4)
            logger.info("ingesting_file_empty", file_path=str(path), duration=duration)
            return IngestionSummary(
                total_files=1,
                total_chunks=0,
                total_tokens=0,
                duration_seconds=duration,
                processed_files=[str(path)],
                errors=[],
            )

        # 3. Dense Embeddings generation
        texts = [chunk.content for chunk in chunks]
        vectors = self.embedder.embed_documents(texts)

        # 4. Upsert dense vectors and payload to Qdrant
        self.qdrant_store.upsert_chunks(chunks=chunks, vectors=vectors)

        # 5. Upsert chunks into BM25 lexical store and persist checkpoint atomically
        self.bm25_store.upsert_chunks(chunks=chunks)
        self.bm25_store.save()

        total_tokens = sum(chunk.token_count for chunk in chunks)
        duration = round(time.perf_counter() - start_time, 4)

        logger.info(
            "ingesting_file_completed",
            file_path=str(path),
            chunks=len(chunks),
            tokens=total_tokens,
            duration=duration,
        )

        return IngestionSummary(
            total_files=1,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            duration_seconds=duration,
            processed_files=[str(path)],
            errors=[],
        )

    def ingest_directory(
        self,
        dir_path: Path | str,
        pattern: str = "**/*.md",
        ignore_errors: bool = False,
    ) -> IngestionSummary:
        """Batch ingest all Markdown files matching pattern from a directory.

        Args:
            dir_path: Directory path containing documentation files.
            pattern: Glob pattern to filter files (default: '**/*.md').
            ignore_errors: If True, log errors for failed files and continue processing.

        Returns:
            IngestionSummary aggregating metrics across all processed files.

        Raises:
            DocumentParsingError: If directory does not exist or is not a directory.
        """
        start_time = time.perf_counter()
        directory = Path(dir_path).resolve()

        if not directory.exists() or not directory.is_dir():
            raise DocumentParsingError(
                message=f"Documentation directory not found: {directory}",
                detail=f"Path '{directory}' does not exist or is not a directory.",
            )

        matching_files = sorted(
            [
                f
                for f in directory.glob(pattern)
                if f.is_file() and f.suffix.lower() in SUPPORTED_MARKDOWN_EXTENSIONS
            ]
        )

        if not matching_files:
            duration = round(time.perf_counter() - start_time, 4)
            logger.info(
                "ingest_directory_no_files_found",
                directory=str(directory),
                pattern=pattern,
            )
            return IngestionSummary(
                total_files=0,
                total_chunks=0,
                total_tokens=0,
                duration_seconds=duration,
                processed_files=[],
                errors=[],
            )

        logger.info(
            "ingest_directory_start",
            directory=str(directory),
            file_count=len(matching_files),
        )

        # Ensure collection exists once upfront
        self.qdrant_store.ensure_collection_exists()

        all_chunks: list[DocumentChunkPayload] = []
        processed_files: list[str] = []
        errors: list[str] = []

        for file_path in matching_files:
            try:
                file_chunks = self.chunker.chunk_file(file_path=file_path)
                all_chunks.extend(file_chunks)
                processed_files.append(str(file_path))
            except Exception as err:
                msg = f"Failed to chunk '{file_path.name}': {err}"
                logger.error("file_chunking_error", file_path=str(file_path), error=str(err))
                if not ignore_errors:
                    raise
                errors.append(msg)

        if all_chunks:
            # Batch embedding across all gathered chunks
            texts = [chunk.content for chunk in all_chunks]
            vectors = self.embedder.embed_documents(texts)

            # Bulk upsert into Qdrant
            self.qdrant_store.upsert_chunks(chunks=all_chunks, vectors=vectors)

            # Bulk upsert into BM25 and save once atomically
            self.bm25_store.upsert_chunks(chunks=all_chunks)
            self.bm25_store.save()

        total_tokens = sum(chunk.token_count for chunk in all_chunks)
        duration = round(time.perf_counter() - start_time, 4)

        logger.info(
            "ingest_directory_completed",
            directory=str(directory),
            files_processed=len(processed_files),
            chunks=len(all_chunks),
            tokens=total_tokens,
            errors_count=len(errors),
            duration=duration,
        )

        return IngestionSummary(
            total_files=len(processed_files),
            total_chunks=len(all_chunks),
            total_tokens=total_tokens,
            duration_seconds=duration,
            processed_files=processed_files,
            errors=errors,
        )

    async def aingest_file(
        self,
        file_path: Path | str,
        doc_id: str | None = None,
    ) -> IngestionSummary:
        """Asynchronously ingest a single Markdown file offloading CPU work to threadpool."""
        return await asyncio.to_thread(self.ingest_file, file_path, doc_id)

    async def aingest_directory(
        self,
        dir_path: Path | str,
        pattern: str = "**/*.md",
        ignore_errors: bool = False,
    ) -> IngestionSummary:
        """Asynchronously ingest a directory of Markdown files offloading CPU work to threadpool."""
        return await asyncio.to_thread(
            self.ingest_directory,
            dir_path=dir_path,
            pattern=pattern,
            ignore_errors=ignore_errors,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the Ingestion Coordinator CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.indexer",
        description="Unified Ingestion Coordinator CLI for OmniDoc RAG.",
    )
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        type=str,
        help="Path to a Markdown file or directory to ingest.",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.md",
        type=str,
        help="Glob pattern for directory traversal (default: '**/*.md').",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Optional explicit document ID (applicable only for single file ingestion).",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        default=False,
        help="Continue directory ingestion on individual file errors.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI execution entrypoint.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    target_path = Path(args.path).resolve()
    if not target_path.exists():
        sys.stderr.write(f"Error: Target path '{args.path}' does not exist.\n")
        return 1

    pipeline = IngestionPipeline()

    try:
        if target_path.is_file():
            summary = pipeline.ingest_file(file_path=target_path, doc_id=args.doc_id)
        elif target_path.is_dir():
            summary = pipeline.ingest_directory(
                dir_path=target_path,
                pattern=args.pattern,
                ignore_errors=args.ignore_errors,
            )
        else:
            sys.stderr.write(
                f"Error: Target path '{args.path}' is neither a file nor a directory.\n"
            )
            return 1

        sys.stdout.write("=" * 60 + "\n")
        sys.stdout.write("Ingestion Pipeline Summary Report\n")
        sys.stdout.write("=" * 60 + "\n")
        sys.stdout.write(f"Files Processed  : {summary.total_files}\n")
        sys.stdout.write(f"Chunks Created   : {summary.total_chunks}\n")
        sys.stdout.write(f"Total Tokens     : {summary.total_tokens}\n")
        sys.stdout.write(f"Duration (s)     : {summary.duration_seconds:.4f}\n")
        if summary.errors:
            sys.stdout.write(f"Errors ({len(summary.errors)}):\n")
            for err in summary.errors:
                sys.stdout.write(f"  - {err}\n")
        sys.stdout.write("=" * 60 + "\n")
        return 0

    except Exception as exc:
        sys.stderr.write(f"Ingestion failed: {exc}\n")
        logger.exception("ingestion_cli_failure", error=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
