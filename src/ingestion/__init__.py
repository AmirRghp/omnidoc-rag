"""Document ingestion, parsing, chunking, and indexing pipeline."""

from src.ingestion.chunker import MarkdownSemanticChunker

__all__ = ["MarkdownSemanticChunker"]
