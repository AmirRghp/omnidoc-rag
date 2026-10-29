"""Persistent bilingual BM25 lexical indexer with atomic file replacement."""

import asyncio
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.common.logger import get_logger
from src.common.schemas import DocumentChunkPayload, RetrievedChunk
from src.common.security import normalize_persian_text, strip_directional_overrides

logger = get_logger(__name__)

DEFAULT_BM25_INDEX_PATH: Path = Path("data/processed/bm25_index.pkl")

# Matches:
# 1. CLI flags/options (e.g., --ipc=host, -v, --max-connections=100)
# 2. Unix-style paths (e.g., /var/run/docker.sock, /etc/postgresql/postgresql.conf)
# 3. Bilingual words and technical identifiers preserving Persian ZWNJ (\u200c),
#    underscores, dots, and hyphens (e.g., ERR_QDRANT_TIMEOUT, max_connections, اشتراک‌گذاری)
_BILINGUAL_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![\w\u0600-\u06FF])--?[a-zA-Z0-9][\w\-=./:]*"
    r"|(?:/[\w.\-]+){2,}"
    r"|[\w\u0600-\u06FF]+(?:[\u200c._\-][\w\u0600-\u06FF]+)*",
    re.UNICODE,
)


def tokenize_bilingual(text: str) -> list[str]:
    """Tokenize bilingual technical text while preserving CLI flags, paths, and Persian ZWNJ.

    Args:
        text: Raw input text (document chunk or user query).

    Returns:
        List of normalized lowercase tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    cleaned = normalize_persian_text(strip_directional_overrides(text)).lower()
    return [match.group(0) for match in _BILINGUAL_TOKEN_PATTERN.finditer(cleaned)]


class BM25StoreManager:
    """Manages bilingual BM25Okapi index construction, atomic persistence, and lexical search."""

    def __init__(self, index_path: Path | str = DEFAULT_BM25_INDEX_PATH) -> None:
        """Initialize the BM25StoreManager.

        Args:
            index_path: File path for the serialized pickle index.
        """
        self.index_path: Path = Path(index_path)
        self._chunks: list[DocumentChunkPayload] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._dirty: bool = False

    @property
    def signature_path(self) -> Path:
        """Return path to the SHA-256 integrity sidecar file."""
        return self.index_path.with_suffix(f"{self.index_path.suffix}.sha256")

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed chunks currently in memory."""
        return len(self._chunks)

    @property
    def chunks(self) -> list[DocumentChunkPayload]:
        """Return a copy of the indexed DocumentChunkPayload list."""
        return list(self._chunks)

    def _rebuild_in_memory_index(self) -> None:
        """Tokenize current chunks and rebuild the in-memory BM25Okapi instance."""
        self._tokenized_corpus = [
            tokenize_bilingual(f"{chunk.section_heading}\n{chunk.content}")
            for chunk in self._chunks
        ]
        if self._tokenized_corpus and any(self._tokenized_corpus):
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def build_index(
        self,
        chunks: list[DocumentChunkPayload],
        persist: bool = True,
    ) -> int:
        """Build a fresh BM25 index over the provided chunks (deduplicating by point_id).

        Args:
            chunks: List of validated DocumentChunkPayload objects.
            persist: Whether to atomically persist the index to disk after building.

        Returns:
            Total number of unique chunks indexed.
        """
        deduped: dict[str, DocumentChunkPayload] = {}
        for chunk in chunks:
            deduped[chunk.point_id] = chunk

        self._chunks = list(deduped.values())
        self._rebuild_in_memory_index()
        self._dirty = True

        if persist:
            self.save()

        logger.info(
            "bm25_index_built",
            chunk_count=len(self._chunks),
            index_path=str(self.index_path),
            persisted=persist,
        )
        return len(self._chunks)

    def upsert_chunks(
        self,
        chunks: list[DocumentChunkPayload],
        persist: bool = True,
    ) -> int:
        """Idempotently merge new or updated chunks into the BM25 index by point_id.

        Args:
            chunks: List of DocumentChunkPayload objects to insert or update.
            persist: Whether to atomically persist the updated index to disk.

        Returns:
            Total number of unique chunks in the index after upsert.
        """
        if not chunks:
            return len(self._chunks)

        existing_by_id: dict[str, DocumentChunkPayload] = {
            chunk.point_id: chunk for chunk in self._chunks
        }
        for chunk in chunks:
            existing_by_id[chunk.point_id] = chunk

        self._chunks = list(existing_by_id.values())
        self._rebuild_in_memory_index()
        self._dirty = True

        if persist:
            self.save()

        logger.info(
            "bm25_chunks_upserted",
            upserted_count=len(chunks),
            total_chunks=len(self._chunks),
        )
        return len(self._chunks)

    def delete_by_doc_id(self, doc_id: str, persist: bool = True) -> int:
        """Remove all chunks associated with a given doc_id and rebuild the index.

        Args:
            doc_id: Document identifier to remove.
            persist: Whether to atomically persist the updated index to disk.

        Returns:
            Number of chunks removed.
        """
        if not doc_id.strip():
            raise ValueError("doc_id cannot be empty")

        initial_count = len(self._chunks)
        self._chunks = [chunk for chunk in self._chunks if chunk.doc_id != doc_id]
        removed_count = initial_count - len(self._chunks)

        if removed_count > 0:
            self._rebuild_in_memory_index()
            self._dirty = True
            if persist:
                self.save()

        logger.info(
            "bm25_chunks_deleted_by_doc_id",
            doc_id=doc_id,
            removed_count=removed_count,
            remaining_chunks=len(self._chunks),
        )
        return removed_count

    def save(self) -> Path:
        """Atomically persist the BM25 index and chunk payloads to disk via os.replace.

        Returns:
            Path to the saved pickle index file.
        """
        if not self._dirty and self.index_path.is_file():
            return self.index_path

        import hashlib

        target_dir = self.index_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "version": 1,
            "chunks": [chunk.model_dump() for chunk in self._chunks],
            "tokenized_corpus": self._tokenized_corpus,
            "bm25": self._bm25,
        }

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target_dir,
                prefix=".bm25_index_",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                pickle.dump(payload, tmp_file, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())

            digest = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
            os.replace(tmp_path, self.index_path)
            self.signature_path.write_text(digest, encoding="utf-8")
            self._dirty = False
            logger.debug(
                "bm25_index_saved_atomically",
                index_path=str(self.index_path),
                chunk_count=len(self._chunks),
            )
            return self.index_path
        except Exception:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def load(self) -> bool:
        """Load the persisted BM25 index from disk if it exists, verifying SHA-256 integrity.

        Returns:
            True if the index was loaded, False if the index file does not exist.

        Raises:
            ValueError: If the SHA-256 sidecar exists and does not match the index file bytes.
        """
        if not self.index_path.is_file():
            logger.warning(
                "bm25_index_file_not_found",
                index_path=str(self.index_path),
            )
            return False

        import hashlib
        import hmac

        raw_bytes = self.index_path.read_bytes()
        if self.signature_path.is_file():
            expected_digest = self.signature_path.read_text(encoding="utf-8").strip()
            actual_digest = hashlib.sha256(raw_bytes).hexdigest()
            if expected_digest and not hmac.compare_digest(expected_digest, actual_digest):
                raise ValueError(
                    f"BM25 index integrity check failed for '{self.index_path}': SHA-256 mismatch."
                )

        data: dict[str, Any] = pickle.loads(raw_bytes)

        raw_chunks = data.get("chunks", [])
        self._chunks = [
            DocumentChunkPayload.model_validate(item) if isinstance(item, dict) else item
            for item in raw_chunks
        ]
        self._tokenized_corpus = list(data.get("tokenized_corpus", []))
        loaded_bm25 = data.get("bm25")

        if isinstance(loaded_bm25, BM25Okapi):
            self._bm25 = loaded_bm25
        elif self._tokenized_corpus and any(self._tokenized_corpus):
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

        self._dirty = False
        logger.info(
            "bm25_index_loaded",
            index_path=str(self.index_path),
            chunk_count=len(self._chunks),
        )
        return True

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Perform lexical BM25 search over the indexed chunks.

        Args:
            query: Bilingual search query string.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of RetrievedChunk instances with scores calibrated in [0.0, 1.0].
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = tokenize_bilingual(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        scored_indices: list[tuple[int, float]] = [
            (idx, float(score)) for idx, score in enumerate(raw_scores) if float(score) > 0.0
        ]

        if not scored_indices:
            return []

        scored_indices.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        top_candidates = scored_indices[:top_k]

        results: list[RetrievedChunk] = []
        for rank, (chunk_idx, raw_score) in enumerate(top_candidates, start=1):
            chunk = self._chunks[chunk_idx]
            calibrated_score = raw_score / (raw_score + 1.0)
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.point_id,
                    source_file=chunk.source_file,
                    content=chunk.content,
                    score=calibrated_score,
                    metadata={
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_id,
                        "section_heading": chunk.section_heading,
                        "language": chunk.language,
                        "raw_bm25_score": raw_score,
                        "bm25_rank": rank,
                    },
                )
            )

        return results

    async def asearch(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Asynchronously execute BM25 lexical search via asyncio.to_thread."""
        return await asyncio.to_thread(self.search, query, top_k)
