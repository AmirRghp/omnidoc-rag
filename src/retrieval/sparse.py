"""Sparse lexical retrieval module using BM25StoreManager."""

import asyncio
from pathlib import Path

from src.common.config import Settings, create_settings
from src.common.logger import get_logger
from src.common.schemas import RetrievedChunk
from src.ingestion.bm25_store import BM25StoreManager

logger = get_logger(__name__)

__all__ = ["SparseRetriever"]


class SparseRetriever:
    """Asynchronous and synchronous sparse lexical retriever wrapping BM25StoreManager."""

    def __init__(
        self,
        bm25_store: BM25StoreManager | None = None,
        index_path: Path | str | None = None,
        auto_load: bool = True,
        settings: Settings | None = None,
    ) -> None:
        """Initialize SparseRetriever.

        Args:
            bm25_store: Optional pre-configured BM25StoreManager.
            index_path: Optional custom path to persisted BM25 pickle file.
            auto_load: If True, automatically load index from disk on first query if empty.
            settings: Optional application settings.
        """
        self.settings = settings or create_settings()
        self.index_path = index_path
        self.auto_load = auto_load
        self._bm25_store = bm25_store

    @property
    def bm25_store(self) -> BM25StoreManager:
        """Get or lazily instantiate BM25StoreManager."""
        if self._bm25_store is None:
            if self.index_path is not None:
                self._bm25_store = BM25StoreManager(index_path=self.index_path)
            else:
                self._bm25_store = BM25StoreManager()

            if self.auto_load:
                loaded = self._bm25_store.load()
                logger.debug(
                    "sparse_retriever_index_autoloaded",
                    loaded=loaded,
                    chunks=self._bm25_store.chunk_count,
                )
        return self._bm25_store

    async def asearch(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Asynchronously execute BM25 lexical search offloaded to a worker thread.

        Args:
            query: User search query string.
            top_k: Maximum number of candidates to retrieve.

        Returns:
            Ranked list of RetrievedChunk objects with calibrated scores in [0.0, 1.0].

        Raises:
            ValueError: If top_k is not positive.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        normalized_query = query.strip()
        if not normalized_query:
            return []

        # If store is empty and auto_load is requested, attempt load asynchronously
        if self.auto_load and self.bm25_store.chunk_count == 0:
            await asyncio.to_thread(self.bm25_store.load)

        return await self.bm25_store.asearch(query=normalized_query, top_k=top_k)

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Synchronously execute BM25 lexical search.

        Args:
            query: User search query string.
            top_k: Maximum number of candidates to retrieve.

        Returns:
            Ranked list of RetrievedChunk objects with calibrated scores in [0.0, 1.0].

        Raises:
            ValueError: If top_k is not positive.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        normalized_query = query.strip()
        if not normalized_query:
            return []

        if self.auto_load and self.bm25_store.chunk_count == 0:
            self.bm25_store.load()

        return self.bm25_store.search(query=normalized_query, top_k=top_k)
