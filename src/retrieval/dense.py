"""Dense retrieval module querying Qdrant vector database with degraded fallback."""

import asyncio
import concurrent.futures
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.common.config import Settings, create_settings
from src.common.exceptions import VectorStoreTimeoutError
from src.common.logger import get_logger
from src.common.schemas import RetrievedChunk
from src.ingestion.embedder import BGEEmbedder

logger = get_logger(__name__)

__all__ = ["DenseRetriever"]


class DenseRetriever:
    """Asynchronous dense vector retriever querying Qdrant HNSW index.

    Features a configurable timeout (default 2.0s) and degraded mode fallback:
    when Qdrant times out or encounters network failures, it logs a warning
    and returns an empty list, allowing the hybrid pipeline to proceed with BM25.
    """

    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
        embedder: BGEEmbedder | None = None,
        collection_name: str | None = None,
        timeout_seconds: float = 2.0,
        raise_on_error: bool = False,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the DenseRetriever.

        Args:
            client: Optional pre-configured AsyncQdrantClient.
            embedder: Optional pre-configured BGEEmbedder instance.
            collection_name: Optional target collection name override.
            timeout_seconds: Maximum seconds to wait for vector search before fallback.
            raise_on_error: If True, raise VectorStoreTimeoutError on timeout/failure
                           instead of returning empty results (degraded mode).
            settings: Optional application settings.
        """
        self.settings = settings or create_settings()
        self.collection_name = collection_name or self.settings.collection_name
        self.timeout_seconds = timeout_seconds
        self.raise_on_error = raise_on_error

        self._client = client
        self._embedder = embedder

    @property
    def client(self) -> AsyncQdrantClient:
        """Get or lazily instantiate AsyncQdrantClient."""
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                grpc_port=self.settings.qdrant_grpc_port,
                api_key=self.settings.qdrant_api_key,
                prefer_grpc=False,
            )
        return self._client

    @property
    def embedder(self) -> BGEEmbedder:
        """Get or lazily instantiate BGEEmbedder."""
        if self._embedder is None:
            self._embedder = BGEEmbedder(settings=self.settings)
        return self._embedder

    async def asearch(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Asynchronously query Qdrant using dense vector embeddings with timeout.

        Args:
            query: User search query string.
            top_k: Number of nearest candidate chunks to retrieve (must be positive).

        Returns:
            Ranked list of RetrievedChunk objects with cosine scores in [0.0, 1.0].
            Returns an empty list in degraded fallback mode when Qdrant is unavailable.

        Raises:
            ValueError: If top_k is not positive.
            VectorStoreTimeoutError: If raise_on_error is True and vector search fails.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        normalized_query = query.strip()
        if not normalized_query:
            return []

        try:
            # 1. Generate dense query embedding
            query_vector = await self.embedder.aembed_query(normalized_query)

            # 2. Query Qdrant with enforced timeout
            timeout_val = int(self.timeout_seconds) if self.timeout_seconds >= 1.0 else 2
            response = await asyncio.wait_for(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True,
                    timeout=timeout_val,
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            logger.warning(
                "dense_retrieval_timeout_degraded",
                collection=self.collection_name,
                timeout=self.timeout_seconds,
                error=str(exc),
            )
            if self.raise_on_error:
                raise VectorStoreTimeoutError(
                    message=f"Vector database request timed out after {self.timeout_seconds}s",
                    detail=str(exc),
                ) from exc
            return []
        except (ResponseHandlingException, UnexpectedResponse, Exception) as exc:
            logger.warning(
                "dense_retrieval_failed_degraded",
                collection=self.collection_name,
                error=str(exc),
            )
            if self.raise_on_error:
                raise VectorStoreTimeoutError(
                    message=f"Vector database communication failed: {exc}",
                    detail=str(exc),
                ) from exc
            return []

        # 3. Parse scored points into domain RetrievedChunk schema
        chunks: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            payload: dict[str, Any] = point.payload or {}
            raw_score = float(point.score)
            calibrated_score = max(0.0, min(1.0, raw_score))

            chunks.append(
                RetrievedChunk(
                    chunk_id=str(point.id),
                    source_file=str(payload.get("source_file", "")),
                    content=str(payload.get("content", "")),
                    score=calibrated_score,
                    metadata={
                        "doc_id": str(payload.get("doc_id", "")),
                        "chunk_index": int(payload.get("chunk_id", 0)),
                        "section_heading": str(payload.get("section_heading", "")),
                        "language": str(payload.get("language", "en")),
                        "raw_dense_score": raw_score,
                        "dense_rank": rank,
                    },
                )
            )

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Synchronously execute dense retrieval, safe against active event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, self.asearch(query, top_k)).result()

        return asyncio.run(self.asearch(query, top_k))
