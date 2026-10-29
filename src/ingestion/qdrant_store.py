"""Qdrant Vector Store Manager and upsert operations."""

import asyncio
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.common.config import Settings, create_settings
from src.common.exceptions import VectorStoreTimeoutError
from src.common.logger import get_logger
from src.common.schemas import DocumentChunkPayload

logger = get_logger(__name__)


def resolve_qdrant_https(settings: Settings) -> bool | None:
    """Resolve HTTPS flag for Qdrant connections, defaulting to False on localhost."""
    if settings.qdrant_https is not None:
        return settings.qdrant_https
    if settings.qdrant_host in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return False
    return None


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Create a synchronous QdrantClient from application settings."""
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        api_key=settings.qdrant_api_key,
        https=resolve_qdrant_https(settings),
        prefer_grpc=False,
    )


def create_async_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Create an asynchronous AsyncQdrantClient from application settings."""
    return AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        api_key=settings.qdrant_api_key,
        https=resolve_qdrant_https(settings),
        prefer_grpc=False,
    )


class QdrantStoreManager:
    """Manages collection lifecycle and batched idempotent upserts in Qdrant."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        async_client: AsyncQdrantClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the QdrantStoreManager.

        Args:
            client: Optional pre-configured synchronous QdrantClient.
            async_client: Optional pre-configured asynchronous AsyncQdrantClient.
            settings: Optional application settings.
        """
        self.settings = settings or create_settings()
        self.collection_name = self.settings.collection_name
        self.embedding_dimension = self.settings.embedding_dimension

        self._client = client
        self._async_client = async_client

    @property
    def client(self) -> QdrantClient:
        """Get or create synchronous QdrantClient."""
        if self._client is None:
            self._client = create_qdrant_client(self.settings)
        return self._client

    @property
    def async_client(self) -> AsyncQdrantClient:
        """Get or create asynchronous AsyncQdrantClient."""
        if self._async_client is None:
            self._async_client = create_async_qdrant_client(self.settings)
        return self._async_client

    def ensure_collection_exists(self) -> bool:
        """Ensure the target collection exists with specified schema and payload indexes.

        Returns:
            True if collection was created, False if it already existed.
        """
        try:
            if self.client.collection_exists(self.collection_name):
                logger.debug(
                    "qdrant_collection_already_exists",
                    collection=self.collection_name,
                )
                return False

            logger.info(
                "creating_qdrant_collection",
                collection=self.collection_name,
                dim=self.embedding_dimension,
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE,
                    on_disk=False,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=1000,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                    indexing_threshold=10000,
                ),
            )

            # Create payload indexes on all schema fields per docs/DATABASE.md §3.1
            payload_index_specs: list[tuple[str, models.PayloadSchemaType]] = [
                ("doc_id", models.PayloadSchemaType.KEYWORD),
                ("chunk_id", models.PayloadSchemaType.INTEGER),
                ("source_file", models.PayloadSchemaType.KEYWORD),
                ("section_heading", models.PayloadSchemaType.KEYWORD),
                ("language", models.PayloadSchemaType.KEYWORD),
                ("token_count", models.PayloadSchemaType.INTEGER),
                ("created_at", models.PayloadSchemaType.DATETIME),
            ]
            for field_name, field_schema in payload_index_specs:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
                logger.debug(
                    "created_payload_index",
                    collection=self.collection_name,
                    field=field_name,
                )

            return True

        except (ResponseHandlingException, TimeoutError) as exc:
            logger.error("qdrant_connection_timeout", error=str(exc))
            raise VectorStoreTimeoutError(f"Timed out communicating with Qdrant: {exc}") from exc
        except UnexpectedResponse as exc:
            logger.error("qdrant_unexpected_response", error=str(exc))
            raise VectorStoreTimeoutError(f"Unexpected response from Qdrant: {exc}") from exc

    def upsert_chunks(
        self,
        chunks: list[DocumentChunkPayload],
        vectors: list[list[float]],
        batch_size: int = 64,
    ) -> int:
        """Idempotently upsert document chunks and dense vectors in batches.

        Args:
            chunks: List of DocumentChunkPayload objects.
            vectors: List of 1024-dimensional float vector embeddings.
            batch_size: Number of points to upsert per batch.

        Returns:
            Total count of points upserted.
        """
        if not chunks:
            return 0

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: chunks length ({len(chunks)}) != vectors length ({len(vectors)})"
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.embedding_dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} does not match "
                    f"expected {self.embedding_dimension}"
                )

            points.append(
                models.PointStruct(
                    id=chunk.point_id,
                    vector=vector,
                    payload=chunk.model_dump(),
                )
            )

        total_upserted = 0
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )
                total_upserted += len(batch)

            logger.info(
                "qdrant_chunks_upserted",
                collection=self.collection_name,
                count=total_upserted,
            )
            return total_upserted

        except (ResponseHandlingException, TimeoutError) as exc:
            logger.error("qdrant_upsert_timeout", error=str(exc))
            raise VectorStoreTimeoutError(f"Timed out upserting points to Qdrant: {exc}") from exc
        except UnexpectedResponse as exc:
            logger.error("qdrant_upsert_unexpected_response", error=str(exc))
            raise VectorStoreTimeoutError(
                f"Unexpected response while upserting points to Qdrant: {exc}"
            ) from exc

    async def aupsert_chunks(
        self,
        chunks: list[DocumentChunkPayload],
        vectors: list[list[float]],
        batch_size: int = 64,
    ) -> int:
        """Asynchronously upsert document chunks via asyncio.to_thread.

        Offloads network I/O and point formatting from the event loop.
        """
        return await asyncio.to_thread(self.upsert_chunks, chunks, vectors, batch_size)

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection metrics and point counts.

        Returns:
            Dictionary containing points_count, indexed_vectors_count, status.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(info.status),
            }
        except (ResponseHandlingException, TimeoutError) as exc:
            raise VectorStoreTimeoutError(f"Qdrant timeout: {exc}") from exc
        except UnexpectedResponse as exc:
            raise VectorStoreTimeoutError(f"Qdrant error: {exc}") from exc

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points associated with a specific document doc_id.

        Args:
            doc_id: Identifier of the document to delete.
        """
        if not doc_id.strip():
            raise ValueError("doc_id cannot be empty")

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id),
                            )
                        ]
                    )
                ),
            )
            logger.info(
                "qdrant_points_deleted_by_doc_id",
                collection=self.collection_name,
                doc_id=doc_id,
            )
        except (ResponseHandlingException, TimeoutError) as exc:
            raise VectorStoreTimeoutError(f"Qdrant timeout: {exc}") from exc
        except UnexpectedResponse as exc:
            raise VectorStoreTimeoutError(f"Qdrant error: {exc}") from exc
