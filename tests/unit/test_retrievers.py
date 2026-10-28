"""Unit tests for Dense and Sparse retrievers with degraded fallback mode."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import QueryResponse

from src.common.exceptions import VectorStoreTimeoutError
from src.common.schemas import RetrievedChunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


def _create_scored_point(
    point_id: str,
    score: float,
    source_file: str = "docs/docker.md",
    content: str = "Docker container lifecycle details.",
    doc_id: str = "doc_docker",
    chunk_id: int = 0,
    section_heading: str = "## Lifecycle",
    language: str = "en",
) -> models.ScoredPoint:
    """Helper to construct a mock Qdrant ScoredPoint."""
    return models.ScoredPoint(
        id=point_id,
        version=1,
        score=score,
        payload={
            "source_file": source_file,
            "content": content,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "section_heading": section_heading,
            "language": language,
        },
    )


class TestDenseRetriever:
    """Unit tests for DenseRetriever."""

    def test_initialization_defaults_and_injection(self) -> None:
        """Test default parameters and component dependency injection."""
        mock_client = MagicMock()
        mock_embedder = MagicMock()

        retriever = DenseRetriever(
            client=mock_client,
            embedder=mock_embedder,
            collection_name="custom_collection",
            timeout_seconds=3.5,
            raise_on_error=True,
        )

        assert retriever.client is mock_client
        assert retriever.embedder is mock_embedder
        assert retriever.collection_name == "custom_collection"
        assert retriever.timeout_seconds == 3.5
        assert retriever.raise_on_error is True

    @pytest.mark.asyncio
    async def test_asearch_success(self) -> None:
        """Normal search converts ScoredPoint results to domain RetrievedChunk models."""
        mock_embedder = MagicMock()
        mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 1024)

        point1 = _create_scored_point("id-1", score=0.88, content="First chunk")
        point2 = _create_scored_point("id-2", score=0.72, content="Second chunk")
        query_response = QueryResponse(points=[point1, point2])

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(return_value=query_response)

        retriever = DenseRetriever(
            client=mock_client,
            embedder=mock_embedder,
            collection_name="test_collection",
            timeout_seconds=2.0,
        )

        results = await retriever.asearch("docker container run", top_k=5)

        mock_embedder.aembed_query.assert_awaited_once_with("docker container run")
        mock_client.query_points.assert_awaited_once_with(
            collection_name="test_collection",
            query=[0.1] * 1024,
            limit=5,
            with_payload=True,
            timeout=2,
        )

        assert len(results) == 2
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].chunk_id == "id-1"
        assert results[0].score == 0.88
        assert results[0].content == "First chunk"
        assert results[0].metadata["dense_rank"] == 1
        assert results[0].metadata["raw_dense_score"] == 0.88

        assert results[1].chunk_id == "id-2"
        assert results[1].score == 0.72
        assert results[1].metadata["dense_rank"] == 2

    def test_search_synchronous_wrapper(self) -> None:
        """Synchronous search wrapper yields results matching asearch."""
        mock_embedder = MagicMock()
        mock_embedder.aembed_query = AsyncMock(return_value=[0.2] * 1024)

        point = _create_scored_point("id-sync", score=0.91, content="Sync content")
        query_response = QueryResponse(points=[point])

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(return_value=query_response)

        retriever = DenseRetriever(
            client=mock_client,
            embedder=mock_embedder,
        )

        results = retriever.search("sync query", top_k=3)
        assert len(results) == 1
        assert results[0].chunk_id == "id-sync"
        assert results[0].score == 0.91

    @pytest.mark.asyncio
    async def test_validation_empty_query_and_invalid_top_k(self) -> None:
        """Invalid top_k raises ValueError; whitespace query returns empty list."""
        retriever = DenseRetriever()

        with pytest.raises(ValueError) as exc:
            await retriever.asearch("query", top_k=0)
        assert "positive" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            await retriever.asearch("query", top_k=-5)
        assert "positive" in str(exc.value)

        # Whitespace and empty queries
        assert await retriever.asearch("") == []
        assert await retriever.asearch("   ") == []

    @pytest.mark.asyncio
    async def test_timeout_fallback_to_degraded_mode(self) -> None:
        """When Qdrant times out with raise_on_error=False, return empty list (degraded)."""
        mock_embedder = MagicMock()
        mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 1024)

        async def slow_query(*args: object, **kwargs: object) -> QueryResponse:
            await asyncio.sleep(0.5)
            return QueryResponse(points=[])

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(side_effect=slow_query)

        retriever = DenseRetriever(
            client=mock_client,
            embedder=mock_embedder,
            timeout_seconds=0.05,  # 50ms timeout triggers quickly
            raise_on_error=False,
        )

        results = await retriever.asearch("timeout test query", top_k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_timeout_raises_when_raise_on_error_true(self) -> None:
        """When Qdrant times out with raise_on_error=True, raise VectorStoreTimeoutError."""
        mock_embedder = MagicMock()
        mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 1024)

        async def slow_query(*args: object, **kwargs: object) -> QueryResponse:
            await asyncio.sleep(0.5)
            return QueryResponse(points=[])

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(side_effect=slow_query)

        retriever = DenseRetriever(
            client=mock_client,
            embedder=mock_embedder,
            timeout_seconds=0.05,
            raise_on_error=True,
        )

        with pytest.raises(VectorStoreTimeoutError) as exc_info:
            await retriever.asearch("failing timeout query", top_k=10)
        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connection_error_fallback_degraded_mode(self) -> None:
        """When Qdrant raises ResponseHandlingException, return empty list in degraded mode."""
        mock_embedder = MagicMock()
        mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 1024)

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            side_effect=ResponseHandlingException("Host unreachable")
        )

        retriever = DenseRetriever(
            client=mock_client,
            embedder=mock_embedder,
            raise_on_error=False,
        )

        results = await retriever.asearch("connection failure query")
        assert results == []

    @pytest.mark.asyncio
    async def test_connection_error_raises_when_requested(self) -> None:
        """When Qdrant fails with raise_on_error=True, raise VectorStoreTimeoutError."""
        mock_embedder = MagicMock()
        mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 1024)

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            side_effect=ResponseHandlingException("Host unreachable")
        )

        retriever = DenseRetriever(
            client=mock_client,
            embedder=mock_embedder,
            raise_on_error=True,
        )

        with pytest.raises(VectorStoreTimeoutError) as exc_info:
            await retriever.asearch("connection failure query")
        assert "failed" in str(exc_info.value)


class TestSparseRetriever:
    """Unit tests for SparseRetriever."""

    @pytest.mark.asyncio
    async def test_sparse_asearch_success(self) -> None:
        """Sparse search calls BM25StoreManager.asearch and returns ranked candidates."""
        mock_chunk = RetrievedChunk(
            chunk_id="point_001",
            source_file="docs/postgres.md",
            content="PostgreSQL connection pooling with PgBouncer",
            score=0.82,
            metadata={"bm25_rank": 1},
        )

        mock_store = MagicMock()
        mock_store.chunk_count = 50
        mock_store.asearch = AsyncMock(return_value=[mock_chunk])

        retriever = SparseRetriever(bm25_store=mock_store, auto_load=False)
        results = await retriever.asearch("connection pooling", top_k=10)

        mock_store.asearch.assert_awaited_once_with(query="connection pooling", top_k=10)
        assert len(results) == 1
        assert results[0].chunk_id == "point_001"
        assert results[0].score == 0.82

    def test_sparse_search_synchronous(self) -> None:
        """Synchronous search calls BM25StoreManager.search."""
        mock_chunk = RetrievedChunk(
            chunk_id="point_sync",
            source_file="docs/linux.md",
            content="Linux cgroups configuration",
            score=0.75,
            metadata={"bm25_rank": 1},
        )

        mock_store = MagicMock()
        mock_store.chunk_count = 10
        mock_store.search.return_value = [mock_chunk]

        retriever = SparseRetriever(bm25_store=mock_store, auto_load=False)
        results = retriever.search("cgroups configuration", top_k=5)

        mock_store.search.assert_called_once_with(query="cgroups configuration", top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "point_sync"

    @pytest.mark.asyncio
    async def test_sparse_validation_and_empty_query(self) -> None:
        """Invalid top_k raises ValueError; whitespace query returns empty list."""
        retriever = SparseRetriever(auto_load=False)

        with pytest.raises(ValueError):
            await retriever.asearch("query", top_k=0)

        with pytest.raises(ValueError):
            retriever.search("query", top_k=-1)

        assert await retriever.asearch("") == []
        assert retriever.search("   ") == []

    @pytest.mark.asyncio
    async def test_sparse_auto_load_triggered_when_empty(self) -> None:
        """When store chunk_count is 0 and auto_load=True, calls store.load()."""
        mock_store = MagicMock()
        mock_store.chunk_count = 0
        mock_store.load.return_value = True
        mock_store.asearch = AsyncMock(return_value=[])

        retriever = SparseRetriever(bm25_store=mock_store, auto_load=True)
        await retriever.asearch("load query", top_k=5)

        mock_store.load.assert_called_once()


class TestDegradedFallbackIntegration:
    """Integration test verifying retrieval resiliency under degraded vector store."""

    @pytest.mark.asyncio
    async def test_degraded_pipeline_proceeds_with_bm25_when_dense_fails(self) -> None:
        """On DenseRetriever timeout, pipeline proceeds with valid sparse results."""
        # Setup failing DenseRetriever
        mock_embedder = MagicMock()
        mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 1024)

        async def slow_points(*args: object, **kwargs: object) -> QueryResponse:
            await asyncio.sleep(0.2)
            return QueryResponse(points=[])

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(side_effect=slow_points)

        dense_retriever = DenseRetriever(
            client=mock_client,
            embedder=mock_embedder,
            timeout_seconds=0.02,  # 20ms timeout
            raise_on_error=False,  # Degraded mode
        )

        # Setup working SparseRetriever
        sparse_chunk = RetrievedChunk(
            chunk_id="bm25_point_01",
            source_file="docs/sample_persian_docker_guide.md",
            content="راهنمای جامع داکر و مدیریت کانتینرها",
            score=0.79,
            metadata={"bm25_rank": 1},
        )
        mock_store = MagicMock()
        mock_store.chunk_count = 10
        mock_store.asearch = AsyncMock(return_value=[sparse_chunk])

        sparse_retriever = SparseRetriever(bm25_store=mock_store, auto_load=False)

        # Run both concurrently, simulating hybrid retrieval coordinator
        query = "دستورات کاربردی داکر"
        dense_results, sparse_results = await asyncio.gather(
            dense_retriever.asearch(query, top_k=20),
            sparse_retriever.asearch(query, top_k=20),
        )

        # Dense failed safely into degraded mode (0 results), while Sparse succeeded
        assert dense_results == []
        assert len(sparse_results) == 1
        assert sparse_results[0].chunk_id == "bm25_point_01"
        assert "داکر" in sparse_results[0].content
