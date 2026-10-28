"""Unit tests for QdrantStoreManager with mocked client for rapid verification."""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.common.config import Settings
from src.common.exceptions import VectorStoreTimeoutError
from src.common.schemas import DocumentChunkPayload
from src.ingestion.qdrant_store import QdrantStoreManager


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing isolated Settings."""
    return Settings(
        collection_name="test_tech_docs",
        embedding_dimension=1024,
    )


@pytest.fixture
def sample_chunk_payload() -> DocumentChunkPayload:
    """Fixture providing a sample DocumentChunkPayload."""
    return DocumentChunkPayload(
        doc_id="docker_guide_01",
        chunk_id=0,
        content="Docker daemon socket explanation and instructions.",
        source_file="manuals/docker_daemon.md",
        section_heading="## Daemon socket option",
        language="en",
        token_count=12,
    )


class TestQdrantStoreManagerInitialization:
    """Test QdrantStoreManager initialization and properties."""

    def test_initialization_defaults(self, mock_settings: Settings) -> None:
        manager = QdrantStoreManager(settings=mock_settings)
        assert manager.collection_name == "test_tech_docs"
        assert manager.embedding_dimension == 1024
        assert manager._client is None
        assert manager._async_client is None

    def test_custom_client_injection(self, mock_settings: Settings) -> None:
        mock_client = MagicMock()
        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        assert manager.client is mock_client


class TestEnsureCollectionExists:
    """Test collection creation and payload indexing logic."""

    def test_creates_collection_when_not_exists(
        self, mock_settings: Settings
    ) -> None:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        created = manager.ensure_collection_exists()

        assert created is True
        mock_client.collection_exists.assert_called_once_with("test_tech_docs")

        # Verify create_collection arguments
        mock_client.create_collection.assert_called_once()
        _, kwargs = mock_client.create_collection.call_args
        assert kwargs["collection_name"] == "test_tech_docs"
        assert kwargs["vectors_config"].size == 1024
        assert kwargs["vectors_config"].distance == models.Distance.COSINE
        assert kwargs["hnsw_config"].m == 16
        assert kwargs["hnsw_config"].ef_construct == 100
        assert kwargs["optimizers_config"].default_segment_number == 2

        # Verify payload indexes creation
        assert mock_client.create_payload_index.call_count == 3
        calls = mock_client.create_payload_index.call_args_list
        indexed_fields = [call.kwargs["field_name"] for call in calls]
        assert indexed_fields == ["doc_id", "source_file", "language"]

    def test_skips_creation_when_already_exists(
        self, mock_settings: Settings
    ) -> None:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True

        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        created = manager.ensure_collection_exists()

        assert created is False
        mock_client.collection_exists.assert_called_once_with("test_tech_docs")
        mock_client.create_collection.assert_not_called()
        mock_client.create_payload_index.assert_not_called()

    def test_timeout_raises_vector_store_timeout_error(
        self, mock_settings: Settings
    ) -> None:
        mock_client = MagicMock()
        mock_client.collection_exists.side_effect = TimeoutError("Connection timed out")

        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        with pytest.raises(VectorStoreTimeoutError, match="Timed out"):
            manager.ensure_collection_exists()


class TestUpsertChunks:
    """Test idempotent chunk upsert and batching."""

    def test_upsert_empty_list_returns_zero(
        self, mock_settings: Settings
    ) -> None:
        mock_client = MagicMock()
        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        count = manager.upsert_chunks([], [])
        assert count == 0
        mock_client.upsert.assert_not_called()

    def test_upsert_mismatched_lengths_raises_value_error(
        self, mock_settings: Settings, sample_chunk_payload: DocumentChunkPayload
    ) -> None:
        mock_client = MagicMock()
        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        with pytest.raises(ValueError, match="Mismatch"):
            manager.upsert_chunks([sample_chunk_payload], [])

    def test_upsert_invalid_batch_size_raises_value_error(
        self, mock_settings: Settings, sample_chunk_payload: DocumentChunkPayload
    ) -> None:
        mock_client = MagicMock()
        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            manager.upsert_chunks([sample_chunk_payload], [[0.1] * 1024], batch_size=0)

    def test_upsert_invalid_vector_dimension_raises_value_error(
        self, mock_settings: Settings, sample_chunk_payload: DocumentChunkPayload
    ) -> None:
        mock_client = MagicMock()
        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        with pytest.raises(ValueError, match="does not match expected 1024"):
            manager.upsert_chunks([sample_chunk_payload], [[0.1] * 512])

    def test_upsert_chunks_success_and_preserves_deterministic_point_id(
        self, mock_settings: Settings, sample_chunk_payload: DocumentChunkPayload
    ) -> None:
        mock_client = MagicMock()
        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)

        chunks = [sample_chunk_payload]
        vectors = [[0.5] * 1024]

        count = manager.upsert_chunks(chunks, vectors, batch_size=64)
        assert count == 1
        mock_client.upsert.assert_called_once()
        _, kwargs = mock_client.upsert.call_args
        assert kwargs["collection_name"] == "test_tech_docs"
        points = kwargs["points"]
        assert len(points) == 1
        assert points[0].id == sample_chunk_payload.point_id
        assert points[0].vector == vectors[0]
        assert points[0].payload["doc_id"] == "docker_guide_01"

    def test_upsert_batching_multiple_chunks(
        self, mock_settings: Settings, sample_chunk_payload: DocumentChunkPayload
    ) -> None:
        mock_client = MagicMock()
        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)

        chunk2 = sample_chunk_payload.model_copy(update={"chunk_id": 1})
        chunk3 = sample_chunk_payload.model_copy(update={"chunk_id": 2})

        chunks = [sample_chunk_payload, chunk2, chunk3]
        vectors = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]

        count = manager.upsert_chunks(chunks, vectors, batch_size=2)
        assert count == 3
        # 3 items with batch_size=2 results in 2 upsert calls
        assert mock_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_aupsert_chunks_async(
        self, mock_settings: Settings, sample_chunk_payload: DocumentChunkPayload
    ) -> None:
        manager = QdrantStoreManager(settings=mock_settings)
        with patch.object(manager, "upsert_chunks", return_value=1) as mock_sync:
            result = await manager.aupsert_chunks(
                [sample_chunk_payload], [[0.1] * 1024], batch_size=32
            )
            mock_sync.assert_called_once_with(
                [sample_chunk_payload], [[0.1] * 1024], 32
            )
            assert result == 1


class TestCollectionInfoAndDelete:
    """Test collection info retrieval and point deletion."""

    def test_get_collection_info(self, mock_settings: Settings) -> None:
        mock_client = MagicMock()
        fake_info = MagicMock()
        fake_info.points_count = 42
        fake_info.indexed_vectors_count = 42
        fake_info.status = models.CollectionStatus.GREEN
        mock_client.get_collection.return_value = fake_info

        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        info = manager.get_collection_info()

        assert info["points_count"] == 42
        assert info["indexed_vectors_count"] == 42
        assert "green" in info["status"].lower()

    def test_delete_by_doc_id_success(self, mock_settings: Settings) -> None:
        mock_client = MagicMock()
        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        manager.delete_by_doc_id("docker_guide_01")

        mock_client.delete.assert_called_once()
        _, kwargs = mock_client.delete.call_args
        assert kwargs["collection_name"] == "test_tech_docs"

    def test_delete_by_empty_doc_id_raises_value_error(
        self, mock_settings: Settings
    ) -> None:
        manager = QdrantStoreManager(settings=mock_settings)
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.delete_by_doc_id("   ")

    def test_qdrant_unexpected_response_raises_vector_store_timeout_error(
        self, mock_settings: Settings
    ) -> None:
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = UnexpectedResponse(
            500, "Internal Server Error", "error", None
        )
        manager = QdrantStoreManager(client=mock_client, settings=mock_settings)
        with pytest.raises(VectorStoreTimeoutError, match="Qdrant error"):
            manager.get_collection_info()
