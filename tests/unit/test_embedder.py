"""Unit tests for BGEEmbedder wrapper using mocked weights for rapid execution."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.common.config import Settings
from src.ingestion.embedder import BGEEmbedder


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing isolated Settings."""
    return Settings(
        dense_model_name="BAAI/bge-m3",
        embedding_dimension=1024,
    )


class TestBGEEmbedderInitialization:
    """Test embedder instantiation and configuration."""

    def test_initialization_defaults(self, mock_settings: Settings) -> None:
        embedder = BGEEmbedder(settings=mock_settings)
        assert embedder.model_name == "BAAI/bge-m3"
        assert embedder.expected_dimension == 1024
        assert embedder.batch_size == 32
        assert embedder.normalize_embeddings is True
        assert embedder._model is None

    def test_explicit_device_configuration(self, mock_settings: Settings) -> None:
        embedder = BGEEmbedder(device="cpu", settings=mock_settings)
        assert embedder.device == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    def test_automatic_device_cuda_when_available(
        self, mock_cuda: Any, mock_settings: Settings
    ) -> None:
        embedder = BGEEmbedder(settings=mock_settings)
        assert embedder.device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_automatic_device_cpu_when_cuda_unavailable(
        self, mock_cuda: Any, mock_settings: Settings
    ) -> None:
        embedder = BGEEmbedder(settings=mock_settings)
        assert embedder.device == "cpu"


class TestBGEEmbedderInference:
    """Test embedding calculation with mocked SentenceTransformer."""

    @patch("src.ingestion.embedder.SentenceTransformer")
    def test_lazy_loading(
        self, mock_st_cls: MagicMock, mock_settings: Settings
    ) -> None:
        embedder = BGEEmbedder(device="cpu", settings=mock_settings)
        # Model not instantiated on __init__
        assert embedder._model is None
        mock_st_cls.assert_not_called()

        # Accessing property triggers instantiation
        _ = embedder.model
        assert embedder._model is not None
        mock_st_cls.assert_called_once_with(
            model_name_or_path="BAAI/bge-m3",
            device="cpu",
        )

    @patch("src.ingestion.embedder.SentenceTransformer")
    def test_embed_documents_batch(
        self, mock_st_cls: MagicMock, mock_settings: Settings
    ) -> None:
        mock_instance = MagicMock()
        mock_st_cls.return_value = mock_instance

        # Return mock 2x1024 array
        fake_vectors = np.ones((2, 1024), dtype=np.float32)
        mock_instance.encode.return_value = fake_vectors

        embedder = BGEEmbedder(device="cpu", settings=mock_settings)
        docs = ["Docker container networking", "راهنمای اتصال به دیتابیس"]
        result = embedder.embed_documents(docs)

        mock_instance.encode.assert_called_once_with(
            sentences=docs,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        assert len(result) == 2
        assert len(result[0]) == 1024
        assert len(result[1]) == 1024
        assert isinstance(result[0][0], float)

    def test_embed_documents_empty_returns_empty(
        self, mock_settings: Settings
    ) -> None:
        embedder = BGEEmbedder(settings=mock_settings)
        assert embedder.embed_documents([]) == []

    @patch("src.ingestion.embedder.SentenceTransformer")
    def test_embed_query_success(
        self, mock_st_cls: MagicMock, mock_settings: Settings
    ) -> None:
        mock_instance = MagicMock()
        mock_st_cls.return_value = mock_instance

        fake_vector = np.full((1, 1024), 0.5, dtype=np.float32)
        mock_instance.encode.return_value = fake_vector

        embedder = BGEEmbedder(device="cpu", settings=mock_settings)
        vec = embedder.embed_query("چگونه پورت را مپ کنیم؟")

        assert len(vec) == 1024
        assert vec[0] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_aembed_query_async(
        self, mock_settings: Settings
    ) -> None:
        with patch.object(
            BGEEmbedder, "embed_query", return_value=[0.1] * 1024
        ) as mock_sync:
            embedder = BGEEmbedder(device="cpu", settings=mock_settings)
            res = await embedder.aembed_query("async test query")
            mock_sync.assert_called_once_with("async test query")
            assert len(res) == 1024

    @pytest.mark.asyncio
    async def test_aembed_documents_async(
        self, mock_settings: Settings
    ) -> None:
        with patch.object(
            BGEEmbedder, "embed_documents", return_value=[[0.2] * 1024]
        ) as mock_sync:
            embedder = BGEEmbedder(device="cpu", settings=mock_settings)
            res = await embedder.aembed_documents(["async doc"])
            mock_sync.assert_called_once_with(["async doc"])
            assert len(res) == 1
            assert len(res[0]) == 1024


class TestBGEEmbedderValidation:
    """Test error handling and boundary conditions."""

    def test_embed_query_empty_raises_value_error(
        self, mock_settings: Settings
    ) -> None:
        embedder = BGEEmbedder(settings=mock_settings)
        with pytest.raises(ValueError, match="cannot be empty"):
            embedder.embed_query("   ")

    def test_embed_query_non_string_raises_type_error(
        self, mock_settings: Settings
    ) -> None:
        embedder = BGEEmbedder(settings=mock_settings)
        with pytest.raises(TypeError, match="must be a string"):
            embedder.embed_query(123)  # type: ignore[arg-type]

    def test_embed_documents_non_list_raises_type_error(
        self, mock_settings: Settings
    ) -> None:
        embedder = BGEEmbedder(settings=mock_settings)
        with pytest.raises(TypeError, match="must be a list of string"):
            embedder.embed_documents("not a list")  # type: ignore[arg-type]

    def test_embed_documents_non_string_element_raises_type_error(
        self, mock_settings: Settings
    ) -> None:
        embedder = BGEEmbedder(settings=mock_settings)
        with pytest.raises(TypeError, match="must be a list of string"):
            embedder.embed_documents(["valid", 123])  # type: ignore[list-item]

    @patch("src.ingestion.embedder.SentenceTransformer")
    def test_dimensional_mismatch_raises_value_error(
        self, mock_st_cls: MagicMock, mock_settings: Settings
    ) -> None:
        mock_instance = MagicMock()
        mock_st_cls.return_value = mock_instance

        # Return vector with incorrect dimension (e.g., 768 instead of 1024)
        wrong_vectors = np.ones((1, 768), dtype=np.float32)
        mock_instance.encode.return_value = wrong_vectors

        embedder = BGEEmbedder(device="cpu", settings=mock_settings)
        with pytest.raises(ValueError, match="does not match expected dimension 1024"):
            embedder.embed_documents(["some doc"])
