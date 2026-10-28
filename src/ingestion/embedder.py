"""Local multilingual embedding generator wrapping BAAI/bge-m3."""

import asyncio
from typing import Any, cast

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.common.config import Settings, create_settings
from src.common.logger import get_logger

logger = get_logger(__name__)


class BGEEmbedder:
    """Multilingual dense embedding generator using BAAI/bge-m3."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the BGEEmbedder configuration.

        Model loading is deferred to the first embedding call (lazy loading).
        """
        app_settings = settings or create_settings()
        self.model_name = model_name or app_settings.dense_model_name
        self.expected_dimension = app_settings.embedding_dimension
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Get or lazily instantiate the SentenceTransformer model."""
        if self._model is None:
            logger.info(
                "loading_embedding_model",
                model_name=self.model_name,
                device=self.device,
            )
            self._model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute normalized dense embeddings for a list of document strings.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of 1024-dimensional float embedding vectors.
        """
        if not texts:
            return []

        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be a list of string documents")

        model_any = cast(Any, self.model)
        embeddings: Any = model_any.encode(
            sentences=texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        # Validate dimensional integrity
        if embeddings_np.ndim != 2 or embeddings_np.shape[1] != self.expected_dimension:
            actual_dim = embeddings_np.shape[1] if embeddings_np.ndim > 1 else "invalid"
            raise ValueError(
                f"Generated embedding dimension {actual_dim} "
                f"does not match expected dimension {self.expected_dimension}."
            )

        return embeddings_np.tolist()  # type: ignore[no-any-return]

    def embed_query(self, text: str) -> list[float]:
        """Compute normalized dense embedding for a single query string.

        Args:
            text: Single query string.

        Returns:
            A single 1024-dimensional float embedding vector.
        """
        if not isinstance(text, str):
            raise TypeError("text query must be a string")

        if not text.strip():
            raise ValueError("query text cannot be empty or whitespace only")

        embeddings = self.embed_documents([text])
        return embeddings[0]

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously compute dense embedding for a query via asyncio.to_thread.

        Prevents blocking the asyncio event loop during model inference.
        """
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously compute dense embeddings for document texts via asyncio.to_thread.

        Prevents blocking the asyncio event loop during model inference.
        """
        return await asyncio.to_thread(self.embed_documents, texts)
