"""Bilingual Cross-Encoder Reranker wrapping BAAI/bge-reranker-v2-m3."""

import asyncio
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from src.common.config import Settings, create_settings
from src.common.logger import get_logger
from src.common.schemas import RetrievedChunk

logger = get_logger(__name__)

DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

__all__ = [
    "BGEReranker",
    "CrossEncoderReranker",
]


class BGEReranker:
    """Bilingual Cross-Encoder reranker using BAAI/bge-reranker-v2-m3.

    Rescores candidate query-passage pairs jointly ([CLS] Query [SEP] Chunk [SEP])
    and calibrates unbounded logits into [0.0, 1.0] using sigmoid activation.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        top_n: int = 5,
        settings: Settings | None = None,
        model: CrossEncoder | None = None,
    ) -> None:
        """Initialize BGEReranker.

        Args:
            model_name: HuggingFace model name/path or None for default.
            device: Computing device ('cuda', 'cpu') or None for auto-detection.
            batch_size: Inference batch size.
            top_n: Number of top-ranked candidates to retain after reranking.
            settings: Optional application settings.
            model: Optional pre-instantiated CrossEncoder instance (for testing).
        """
        app_settings = settings or create_settings()
        self.model_name = model_name or app_settings.rerank_model_name or DEFAULT_RERANKER_MODEL
        self.batch_size = batch_size
        self.top_n = top_n

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = model

    @property
    def model(self) -> CrossEncoder:
        """Get or lazily instantiate the CrossEncoder model."""
        if self._model is None:
            logger.info(
                "loading_reranker_model",
                model_name=self.model_name,
                device=self.device,
            )
            self._model = CrossEncoder(
                model_name_or_path=self.model_name,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """Synchronously rescore and reorder retrieved chunks using the Cross-Encoder.

        Args:
            query: User search query string.
            chunks: Candidate chunks to rescore (typically Top-15 from RRF).
            top_n: Optional override for the number of top candidates to return.

        Returns:
            Top-N rescored RetrievedChunk objects sorted descending by relevance score.

        Raises:
            ValueError: If top_n is non-positive.
        """
        effective_top_n = top_n if top_n is not None else self.top_n
        if effective_top_n <= 0:
            raise ValueError(f"top_n must be positive, got {effective_top_n}")

        normalized_query = query.strip()
        if not normalized_query or not chunks:
            return []

        # 1. Prepare joint query-passage sequence pairs
        pairs = [(normalized_query, chunk.content) for chunk in chunks]

        # 2. Run joint Cross-Encoder inference
        raw_logits = self.model.predict(pairs, batch_size=self.batch_size)

        # Ensure raw_logits is an array
        logits_arr = np.asarray(raw_logits, dtype=np.float64)
        if logits_arr.ndim == 0:
            logits_arr = np.array([float(logits_arr)])

        # 3. Apply sigmoid calibration: sigma(z) = 1 / (1 + exp(-z))
        calibrated_probs = 1.0 / (1.0 + np.exp(-logits_arr))

        # 4. Pair chunks with scores and ranks
        scored_chunks: list[tuple[float, float, RetrievedChunk]] = []
        for idx, chunk in enumerate(chunks):
            raw_logit = float(logits_arr[idx])
            calibrated_score = max(0.0, min(1.0, float(calibrated_probs[idx])))
            scored_chunks.append((calibrated_score, raw_logit, chunk))

        # 5. Sort descending by calibrated score
        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        # 6. Sift Top-N candidates and update metadata
        top_candidates = scored_chunks[:effective_top_n]
        results: list[RetrievedChunk] = []

        for rank, (score, raw_score, original_chunk) in enumerate(top_candidates, start=1):
            updated_metadata: dict[str, Any] = {
                **original_chunk.metadata,
                "rerank_score": score,
                "raw_rerank_score": raw_score,
                "rerank_rank": rank,
            }
            results.append(
                RetrievedChunk(
                    chunk_id=original_chunk.chunk_id,
                    source_file=original_chunk.source_file,
                    content=original_chunk.content,
                    score=score,
                    metadata=updated_metadata,
                )
            )

        return results

    async def arerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """Asynchronously rescore candidates by offloading CPU-bound inference to threadpool.

        Guarantees non-blocking execution inside the asyncio event loop.
        """
        return await asyncio.to_thread(self.rerank, query, chunks, top_n)


# Alias conforming to architecture specification
CrossEncoderReranker = BGEReranker
