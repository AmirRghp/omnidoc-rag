"""Unit tests for BGEReranker / CrossEncoderReranker."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.common.schemas import RetrievedChunk
from src.retrieval.reranker import BGEReranker, CrossEncoderReranker


def _create_sample_chunk(
    chunk_id: str,
    content: str,
    score: float = 0.5,
    source_file: str = "docs/docker.md",
) -> RetrievedChunk:
    """Helper to create a sample RetrievedChunk."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_file=source_file,
        content=content,
        score=score,
        metadata={"original_rank": 1},
    )


class TestBGERerankerInitialization:
    """Tests for reranker initialization and model wiring."""

    def test_default_initialization(self) -> None:
        """Default initialization configures model name and auto device."""
        reranker = BGEReranker()
        assert reranker.model_name == "BAAI/bge-reranker-v2-m3" or "reranker" in reranker.model_name
        assert reranker.device in {"cuda", "cpu"}
        assert reranker.top_n == 5
        assert reranker.batch_size == 32
        assert reranker._model is None

    def test_custom_initialization_and_injection(self) -> None:
        """Custom parameters and injected mock model are preserved."""
        mock_model = MagicMock()
        reranker = BGEReranker(
            model_name="custom/reranker-model",
            device="cpu",
            batch_size=16,
            top_n=3,
            model=mock_model,
        )

        assert reranker.model_name == "custom/reranker-model"
        assert reranker.device == "cpu"
        assert reranker.batch_size == 16
        assert reranker.top_n == 3
        assert reranker.model is mock_model

    def test_lazy_model_loading_triggers_instantiation(self) -> None:
        """Accessing model property lazily instantiates CrossEncoder."""
        reranker = BGEReranker(device="cpu")
        assert reranker._model is None

        with patch("src.retrieval.reranker.CrossEncoder") as mock_ce_cls:
            mock_ce_instance = MagicMock()
            mock_ce_cls.return_value = mock_ce_instance

            model = reranker.model
            assert model is mock_ce_instance
            mock_ce_cls.assert_called_once_with(
                model_name_or_path=reranker.model_name,
                device="cpu",
            )


class TestBGERerankerRerank:
    """Tests for rerank scoring, sigmoid calibration, and sorting."""

    def test_rerank_sigmoid_calibration_and_sorting(self) -> None:
        """Raw logits are calibrated via sigmoid into [0.0, 1.0] and sorted descending."""
        chunks = [
            _create_sample_chunk("c1", "Low relevance chunk"),
            _create_sample_chunk("c2", "High relevance chunk"),
            _create_sample_chunk("c3", "Neutral relevance chunk"),
        ]

        # Raw logits: c1 -> -2.0, c2 -> 3.0, c3 -> 0.0
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-2.0, 3.0, 0.0])

        reranker = BGEReranker(model=mock_model)
        rescored = reranker.rerank("docker container network", chunks, top_n=3)

        mock_model.predict.assert_called_once_with(
            [
                ("docker container network", "Low relevance chunk"),
                ("docker container network", "High relevance chunk"),
                ("docker container network", "Neutral relevance chunk"),
            ],
            batch_size=32,
        )

        assert len(rescored) == 3
        # Expected sigmoid scores:
        # c2 (logit 3.0) -> 1 / (1 + exp(-3.0)) ≈ 0.952574
        # c3 (logit 0.0) -> 1 / (1 + exp(0.0)) = 0.5
        # c1 (logit -2.0) -> 1 / (1 + exp(2.0)) ≈ 0.119203
        assert rescored[0].chunk_id == "c2"
        assert pytest.approx(rescored[0].score, rel=1e-4) == 0.952574
        assert rescored[0].metadata["rerank_rank"] == 1
        assert rescored[0].metadata["raw_rerank_score"] == 3.0
        assert pytest.approx(rescored[0].metadata["rerank_score"], rel=1e-4) == 0.952574

        assert rescored[1].chunk_id == "c3"
        assert pytest.approx(rescored[1].score, rel=1e-4) == 0.5
        assert rescored[1].metadata["rerank_rank"] == 2

        assert rescored[2].chunk_id == "c1"
        assert pytest.approx(rescored[2].score, rel=1e-4) == 0.119203
        assert rescored[2].metadata["rerank_rank"] == 3

    def test_top_n_truncation(self) -> None:
        """Rerank strictly slices output to top_n candidates."""
        chunks = [_create_sample_chunk(f"c{i}", f"content {i}") for i in range(10)]
        logits = np.linspace(-5.0, 5.0, 10)

        mock_model = MagicMock()
        mock_model.predict.return_value = logits

        reranker = BGEReranker(model=mock_model)
        rescored = reranker.rerank("query", chunks, top_n=4)

        assert len(rescored) == 4
        assert rescored[0].metadata["rerank_rank"] == 1
        assert rescored[-1].metadata["rerank_rank"] == 4

    def test_single_scalar_logit_handled_cleanly(self) -> None:
        """Single chunk producing a 0-d scalar logit array is handled safely."""
        chunk = _create_sample_chunk("single_c", "Single content")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.float64(2.5)

        reranker = BGEReranker(model=mock_model)
        rescored = reranker.rerank("query", [chunk])

        assert len(rescored) == 1
        assert rescored[0].chunk_id == "single_c"
        assert 0.0 <= rescored[0].score <= 1.0


class TestValidationAndEdgeCases:
    """Tests input validation and empty edge cases."""

    def test_empty_chunks_returns_empty(self) -> None:
        """Empty candidates list returns empty list without calling model."""
        mock_model = MagicMock()
        reranker = BGEReranker(model=mock_model)

        assert reranker.rerank("query", []) == []
        mock_model.predict.assert_not_called()

    def test_empty_or_whitespace_query_returns_empty(self) -> None:
        """Empty or whitespace query returns empty list without calling model."""
        mock_model = MagicMock()
        reranker = BGEReranker(model=mock_model)
        chunks = [_create_sample_chunk("c1", "content")]

        assert reranker.rerank("", chunks) == []
        assert reranker.rerank("   ", chunks) == []
        mock_model.predict.assert_not_called()

    def test_invalid_top_n_raises_value_error(self) -> None:
        """Non-positive top_n raises ValueError."""
        reranker = BGEReranker()
        chunks = [_create_sample_chunk("c1", "content")]

        with pytest.raises(ValueError) as exc:
            reranker.rerank("query", chunks, top_n=0)
        assert "positive" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            reranker.rerank("query", chunks, top_n=-3)
        assert "positive" in str(exc.value)


class TestAsyncRerank:
    """Tests for async arerank wrapper offloading to thread."""

    @pytest.mark.asyncio
    async def test_arerank_executes_in_threadpool(self) -> None:
        """arerank executes asynchronously without blocking the event loop."""
        chunks = [
            _create_sample_chunk("c1", "PostgreSQL tuning"),
            _create_sample_chunk("c2", "Linux kernel flags"),
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.5, -0.5])

        reranker = BGEReranker(model=mock_model)
        rescored = await reranker.arerank("database performance", chunks, top_n=2)

        assert len(rescored) == 2
        assert rescored[0].chunk_id == "c1"
        assert rescored[1].chunk_id == "c2"
        assert rescored[0].score > rescored[1].score


class TestAliasConformity:
    """Tests architectural alias naming."""

    def test_cross_encoder_reranker_alias(self) -> None:
        """CrossEncoderReranker is an alias of BGEReranker."""
        assert CrossEncoderReranker is BGEReranker
