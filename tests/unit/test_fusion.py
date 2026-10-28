"""Unit tests for Reciprocal Rank Fusion (RRF) engine."""

import time

import pytest

from src.common.schemas import RetrievedChunk
from src.retrieval.fusion import (
    afuse_dense_and_sparse,
    fuse_dense_and_sparse,
    reciprocal_rank_fusion,
)


def _create_sample_chunk(
    chunk_id: str,
    doc_id: str,
    content: str,
    score: float = 0.8,
    source_file: str = "docs/sample.md",
) -> RetrievedChunk:
    """Helper to create a typed RetrievedChunk."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_file=source_file,
        content=content,
        score=score,
        metadata={"doc_id": doc_id},
    )


class TestRRFFusionScoringOrder:
    """Tests confirming RRF scoring order conforming to docs/TESTING.md Section 3.1."""

    def test_rrf_scoring_order(self) -> None:
        """doc_2 present in both lists must rank #1 with highest RRF score."""
        list_a = [{"doc_id": "doc_1", "content": "a"}, {"doc_id": "doc_2", "content": "b"}]
        list_b = [{"doc_id": "doc_2", "content": "b"}, {"doc_id": "doc_3", "content": "c"}]

        # doc_2 is present in both lists and must rank #1
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)

        assert fused[0]["doc_id"] == "doc_2"
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]
        assert len(fused) == 3


class TestRRFMathematicalCorrectness:
    """Tests for exact mathematical formula computation."""

    def test_exact_rrf_score_calculation(self) -> None:
        """Verify exact calculation of 1 / (k + rank) across channels."""
        # doc_a: rank 1 in list_a (1/(60+1)), rank 2 in list_b (1/(60+2))
        # doc_b: rank 2 in list_a (1/(60+2)), rank 1 in list_b (1/(60+1))
        # doc_c: rank 3 in list_a (1/(60+3)), not in list_b
        list_a = [{"doc_id": "doc_a"}, {"doc_id": "doc_b"}, {"doc_id": "doc_c"}]
        list_b = [{"doc_id": "doc_b"}, {"doc_id": "doc_a"}]

        fused = reciprocal_rank_fusion([list_a, list_b], k=60)

        expected_score_ab = (1.0 / 61) + (1.0 / 62)
        expected_score_c = 1.0 / 63

        assert pytest.approx(fused[0].score, rel=1e-6) == expected_score_ab
        assert pytest.approx(fused[1].score, rel=1e-6) == expected_score_ab
        assert pytest.approx(fused[2].score, rel=1e-6) == expected_score_c

        assert fused[0].metadata["rrf_rank"] == 1
        assert fused[1].metadata["rrf_rank"] == 2
        assert fused[2].metadata["rrf_rank"] == 3


class TestRRFValidationAndEdgeCases:
    """Tests parameter validation and edge cases."""

    def test_invalid_parameters_raise_value_error(self) -> None:
        """k <= 0 or top_n <= 0 must raise ValueError."""
        with pytest.raises(ValueError) as exc:
            reciprocal_rank_fusion([], k=0)
        assert "positive" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            reciprocal_rank_fusion([], top_n=0)
        assert "positive" in str(exc.value)

    def test_unsupported_item_type_raises_type_error(self) -> None:
        """Passing an object that is neither RetrievedChunk nor dict raises TypeError."""
        with pytest.raises(TypeError) as exc:
            reciprocal_rank_fusion([["invalid_string"]])  # type: ignore[list-item]
        assert "Unsupported item type" in str(exc.value)

    def test_empty_rankings_returns_empty_list(self) -> None:
        """Empty rankings or empty sub-lists return empty candidate list."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_active_channel_degraded_fallback(self) -> None:
        """When dense channel is empty (degraded), sparse ranking order is preserved."""
        sparse_list = [
            _create_sample_chunk("s1", "doc_s1", "content 1", score=0.9),
            _create_sample_chunk("s2", "doc_s2", "content 2", score=0.7),
        ]

        fused = fuse_dense_and_sparse(dense_results=[], sparse_results=sparse_list, k=60)
        assert len(fused) == 2
        assert fused[0].chunk_id == "s1"
        assert fused[1].chunk_id == "s2"
        assert fused[0].score > fused[1].score
        assert fused[0].metadata["source_ranks"] == {"sparse": 1}

    def test_deduplication_within_same_channel(self) -> None:
        """Duplicate item appearing twice in the same list is counted only at its best rank."""
        list_with_dupes = [
            {"chunk_id": "c1", "content": "repeat"},
            {"chunk_id": "c1", "content": "repeat lower rank"},
        ]

        fused = reciprocal_rank_fusion([list_with_dupes], k=60)
        assert len(fused) == 1
        assert pytest.approx(fused[0].score, rel=1e-6) == (1.0 / 61)

    def test_top_n_truncation(self) -> None:
        """Results are strictly truncated to top_n parameter."""
        items = [{"chunk_id": f"chunk_{i}", "content": f"text {i}"} for i in range(30)]
        fused = reciprocal_rank_fusion([items], k=60, top_n=15)
        assert len(fused) == 15
        assert fused[0].metadata["rrf_rank"] == 1
        assert fused[-1].metadata["rrf_rank"] == 15

    def test_deterministic_tie_breaking(self) -> None:
        """Identical scores break ties deterministically by ID."""
        list_a = [{"chunk_id": "chunk_z"}, {"chunk_id": "chunk_a"}]
        # In list_a, chunk_z is rank 1, chunk_a is rank 2
        fused = reciprocal_rank_fusion([list_a], k=60)
        assert fused[0].chunk_id == "chunk_z"
        assert fused[1].chunk_id == "chunk_a"


class TestFuseDenseAndSparse:
    """Tests for the typed fuse_dense_and_sparse wrapper."""

    def test_fuse_dense_and_sparse_labels_channel_ranks(self) -> None:
        """fuse_dense_and_sparse sets source_ranks with 'dense' and 'sparse' keys."""
        dense_chunk = _create_sample_chunk("c1", "doc1", "Dense content")
        sparse_chunk = _create_sample_chunk("c1", "doc1", "Sparse content")

        fused = fuse_dense_and_sparse([dense_chunk], [sparse_chunk], k=60)
        assert len(fused) == 1
        assert fused[0].metadata["source_ranks"] == {"dense": 1, "sparse": 1}
        assert fused[0].metadata["rrf_score"] == pytest.approx(2.0 / 61)

    @pytest.mark.asyncio
    async def test_afuse_dense_and_sparse(self) -> None:
        """Async wrapper afuse_dense_and_sparse executes and returns fused candidates."""
        dense = [_create_sample_chunk("d1", "doc1", "content")]
        sparse = [_create_sample_chunk("s1", "doc2", "content")]

        fused = await afuse_dense_and_sparse(dense, sparse, k=60, top_n=5)
        assert len(fused) == 2


class TestRRFPerformance:
    """Performance test validating sub-60ms fusion latency."""

    def test_latency_under_60ms_threshold(self) -> None:
        """Fusing 100 dense + 100 sparse items must complete well under 60ms."""
        dense = [{"chunk_id": f"chunk_d_{i}", "content": f"d {i}"} for i in range(100)]
        sparse = [{"chunk_id": f"chunk_s_{i}", "content": f"s {i}"} for i in range(100)]

        start = time.perf_counter()
        fused = reciprocal_rank_fusion([dense, sparse], k=60, top_n=15)
        duration_ms = (time.perf_counter() - start) * 1000

        assert len(fused) == 15
        assert duration_ms < 60.0  # PRD requirement: < 60ms
        assert duration_ms < 10.0  # Practical target: < 10ms


class TestRetrievedChunkSubscript:
    """Tests for subscript access on RetrievedChunk model."""

    def test_subscript_access_and_contains(self) -> None:
        """RetrievedChunk supports dictionary-like indexing and membership test."""
        chunk = RetrievedChunk(
            chunk_id="ch_100",
            source_file="docs/intro.md",
            content="Introduction text",
            score=0.95,
            metadata={"doc_id": "doc_main", "custom_tag": "val"},
        )

        assert chunk["chunk_id"] == "ch_100"
        assert chunk["source_file"] == "docs/intro.md"
        assert chunk["content"] == "Introduction text"
        assert chunk["score"] == 0.95
        assert chunk["doc_id"] == "doc_main"
        assert chunk["custom_tag"] == "val"

        assert "doc_id" in chunk
        assert "score" in chunk
        assert "non_existent_key" not in chunk

        with pytest.raises(KeyError):
            _ = chunk["non_existent_key"]
