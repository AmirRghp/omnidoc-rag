"""Reciprocal Rank Fusion (RRF) Engine for merging Dense and Sparse retrieval channels."""

import asyncio
from collections.abc import Sequence
from typing import Any

from src.common.schemas import RetrievedChunk

__all__ = [
    "afuse_dense_and_sparse",
    "fuse_dense_and_sparse",
    "reciprocal_rank_fusion",
]


def _extract_item_info(
    item: RetrievedChunk | dict[str, Any],
) -> tuple[str, str, str, str, dict[str, Any]]:
    """Extract (identifier, doc_id, source_file, content, metadata) from chunk or dict."""
    if isinstance(item, RetrievedChunk):
        item_id = str(item.chunk_id)
        doc_id = str(item.metadata.get("doc_id", item_id))
        source_file = item.source_file
        content = item.content
        metadata = dict(item.metadata)
        metadata.setdefault("doc_id", doc_id)
        return item_id, doc_id, source_file, content, metadata

    if isinstance(item, dict):
        item_id = str(
            item.get("chunk_id")
            or item.get("doc_id")
            or item.get("id")
            or item.get("point_id")
            or "unknown"
        )
        doc_id = str(item.get("doc_id") or item_id)
        source_file = str(item.get("source_file") or f"{doc_id}.md")
        content = str(item.get("content") or doc_id)
        metadata = {k: v for k, v in item.items() if k not in {"content", "source_file"}}
        metadata.setdefault("doc_id", doc_id)
        return item_id, doc_id, source_file, content, metadata

    raise TypeError(f"Unsupported item type in rankings: {type(item).__name__}")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk | dict[str, Any]]],
    k: int = 60,
    top_n: int = 15,
    channel_names: Sequence[str] | None = None,
) -> list[RetrievedChunk]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF Score(d) = sum_{m in rankings} 1 / (k + r_m(d))

    Args:
        rankings: Sequence of ranked candidate lists from different retrieval channels.
        k: Smoothing constant parameter (default: 60).
        top_n: Maximum number of merged and deduplicated candidates to return (default: 15).
        channel_names: Optional sequence of names corresponding to each ranking list.

    Returns:
        Deduplicated, sorted list of top_n RetrievedChunk objects.

    Raises:
        ValueError: If k <= 0 or top_n <= 0.
    """
    if k <= 0:
        raise ValueError(f"Smoothing constant k must be positive, got {k}")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")

    if not rankings or all(len(r) == 0 for r in rankings):
        return []

    rrf_scores: dict[str, float] = {}
    item_channel_ranks: dict[str, dict[str, int]] = {}
    item_data: dict[str, tuple[str, str, str, dict[str, Any]]] = {}
    best_ranks: dict[str, int] = {}

    for channel_idx, ranking in enumerate(rankings):
        channel_label = (
            channel_names[channel_idx]
            if channel_names and channel_idx < len(channel_names)
            else f"channel_{channel_idx}"
        )

        seen_in_channel: set[str] = set()

        for rank_zero_based, raw_item in enumerate(ranking):
            item_id, doc_id, source_file, content, metadata = _extract_item_info(raw_item)

            # Avoid counting duplicate occurrences in the same channel multiple times
            if item_id in seen_in_channel:
                continue
            seen_in_channel.add(item_id)

            rank_1_based = rank_zero_based + 1
            rrf_increment = 1.0 / (k + rank_1_based)

            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + rrf_increment

            if item_id not in item_channel_ranks:
                item_channel_ranks[item_id] = {}
            item_channel_ranks[item_id][channel_label] = rank_1_based

            if item_id not in best_ranks or rank_1_based < best_ranks[item_id]:
                best_ranks[item_id] = rank_1_based

            # Retain the most complete item representation
            if item_id not in item_data:
                item_data[item_id] = (doc_id, source_file, content, metadata)
            else:
                # Merge metadata from other channels
                (
                    existing_doc_id,
                    existing_source,
                    existing_content,
                    existing_meta,
                ) = item_data[item_id]
                merged_meta = {**metadata, **existing_meta}
                merged_content = existing_content or content
                merged_source = existing_source if existing_source != "unknown" else source_file
                item_data[item_id] = (existing_doc_id, merged_source, merged_content, merged_meta)

    # Sort deterministically:
    # 1. Higher RRF score first (-rrf_score)
    # 2. Lower best rank across channels (best_rank)
    # 3. Alphabetical item_id
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda i_id: (-rrf_scores[i_id], best_ranks.get(i_id, 999999), i_id),
    )

    fused_candidates: list[RetrievedChunk] = []

    for final_rank, item_id in enumerate(sorted_ids[:top_n], start=1):
        doc_id, source_file, content, base_meta = item_data[item_id]
        score_val = rrf_scores[item_id]
        calibrated_score = max(0.0, min(1.0, score_val))

        combined_metadata = {
            **base_meta,
            "doc_id": doc_id,
            "rrf_score": score_val,
            "rrf_rank": final_rank,
            "source_ranks": item_channel_ranks[item_id],
        }

        fused_candidates.append(
            RetrievedChunk(
                chunk_id=item_id,
                source_file=source_file,
                content=content,
                score=calibrated_score,
                metadata=combined_metadata,
            )
        )

    return fused_candidates


def fuse_dense_and_sparse(
    dense_results: Sequence[RetrievedChunk | dict[str, Any]],
    sparse_results: Sequence[RetrievedChunk | dict[str, Any]],
    k: int = 60,
    top_n: int = 15,
) -> list[RetrievedChunk]:
    """Fuse dense (semantic) and sparse (lexical) retrieval channels via RRF.

    Args:
        dense_results: Ranked candidates from dense vector search (Qdrant).
        sparse_results: Ranked candidates from sparse lexical search (BM25).
        k: Smoothing constant parameter (default: 60).
        top_n: Maximum number of merged candidates to return (default: 15).

    Returns:
        Top-N balanced candidate passages.
    """
    return reciprocal_rank_fusion(
        rankings=[dense_results, sparse_results],
        k=k,
        top_n=top_n,
        channel_names=["dense", "sparse"],
    )


async def afuse_dense_and_sparse(
    dense_results: Sequence[RetrievedChunk | dict[str, Any]],
    sparse_results: Sequence[RetrievedChunk | dict[str, Any]],
    k: int = 60,
    top_n: int = 15,
) -> list[RetrievedChunk]:
    """Asynchronous wrapper for reciprocal rank fusion of dense and sparse channels."""
    return await asyncio.to_thread(
        fuse_dense_and_sparse,
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=k,
        top_n=top_n,
    )
