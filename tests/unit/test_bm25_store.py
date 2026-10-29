"""Unit tests for bilingual BM25 tokenizer and persistent BM25StoreManager."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.common.schemas import DocumentChunkPayload
from src.ingestion.bm25_store import BM25StoreManager, tokenize_bilingual


@pytest.fixture
def sample_bilingual_chunks() -> list[DocumentChunkPayload]:
    """Provide a bilingual set of technical chunks for BM25 indexing tests."""
    return [
        DocumentChunkPayload(
            doc_id="docker_ipc",
            chunk_id=0,
            content="Use the --ipc=host flag when running Docker containers that share memory.",
            source_file="manuals/docker_cli_run.md",
            section_heading="## IPC settings (--ipc)",
            language="en",
            token_count=15,
        ),
        DocumentChunkPayload(
            doc_id="postgres_conn",
            chunk_id=0,
            content="Increase max_connections and shared_buffers in postgresql.conf for high load.",
            source_file="manuals/postgres_memory.md",
            section_heading="## Memory Configuration",
            language="en",
            token_count=14,
        ),
        DocumentChunkPayload(
            doc_id="persian_docker_guide",
            chunk_id=0,
            content=(
                "برای اشتراک‌گذاری حافظه بین کانتینرهای داکر از فلگ --ipc=host "
                "و مسیر /var/run/docker.sock استفاده کنید."
            ),
            source_file="manuals/sample_persian_docker_guide.md",
            section_heading="## تنظیمات حافظه مشترک در داکر",
            language="fa",
            token_count=22,
        ),
    ]


class TestTokenizeBilingual:
    """Test bilingual technical tokenizer behavior."""

    def test_preserves_cli_flags_and_options(self) -> None:
        text = "Run docker with --ipc=host -v and --max-connections=100"
        tokens = tokenize_bilingual(text)
        assert "--ipc=host" in tokens
        assert "-v" in tokens
        assert "--max-connections=100" in tokens

    def test_preserves_error_codes_and_paths(self) -> None:
        text = "Error ERR_QDRANT_TIMEOUT on /var/run/docker.sock or max_connections"
        tokens = tokenize_bilingual(text)
        assert "err_qdrant_timeout" in tokens
        assert "/var/run/docker.sock" in tokens
        assert "max_connections" in tokens

    def test_normalizes_arabic_chars_and_preserves_persian_zwnj(self) -> None:
        # Contains Arabic Yeh (ي), Arabic Kaf (ك), and Persian ZWNJ (\u200c)
        text = "براي اشتراك\u200cگذاري كانتينرها با --network=bridge"
        tokens = tokenize_bilingual(text)
        assert "برای" in tokens
        assert "اشتراک\u200cگذاری" in tokens
        assert "کانتینرها" in tokens
        assert "--network=bridge" in tokens

    def test_strips_directional_overrides_and_handles_empty_input(self) -> None:
        assert tokenize_bilingual("") == []
        assert tokenize_bilingual("   ... !!!   ") == []
        tokens = tokenize_bilingual("\u202edocker\u202c --privileged")
        assert "docker" in tokens
        assert "--privileged" in tokens


class TestBM25StoreManager:
    """Test BM25 index construction, atomic persistence, and bilingual search."""

    def test_default_index_path(self) -> None:
        manager = BM25StoreManager()
        assert manager.index_path == Path("data/processed/bm25_index.pkl")
        assert manager.chunk_count == 0

    def test_build_index_and_atomic_save_via_os_replace(
        self,
        tmp_path: Path,
        sample_bilingual_chunks: list[DocumentChunkPayload],
    ) -> None:
        index_file = tmp_path / "processed" / "bm25_index.pkl"
        manager = BM25StoreManager(index_path=index_file)

        with patch("src.ingestion.bm25_store.os.replace", wraps=os.replace) as mock_replace:
            indexed_count = manager.build_index(sample_bilingual_chunks, persist=True)

            assert indexed_count == 3
            assert manager.chunk_count == 3
            assert index_file.exists()
            mock_replace.assert_called_once()
            src_arg, dst_arg = mock_replace.call_args[0]
            assert Path(dst_arg) == index_file
            assert Path(src_arg).parent == index_file.parent
            # Temporary file should no longer exist after atomic replacement
            assert not Path(src_arg).exists()

    def test_load_persisted_index_and_search(
        self,
        tmp_path: Path,
        sample_bilingual_chunks: list[DocumentChunkPayload],
    ) -> None:
        index_file = tmp_path / "bm25_index.pkl"
        writer = BM25StoreManager(index_path=index_file)
        writer.build_index(sample_bilingual_chunks, persist=True)

        reader = BM25StoreManager(index_path=index_file)
        assert reader.load() is True
        assert reader.chunk_count == 3

        # Search for Persian + CLI flag query
        results = reader.search("فلگ --ipc=host در داکر", top_k=2)
        assert len(results) == 2
        assert results[0].source_file == "manuals/sample_persian_docker_guide.md"
        assert 0.0 <= results[0].score <= 1.0
        assert results[0].score >= results[1].score
        assert "raw_bm25_score" in results[0].metadata
        assert results[0].metadata["bm25_rank"] == 1

    def test_load_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "nonexistent.pkl"
        manager = BM25StoreManager(index_path=missing_file)
        assert manager.load() is False

    def test_idempotent_upsert_chunks_deduplicates_by_point_id(
        self,
        tmp_path: Path,
        sample_bilingual_chunks: list[DocumentChunkPayload],
    ) -> None:
        index_file = tmp_path / "bm25_index.pkl"
        manager = BM25StoreManager(index_path=index_file)
        manager.upsert_chunks(sample_bilingual_chunks[:2], persist=False)
        assert manager.chunk_count == 2

        # Re-upserting chunk 0 with updated content plus chunk 2
        updated_chunk_0 = sample_bilingual_chunks[0].model_copy(
            update={"content": "Updated content with --shm-size=2g flag."}
        )
        total = manager.upsert_chunks([updated_chunk_0, sample_bilingual_chunks[2]], persist=True)
        assert total == 3
        assert manager.chunk_count == 3

        results = manager.search("--shm-size=2g", top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == updated_chunk_0.point_id
        assert "--shm-size=2g" in results[0].content

    def test_delete_by_doc_id_removes_document_and_rebuilds(
        self,
        tmp_path: Path,
        sample_bilingual_chunks: list[DocumentChunkPayload],
    ) -> None:
        index_file = tmp_path / "bm25_index.pkl"
        manager = BM25StoreManager(index_path=index_file)
        manager.build_index(sample_bilingual_chunks, persist=False)

        removed = manager.delete_by_doc_id("postgres_conn", persist=True)
        assert removed == 1
        assert manager.chunk_count == 2

        results = manager.search("max_connections shared_buffers", top_k=5)
        assert all(r.metadata.get("doc_id") != "postgres_conn" for r in results)

    @pytest.mark.asyncio
    async def test_asearch_executes_via_thread(
        self,
        tmp_path: Path,
        sample_bilingual_chunks: list[DocumentChunkPayload],
    ) -> None:
        manager = BM25StoreManager(index_path=tmp_path / "bm25_index.pkl")
        manager.build_index(sample_bilingual_chunks, persist=False)

        results = await manager.asearch("max_connections", top_k=1)
        assert len(results) == 1
        assert results[0].source_file == "manuals/postgres_memory.md"

    def test_search_empty_index_or_unmatched_query_returns_empty(
        self,
        tmp_path: Path,
        sample_bilingual_chunks: list[DocumentChunkPayload],
    ) -> None:
        manager = BM25StoreManager(index_path=tmp_path / "bm25_index.pkl")
        assert manager.search("docker") == []

        manager.build_index(sample_bilingual_chunks, persist=False)
        assert len(manager.chunks) == 3
        assert manager.search("   ") == []
        assert manager.search("completely_nonexistent_token_xyz123") == []

    def test_validation_errors_and_atomic_save_cleanup(
        self,
        tmp_path: Path,
        sample_bilingual_chunks: list[DocumentChunkPayload],
    ) -> None:
        index_file = tmp_path / "bm25_index.pkl"
        manager = BM25StoreManager(index_path=index_file)
        assert manager.upsert_chunks([], persist=False) == 0

        with pytest.raises(ValueError, match="top_k must be positive"):
            manager.search("docker", top_k=0)

        with pytest.raises(ValueError, match="doc_id cannot be empty"):
            manager.delete_by_doc_id("   ")

        with (
            patch("src.ingestion.bm25_store.pickle.dump", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            manager.build_index(sample_bilingual_chunks, persist=True)

        # Verify no leftover temporary files remain after failed save
        assert list(tmp_path.glob(".bm25_index_*.tmp")) == []
        assert not index_file.exists()

    def test_sha256_integrity_verification_rejects_tampered_pickle(
        self,
        tmp_path: Path,
        sample_bilingual_chunks: list[DocumentChunkPayload],
    ) -> None:
        index_file = tmp_path / "bm25_index.pkl"
        manager = BM25StoreManager(index_path=index_file)
        manager.build_index(sample_bilingual_chunks, persist=True)
        assert manager.signature_path.is_file()

        # Tamper with the persisted pickle file
        index_file.write_bytes(index_file.read_bytes() + b"corrupted")

        reader = BM25StoreManager(index_path=index_file)
        with pytest.raises(ValueError, match="BM25 index integrity check failed"):
            reader.load()
