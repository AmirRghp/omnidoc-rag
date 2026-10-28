"""Unit tests for two-tier Redis exact & semantic query cache (Task 6.3)."""

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError

from src.common.config import Settings
from src.common.exceptions import CacheServiceUnavailableError
from src.common.schemas import CitationItem
from src.services.cache import (
    CachedResponse,
    CacheService,
    RedisQueryCache,
    compute_query_hash,
    normalize_cache_query,
)


class AsyncMockRedis:
    """In-memory async Redis double for isolated unit testing."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.closed: bool = False
        self.should_fail_ping: bool = False
        self.should_fail_get: bool = False
        self.should_fail_set: bool = False
        self.get_call_count: int = 0

    async def get(self, name: str) -> str | None:
        self.get_call_count += 1
        if self.should_fail_get:
            raise RedisConnectionError("Simulated Redis connection failure during GET")
        return self.data.get(name)

    async def set(
        self,
        name: str,
        value: str,
        ex: int | None = None,
        **kwargs: Any,
    ) -> bool:
        if self.should_fail_set:
            raise RedisConnectionError("Simulated Redis connection failure during SET")
        self.data[name] = value
        if ex is not None:
            self.ttls[name] = ex
        return True

    async def delete(self, *names: str) -> int:
        deleted = 0
        for name in names:
            if name in self.data:
                del self.data[name]
                self.ttls.pop(name, None)
                deleted += 1
        return deleted

    async def scan_iter(
        self,
        match: str | None = None,
        count: int | None = None,
    ) -> AsyncIterator[str]:
        prefix = match.rstrip("*") if match else ""
        for key in list(self.data.keys()):
            if key.startswith(prefix):
                yield key

    async def ping(self) -> bool:
        if self.should_fail_ping:
            raise RedisConnectionError("Simulated Redis unreachable on ping")
        return True

    async def aclose(self) -> None:
        self.closed = True


def create_test_cache(
    mock_redis: AsyncMockRedis,
    settings: Settings,
    **kwargs: Any,
) -> RedisQueryCache:
    """Helper to instantiate RedisQueryCache with typed mock redis client."""
    return RedisQueryCache(
        client=cast(aioredis.Redis, mock_redis),
        settings=settings,
        **kwargs,
    )


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing isolated mock settings for testing."""
    return Settings(
        REDIS_HOST="mock-redis",
        REDIS_PORT=6379,
        CACHE_TTL_SECONDS=86400,
        SEMANTIC_CACHE_ENABLED=True,
    )


@pytest.fixture
def sample_citations() -> list[CitationItem]:
    """Sample citation items for caching round-trip tests."""
    return [
        CitationItem(
            source_file="manuals/docker.md",
            section_heading="Daemon IPC Config",
            relevance_score=0.92,
            snippet="Use `--ipc=host` flag to share memory with host.",
        ),
        CitationItem(
            source_file="manuals/postgres_fa.md",
            section_heading="تنظیمات حافظه",
            relevance_score=0.88,
            snippet="پارامتر shared_buffers مقدار حافظه را تعیین می‌کند.",
        ),
    ]


class TestQueryNormalizationAndHashing:
    """Tests for bilingual Unicode normalization, whitespace collapsing, and SHA-256."""

    def test_persian_character_normalization(self) -> None:
        """Arabic Yeh/Kaf variants normalize to Persian while preserving ZWNJ."""
        # Query with Arabic yeh (ي) and kaf (ك)
        arabic_variant = "دستور اجراي داكر با كانفيگ"
        # Query with Persian yeh (ی) and kaf (ک)
        persian_standard = "دستور اجرای داکر با کانفیگ"

        assert normalize_cache_query(arabic_variant) == normalize_cache_query(persian_standard)
        assert compute_query_hash(normalize_cache_query(arabic_variant)) == compute_query_hash(
            normalize_cache_query(persian_standard)
        )

    def test_preserves_zwnj_in_persian_orthography(self) -> None:
        """Zero-Width Non-Joiner (U+200C) is strictly preserved."""
        query_with_zwnj = "مستندسازی کامپوننت\u200cها"
        normalized = normalize_cache_query(query_with_zwnj)

        assert "\u200c" in normalized
        assert normalized == "مستندسازی کامپوننت\u200cها"

    def test_case_folding_and_whitespace_collapsing(self) -> None:
        """English casing and redundant internal whitespace produce identical hash."""
        query1 = "  DOCKER   run   --ipc=host  "
        query2 = "docker run --ipc=host"

        assert normalize_cache_query(query1) == "docker run --ipc=host"
        assert normalize_cache_query(query2) == "docker run --ipc=host"
        assert compute_query_hash(normalize_cache_query(query1)) == compute_query_hash(
            normalize_cache_query(query2)
        )

    def test_cache_key_construction_with_default_and_custom_prefix(self) -> None:
        """build_cache_key constructs valid namespaced Redis keys."""
        cache_default = RedisQueryCache(key_prefix="cache:response")
        key, norm, h = cache_default.build_cache_key("PostgreSQL Config")
        assert key == f"cache:response:{h}"
        assert norm == "postgresql config"

        cache_custom = RedisQueryCache(key_prefix="cache:query")
        custom_key, _, _ = cache_custom.build_cache_key("PostgreSQL Config")
        assert custom_key == f"cache:query:{h}"


class TestRedisQueryCacheOperations:
    """Tests for core get, set, delete, and clear operations."""

    @pytest.mark.asyncio
    async def test_set_and_get_roundtrip_with_citations(
        self,
        mock_settings: Settings,
        sample_citations: list[CitationItem],
    ) -> None:
        """Cached entry is serialized to JSON and correctly deserialized with citations."""
        mock_redis = AsyncMockRedis()
        cache = create_test_cache(mock_redis, mock_settings)

        query = "How to configure shared memory in Docker?"
        response_text = "Use `--ipc=host` during container initialization."

        # Set cache
        success = await cache.set(
            query=query,
            response=response_text,
            citations=sample_citations,
            model="Qwen/Qwen2.5-7B-Instruct",
            ttl=7200,
        )
        assert success is True

        # Verify Redis key and TTL
        key, _, _ = cache.build_cache_key(query)
        assert key in mock_redis.data
        assert mock_redis.ttls[key] == 7200

        # Invalidate L1 to force Redis L2 read
        cache._l1_cache.clear()

        # Get cache
        cached = await cache.get(query)
        assert cached is not None
        assert isinstance(cached, CachedResponse)
        assert cached.query == query
        assert cached.response == response_text
        assert cached.model == "Qwen/Qwen2.5-7B-Instruct"
        assert len(cached.citations) == 2
        assert cached.citations[0].source_file == "manuals/docker.md"
        assert cached.citations[1].section_heading == "تنظیمات حافظه"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key_returns_none(self, mock_settings: Settings) -> None:
        """get() returns None when query is not in cache."""
        mock_redis = AsyncMockRedis()
        cache = create_test_cache(mock_redis, mock_settings)

        result = await cache.get("Uncached query")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_invalidates_l1_and_l2(self, mock_settings: Settings) -> None:
        """delete() evicts key from both L1 memory and L2 Redis."""
        mock_redis = AsyncMockRedis()
        cache = create_test_cache(mock_redis, mock_settings)

        query = "Clear me"
        await cache.set(query, "Will be deleted")
        key, _, _ = cache.build_cache_key(query)

        assert key in cache._l1_cache
        assert key in mock_redis.data

        deleted = await cache.delete(query)
        assert deleted is True
        assert key not in cache._l1_cache
        assert key not in mock_redis.data

    @pytest.mark.asyncio
    async def test_clear_removes_all_prefixed_keys(self, mock_settings: Settings) -> None:
        """clear() scans and deletes all keys under the cache prefix."""
        mock_redis = AsyncMockRedis()
        cache = create_test_cache(
            mock_redis,
            mock_settings,
            key_prefix="cache:response",
        )

        await cache.set("Query 1", "Ans 1")
        await cache.set("Query 2", "Ans 2")
        mock_redis.data["unrelated:key"] = "Keep me"

        cleared = await cache.clear()
        assert cleared is True
        assert len(cache._l1_cache) == 0
        assert len(mock_redis.data) == 1
        assert "unrelated:key" in mock_redis.data


class TestTwoTierCachingL1AndL2:
    """Tests for L1 in-memory fast caching and LRU eviction."""

    @pytest.mark.asyncio
    async def test_l1_hit_bypasses_redis_get(self, mock_settings: Settings) -> None:
        """When L1 contains entry, get() returns immediately without calling Redis."""
        mock_redis = AsyncMockRedis()
        cache = create_test_cache(mock_redis, mock_settings)

        query = "Fast query"
        await cache.set(query, "Fast answer")
        assert mock_redis.get_call_count == 0

        # Retrieve: hits L1 directly
        res = await cache.get(query)
        assert res is not None
        assert res.response == "Fast answer"
        assert mock_redis.get_call_count == 0  # Redis was not queried

    @pytest.mark.asyncio
    async def test_l1_eviction_when_maxsize_exceeded(self, mock_settings: Settings) -> None:
        """L1 cache evicts oldest entry when l1_maxsize is reached."""
        mock_redis = AsyncMockRedis()
        cache = create_test_cache(
            mock_redis,
            mock_settings,
            l1_maxsize=2,
        )

        await cache.set("Q1", "A1")
        await cache.set("Q2", "A2")
        key1, _, _ = cache.build_cache_key("Q1")
        key2, _, _ = cache.build_cache_key("Q2")
        assert key1 in cache._l1_cache
        assert key2 in cache._l1_cache

        # Adding 3rd entry evicts Q1 (oldest)
        await cache.set("Q3", "A3")
        key3, _, _ = cache.build_cache_key("Q3")
        assert key1 not in cache._l1_cache
        assert key2 in cache._l1_cache
        assert key3 in cache._l1_cache


class TestFaultToleranceAndGracefulDegradation:
    """Tests for self-healing behavior when Redis is unreachable (docs/ARCHITECTURE.md §9)."""

    @pytest.mark.asyncio
    async def test_redis_get_failure_degrades_gracefully_by_default(
        self,
        mock_settings: Settings,
    ) -> None:
        """When Redis get fails, cache returns None and logs warning without throwing."""
        mock_redis = AsyncMockRedis()
        mock_redis.should_fail_get = True
        cache = create_test_cache(
            mock_redis,
            mock_settings,
            raise_on_error=False,
        )

        result = await cache.get("Failing query")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_set_failure_returns_false_without_raising(
        self,
        mock_settings: Settings,
    ) -> None:
        """When Redis set fails, cache returns False without interrupting generation pipeline."""
        mock_redis = AsyncMockRedis()
        mock_redis.should_fail_set = True
        cache = create_test_cache(
            mock_redis,
            mock_settings,
            raise_on_error=False,
        )

        success = await cache.set("Failing query", "Answer")
        assert success is False

    @pytest.mark.asyncio
    async def test_redis_failure_raises_when_raise_on_error_true(
        self,
        mock_settings: Settings,
    ) -> None:
        """When raise_on_error=True, CacheServiceUnavailableError is raised on failure."""
        mock_redis = AsyncMockRedis()
        mock_redis.should_fail_get = True
        cache = create_test_cache(
            mock_redis,
            mock_settings,
            raise_on_error=True,
        )

        with pytest.raises(CacheServiceUnavailableError) as exc_info:
            await cache.get("Strict error check")
        assert exc_info.value.status_code == 503
        assert exc_info.value.code == "ERR_REDIS_UNAVAILABLE"


class TestHealthCheckAndLifecycle:
    """Tests for health check probe, context manager, and architectural alias."""

    @pytest.mark.asyncio
    async def test_check_health_returns_true_on_ping_success(
        self,
        mock_settings: Settings,
    ) -> None:
        """check_health returns True when Redis responds to ping."""
        mock_redis = AsyncMockRedis()
        cache = create_test_cache(mock_redis, mock_settings)
        assert await cache.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_returns_false_on_ping_failure(
        self,
        mock_settings: Settings,
    ) -> None:
        """check_health returns False when Redis ping raises connection error."""
        mock_redis = AsyncMockRedis()
        mock_redis.should_fail_ping = True
        cache = create_test_cache(mock_redis, mock_settings)
        assert await cache.check_health() is False

    def test_architectural_alias(self) -> None:
        """CacheService is an architectural alias of RedisQueryCache."""
        assert CacheService is RedisQueryCache

    @pytest.mark.asyncio
    async def test_async_context_manager_lifecycle(self, mock_settings: Settings) -> None:
        """Cache properly supports async context manager protocol."""
        async with RedisQueryCache(
            host="mock-host",
            settings=mock_settings,
        ) as cache:
            assert cache.ttl_seconds == 86400
