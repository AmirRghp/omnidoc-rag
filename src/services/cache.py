"""Two-tier (L1 In-Memory + L2 Redis) semantic and exact query cache."""

import hashlib
import time
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timezone
from types import TracebackType
from typing import Any

import redis.asyncio as aioredis
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError

from src.common.config import Settings, create_settings
from src.common.exceptions import CacheServiceUnavailableError
from src.common.logger import get_logger
from src.common.schemas import CitationItem
from src.common.security import normalize_persian_text, strip_directional_overrides

UTC = timezone.utc

logger = get_logger(__name__)


def normalize_cache_query(query: str) -> str:
    """Normalize query for deterministic bilingual cache key calculation.

    Strips invisible Unicode directional overrides, converts Arabic variants
    to standard Persian while preserving ZWNJ, lowercases English tokens,
    and collapses redundant whitespace.

    Args:
        query: Raw query text.

    Returns:
        Canonical normalized query string.
    """
    cleaned = strip_directional_overrides(query)
    persian_normalized = normalize_persian_text(cleaned)
    return " ".join(persian_normalized.lower().split())


def compute_query_hash(normalized_query: str) -> str:
    """Calculate deterministic SHA-256 digest of normalized query string.

    Args:
        normalized_query: Canonical query string.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


class CachedResponse(BaseModel):
    """Schema for cached RAG query responses in Redis and in-memory cache."""

    model_config = ConfigDict(strict=True, frozen=True)

    query: str = Field(..., description="Original user query.")
    normalized_query: str = Field(..., description="Canonical normalized query.")
    query_hash: str = Field(..., description="SHA-256 hash of normalized query.")
    response: str = Field(..., description="Generated answer text.")
    citations: list[CitationItem] = Field(
        default_factory=list,
        description="Grounded source citations.",
    )
    model: str = Field(default="", description="Inference model name.")
    cached_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="UTC ISO-8601 caching timestamp.",
    )


class RedisQueryCache:
    """Two-tier (L1 In-Memory LRU + L2 Redis) cache for bilingual queries."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        password: str | None = None,
        ttl_seconds: int | None = None,
        key_prefix: str = "cache:query",
        client: aioredis.Redis | None = None,
        settings: Settings | None = None,
        l1_maxsize: int = 1000,
        l1_ttl_seconds: int = 300,
        raise_on_error: bool = False,
    ) -> None:
        """Initialize the query cache.

        Args:
            host: Redis host address (defaults to settings.redis_host).
            port: Redis port (defaults to settings.redis_port).
            password: Redis authentication password.
            ttl_seconds: Default expiration for cached entries in seconds.
            key_prefix: Redis key namespace prefix (e.g. 'cache:response' or 'cache:query').
            client: Optional injected redis.asyncio.Redis instance.
            settings: Optional application settings.
            l1_maxsize: Maximum number of entries kept in memory L1 cache.
            l1_ttl_seconds: TTL for entries in memory L1 cache.
            raise_on_error: If True, raises CacheServiceUnavailableError on failure
                instead of degrading gracefully.
        """
        self.settings = settings or create_settings()
        self.host = host or self.settings.redis_host
        self.port = port or self.settings.redis_port
        self.password = password or self.settings.redis_password
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else self.settings.cache_ttl_seconds
        )
        self.key_prefix = key_prefix.rstrip(":")
        self.l1_maxsize = max(1, l1_maxsize)
        self.l1_ttl_seconds = max(1, l1_ttl_seconds)
        self.raise_on_error = raise_on_error

        # In-memory L1 cache: key -> (CachedResponse, expiry_monotonic_timestamp)
        self._l1_cache: OrderedDict[str, tuple[CachedResponse, float]] = OrderedDict()

        self._injected_client = client is not None
        if client is not None:
            self._client = client
        else:
            self._client = aioredis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )

    def _l1_get(self, key: str) -> CachedResponse | None:
        """Retrieve entry from in-memory L1 cache if unexpired."""
        entry = self._l1_cache.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.monotonic() > expiry:
            self._l1_cache.pop(key, None)
            return None
        self._l1_cache.move_to_end(key)
        return value

    def _l1_set(self, key: str, value: CachedResponse) -> None:
        """Set entry in in-memory L1 cache with LRU eviction."""
        if key in self._l1_cache:
            self._l1_cache.move_to_end(key)
        elif len(self._l1_cache) >= self.l1_maxsize:
            self._l1_cache.popitem(last=False)
        self._l1_cache[key] = (value, time.monotonic() + self.l1_ttl_seconds)

    def _l1_delete(self, key: str) -> None:
        """Delete entry from in-memory L1 cache."""
        self._l1_cache.pop(key, None)

    def build_cache_key(self, query: str) -> tuple[str, str, str]:
        """Generate cache key, normalized query, and SHA-256 hash.

        Args:
            query: Raw user query string.

        Returns:
            Tuple of (redis_key, normalized_query, sha256_hash).
        """
        normalized = normalize_cache_query(query)
        query_hash = compute_query_hash(normalized)
        redis_key = f"{self.key_prefix}:{query_hash}"
        return redis_key, normalized, query_hash

    async def get(self, query: str) -> CachedResponse | None:
        """Retrieve cached response for query.

        Checks L1 in-memory cache first, then Redis. If Redis is unavailable,
        logs warning and degrades gracefully (returns None).

        Args:
            query: User query string.

        Returns:
            CachedResponse if cached and valid, None otherwise.
        """
        redis_key, normalized, _ = self.build_cache_key(query)
        if not normalized:
            return None

        # Check L1 cache
        l1_hit = self._l1_get(redis_key)
        if l1_hit is not None:
            logger.debug("L1 cache hit for key %s", redis_key)
            return l1_hit

        # Check L2 Redis
        try:
            raw_data = await self._client.get(redis_key)
            if raw_data:
                logger.debug("L2 Redis cache hit for key %s", redis_key)
                cached = CachedResponse.model_validate_json(raw_data)
                self._l1_set(redis_key, cached)
                return cached
            return None
        except (RedisError, Exception) as exc:
            logger.warning(
                "Redis query cache lookup failed for key %s: %s. Bypassing cache.",
                redis_key,
                exc,
            )
            if self.raise_on_error:
                raise CacheServiceUnavailableError(
                    message="Redis cache service is unreachable",
                    detail=f"Failed getting key {redis_key}: {exc}",
                ) from exc
            return None

    async def set(
        self,
        query: str,
        response: str,
        citations: Sequence[CitationItem | dict[str, Any]] | None = None,
        model: str | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Store verified response and citations in cache.

        Args:
            query: User query text.
            response: Generated answer text.
            citations: List of CitationItem instances or dictionaries.
            model: Optional model name.
            ttl: Optional TTL override in seconds (defaults to self.ttl_seconds).

        Returns:
            True if stored successfully, False if skipped or failed.
        """
        redis_key, normalized, query_hash = self.build_cache_key(query)
        if not normalized or not response:
            return False

        formatted_citations: list[CitationItem] = []
        if citations:
            for c in citations:
                if isinstance(c, CitationItem):
                    formatted_citations.append(c)
                elif isinstance(c, dict):
                    formatted_citations.append(CitationItem(**c))

        cached_response = CachedResponse(
            query=query,
            normalized_query=normalized,
            query_hash=query_hash,
            response=response,
            citations=formatted_citations,
            model=model or self.settings.llm_model,
        )

        # Populate L1 cache
        self._l1_set(redis_key, cached_response)

        # Store in L2 Redis
        effective_ttl = ttl if ttl is not None else self.ttl_seconds
        payload_json = cached_response.model_dump_json()

        try:
            await self._client.set(redis_key, payload_json, ex=effective_ttl)
            logger.debug(
                "Stored query response in cache key %s (TTL: %ds)",
                redis_key,
                effective_ttl,
            )
            return True
        except (RedisError, Exception) as exc:
            logger.warning(
                "Redis query cache set failed for key %s: %s. Continuing without cache.",
                redis_key,
                exc,
            )
            if self.raise_on_error:
                raise CacheServiceUnavailableError(
                    message="Redis cache service is unreachable",
                    detail=f"Failed setting key {redis_key}: {exc}",
                ) from exc
            return False

    async def delete(self, query: str) -> bool:
        """Delete cached entry for a query from L1 and L2 cache.

        Args:
            query: User query text.

        Returns:
            True if deleted or absent, False on Redis error.
        """
        redis_key, _, _ = self.build_cache_key(query)
        self._l1_delete(redis_key)

        try:
            await self._client.delete(redis_key)
            return True
        except (RedisError, Exception) as exc:
            logger.warning("Redis cache delete failed for key %s: %s", redis_key, exc)
            if self.raise_on_error:
                raise CacheServiceUnavailableError(
                    message="Redis cache service is unreachable",
                    detail=f"Failed deleting key {redis_key}: {exc}",
                ) from exc
            return False

    async def clear(self) -> bool:
        """Clear all entries matching the key prefix from L1 and L2 cache.

        Returns:
            True if cleared, False on Redis error.
        """
        self._l1_cache.clear()

        try:
            pattern = f"{self.key_prefix}:*"
            keys: list[str] = []
            async for k in self._client.scan_iter(match=pattern, count=100):
                keys.append(str(k))

            if keys:
                await self._client.delete(*keys)
            logger.info("Cleared %d Redis cache keys matching %s", len(keys), pattern)
            return True
        except (RedisError, Exception) as exc:
            logger.warning("Redis cache clear failed: %s", exc)
            if self.raise_on_error:
                raise CacheServiceUnavailableError(
                    message="Redis cache service is unreachable",
                    detail=f"Failed clearing keys: {exc}",
                ) from exc
            return False

    async def check_health(self) -> bool:
        """Probe Redis health via ping.

        Returns:
            True if Redis responds to ping, False otherwise.
        """
        try:
            result = await self._client.ping()
            return bool(result)
        except Exception:
            return False

    async def aclose(self) -> None:
        """Close Redis connection if not externally injected."""
        self._l1_cache.clear()
        if not self._injected_client:
            await self._client.aclose()

    async def __aenter__(self) -> "RedisQueryCache":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager and close client."""
        await self.aclose()


# Architectural alias matching ADRs and documentation
CacheService = RedisQueryCache
