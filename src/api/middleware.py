"""FastAPI HTTP middlewares for trace ID propagation and sliding-window rate limiting."""

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

import redis.asyncio as aioredis
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.api.errors import build_problem_details
from src.common.config import Settings, create_settings
from src.common.exceptions import RateLimitExceededError
from src.common.logger import clear_trace_id, get_logger, set_trace_id

logger = get_logger(__name__)


class TraceIDMiddleware(BaseHTTPMiddleware):
    """Middleware that extracts or generates X-Trace-ID and cleans up contextvars per request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Bind trace_id to request.state and contextvar, then guarantee cleanup on exit."""
        raw_header = request.headers.get("X-Trace-ID") or request.headers.get("x-trace-id")
        trace_id = (
            raw_header.strip()
            if raw_header and raw_header.strip()
            else f"tr_{uuid.uuid4().hex[:12]}"
        )
        request.state.trace_id = trace_id
        set_trace_id(trace_id)
        try:
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            clear_trace_id()


class SlidingWindowRateLimiter:
    """Two-tier sliding-window rate limiter backed by Redis with in-memory fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        redis_client: aioredis.Redis | None = None,
    ) -> None:
        """Initialize the sliding-window rate limiter.

        Args:
            settings: Application settings containing rate limit thresholds.
            redis_client: Optional async Redis client for distributed rate limiting.
        """
        self.settings = settings or create_settings()
        self._redis = redis_client
        self._memory_windows: dict[str, deque[float]] = defaultdict(deque)

    def _check_memory_window(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """Check and update in-memory sliding window for key.

        Returns:
            Tuple of (allowed, retry_after_seconds).
        """
        now = time.monotonic()
        cutoff = now - window_seconds
        window = self._memory_windows[key]

        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= limit:
            retry_after = max(1, int(window[0] + window_seconds - now))
            return False, retry_after

        window.append(now)
        return True, 0

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> tuple[bool, int]:
        """Evaluate whether a request under `key` is within `limit` per `window_seconds`.

        Uses Redis atomic counter with TTL when Redis is injected/reachable, and
        falls back gracefully to in-memory sliding window if Redis is unavailable.
        """
        if self._redis is not None:
            try:
                current_count = await self._redis.incr(key)
                if current_count == 1:
                    await self._redis.expire(key, window_seconds)
                if int(current_count) > limit:
                    ttl = await self._redis.ttl(key)
                    retry_after = max(1, int(ttl)) if ttl and int(ttl) > 0 else window_seconds
                    return False, retry_after
                return True, 0
            except Exception as exc:
                logger.warning("Redis rate limiter unavailable (%s); using in-memory window", exc)

        return self._check_memory_window(key=key, limit=limit, window_seconds=window_seconds)

    def reset(self) -> None:
        """Clear all in-memory rate limit windows (primarily for testing)."""
        self._memory_windows.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware enforcing per-IP sliding-window rate limits (RFC 7807 429 on breach)."""

    def __init__(
        self,
        app: ASGIApp,
        settings: Settings | None = None,
        limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        super().__init__(app)
        self.settings = settings or create_settings()
        self.limiter = limiter or SlidingWindowRateLimiter(settings=self.settings)

    @staticmethod
    def _extract_client_ip(request: Request) -> str:
        """Resolve client IP from X-Forwarded-For or connection peer."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            first_ip = forwarded.split(",")[0].strip()
            if first_ip:
                return first_ip
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Enforce 30 req/min general limit and 5 uploads/hour on POST /api/v1/ingest."""
        path = request.url.path
        if path in {"/health", "/api/v1/health", "/docs", "/redoc", "/openapi.json"}:
            return await call_next(request)

        client_ip = self._extract_client_ip(request)
        trace_id = getattr(request.state, "trace_id", None) or f"tr_{uuid.uuid4().hex[:12]}"

        # 1. Check general per-IP rate limit (default: 30 requests / minute)
        ip_key = f"ratelimit:ip:{client_ip}"
        allowed, retry_after = await self.limiter.is_allowed(
            key=ip_key,
            limit=self.settings.rate_limit_per_minute,
            window_seconds=60,
        )
        if not allowed:
            exc = RateLimitExceededError(
                message="Rate limit exceeded",
                detail=(
                    f"Client IP {client_ip} exceeded the maximum rate of "
                    f"{self.settings.rate_limit_per_minute} requests per minute."
                ),
            )
            payload = build_problem_details(exc=exc, instance=path, trace_id=trace_id)
            return JSONResponse(
                status_code=429,
                content=payload,
                headers={"X-Trace-ID": trace_id, "Retry-After": str(retry_after)},
            )

        # 2. Check stricter upload rate limit on POST /api/v1/ingest (default: 5 uploads / hour)
        if request.method.upper() == "POST" and path.rstrip("/") == "/api/v1/ingest":
            upload_key = f"ratelimit:upload:{client_ip}"
            upload_allowed, upload_retry = await self.limiter.is_allowed(
                key=upload_key,
                limit=self.settings.upload_rate_limit_per_hour,
                window_seconds=3600,
            )
            if not upload_allowed:
                exc = RateLimitExceededError(
                    message="Ingestion upload rate limit exceeded",
                    detail=(
                        f"Client IP {client_ip} exceeded the maximum rate of "
                        f"{self.settings.upload_rate_limit_per_hour} document uploads per hour."
                    ),
                )
                payload = build_problem_details(exc=exc, instance=path, trace_id=trace_id)
                return JSONResponse(
                    status_code=429,
                    content=payload,
                    headers={"X-Trace-ID": trace_id, "Retry-After": str(upload_retry)},
                )

        return await call_next(request)
