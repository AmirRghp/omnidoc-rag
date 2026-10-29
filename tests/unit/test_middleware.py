"""Unit tests for TraceIDMiddleware, RateLimitMiddleware, RBAC, and new API endpoints (Task 8.2)."""

import hashlib
import io

import httpx
import pytest
from fastapi import FastAPI

from src.common.config import Settings
from src.common.logger import get_trace_id
from src.main import create_app

USER_KEY = "user_secret_token_123"
ADMIN_KEY = "admin_secret_token_456"


@pytest.fixture
def rbac_app() -> FastAPI:
    """Create a FastAPI app configured with user key, admin key, and strict rate limits."""
    settings = Settings(
        ENVIRONMENT="testing",
        API_KEY=USER_KEY,
        ADMIN_API_KEY=ADMIN_KEY,
        RATE_LIMIT_PER_MINUTE=3,
        UPLOAD_RATE_LIMIT_PER_HOUR=2,
    )
    return create_app(settings=settings)


@pytest.mark.asyncio
async def test_trace_id_middleware_propagates_and_cleans_contextvar(rbac_app: FastAPI) -> None:
    """TraceIDMiddleware injects X-Trace-ID on response and clears contextvar on exit."""
    transport = httpx.ASGITransport(app=rbac_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/health", headers={"X-Trace-ID": "tr_custom_99"})
        assert res.status_code == 200
        assert res.headers.get("X-Trace-ID") == "tr_custom_99"
        assert res.json()["version"] == "0.1.0"

    # Contextvar must be cleaned after request lifecycle completes
    assert get_trace_id() is None


@pytest.mark.asyncio
async def test_rate_limit_middleware_returns_429_rfc7807(rbac_app: FastAPI) -> None:
    """Exceeding rate_limit_per_minute (3 req/min) returns HTTP 429 Problem Details."""
    headers = {"Authorization": f"Bearer {USER_KEY}"}
    payload = {"trace_id": "tr_rl_1", "rating": "up", "score": 5}

    transport = httpx.ASGITransport(app=rbac_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            ok_res = await client.post("/api/v1/feedback", json=payload, headers=headers)
            assert ok_res.status_code == 200

        blocked_res = await client.post("/api/v1/feedback", json=payload, headers=headers)
        assert blocked_res.status_code == 429
        assert "Retry-After" in blocked_res.headers
        body = blocked_res.json()
        assert body["status"] == 429
        assert body["code"] == "ERR_RATE_LIMIT_EXCEEDED"


@pytest.mark.asyncio
async def test_sha256_api_key_authentication() -> None:
    """verify_api_key accepts a Bearer token matching settings.api_key_sha256."""
    raw_token = "high_entropy_production_token"
    digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    settings = Settings(
        ENVIRONMENT="testing",
        API_KEY="unused_placeholder_key",
        API_KEY_SHA256=digest,
    )
    app = create_app(settings=settings)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/v1/feedback",
            json={"trace_id": "tr_sha_1", "rating": "up", "score": 4},
            headers={"Authorization": f"Bearer {raw_token}"},
        )
        assert res.status_code == 200


@pytest.mark.asyncio
async def test_admin_rbac_and_ingestion_endpoints(rbac_app: FastAPI) -> None:
    """POST /api/v1/ingest enforces Admin RBAC (403 for user key, 202 for admin key)."""
    transport = httpx.ASGITransport(app=rbac_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        md_bytes = b"# Docker Guide\nUse `docker run`."
        files = {"file": ("test_guide.md", io.BytesIO(md_bytes), "text/markdown")}

        # Standard user key receives 403 Forbidden on admin ingestion endpoint
        forbidden = await client.post(
            "/api/v1/ingest",
            files=files,
            headers={"Authorization": f"Bearer {USER_KEY}"},
        )
        assert forbidden.status_code == 403
        assert forbidden.json()["code"] == "ERR_FORBIDDEN"

        # Admin key succeeds with 202 Accepted
        files_admin = {
            "file": ("test_guide.md", io.BytesIO(md_bytes), "text/markdown")
        }
        accepted = await client.post(
            "/api/v1/ingest",
            files=files_admin,
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        )
        assert accepted.status_code == 202
        job_data = accepted.json()
        job_id = job_data["job_id"]
        assert job_data["filename"] == "test_guide.md"

        # Poll status endpoint
        status_res = await client.get(
            f"/api/v1/ingest/status/{job_id}",
            headers={"Authorization": f"Bearer {USER_KEY}"},
        )
        assert status_res.status_code == 200
        assert status_res.json()["job_id"] == job_id
