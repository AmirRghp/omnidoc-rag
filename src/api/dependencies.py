"""FastAPI dependency injection providers for settings, workflows, and security."""

import hashlib
import secrets
from typing import Annotated

from fastapi import Depends, Header, Request

from src.common.config import Settings, create_settings
from src.common.exceptions import AuthenticationError, AuthorizationError
from src.graph.workflow import CRAGWorkflow
from src.services.cache import RedisQueryCache

_cached_workflow: CRAGWorkflow | None = None


def get_settings() -> Settings:
    """Provide the application settings instance."""
    return create_settings()


def get_crag_workflow(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CRAGWorkflow:
    """Provide a singleton CRAGWorkflow instance for request handlers."""
    global _cached_workflow
    if _cached_workflow is None:
        cache = RedisQueryCache(settings=settings) if settings.semantic_cache_enabled else None
        _cached_workflow = CRAGWorkflow(settings=settings, cache=cache)
    return _cached_workflow


def reset_crag_workflow() -> None:
    """Reset the cached CRAGWorkflow singleton (primarily for testing)."""
    global _cached_workflow
    _cached_workflow = None


def _extract_bearer_token(authorization: str | None) -> str:
    """Validate and extract the Bearer token from the Authorization header."""
    if not authorization:
        raise AuthenticationError(
            message="Missing authorization header",
            detail="The 'Authorization: Bearer <API_KEY>' header is required for this endpoint.",
        )

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthenticationError(
            message="Malformed authorization header",
            detail="Authorization header must follow the 'Bearer <API_KEY>' scheme.",
        )

    return parts[1]


def _token_matches_secret(
    provided_token: str,
    expected_token: str,
    expected_sha256: str | None = None,
) -> bool:
    """Compare a provided Bearer token against plaintext and optional SHA-256 digest."""
    if expected_sha256:
        provided_digest = hashlib.sha256(provided_token.encode("utf-8")).hexdigest().lower()
        if secrets.compare_digest(provided_digest, expected_sha256.strip().lower()):
            return True
    return secrets.compare_digest(provided_token, expected_token)


async def verify_api_key(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    """Validate Bearer API token against configured environment secrets.

    Args:
        request: Incoming FastAPI request instance.
        settings: Application settings containing api_key / api_key_sha256.
        authorization: Raw Authorization header value.

    Returns:
        Validated API key string.

    Raises:
        AuthenticationError: If Authorization header is missing, malformed,
            or does not match the configured API_KEY.
    """
    provided_token = _extract_bearer_token(authorization)

    is_valid_user = _token_matches_secret(
        provided_token=provided_token,
        expected_token=settings.api_key,
        expected_sha256=settings.api_key_sha256,
    )
    is_valid_admin = bool(
        settings.admin_api_key
        and secrets.compare_digest(provided_token, settings.admin_api_key)
    )

    if not (is_valid_user or is_valid_admin):
        raise AuthenticationError(
            message="Invalid API key",
            detail="Provided Bearer token is invalid or unauthorized.",
        )

    return provided_token


async def verify_admin_api_key(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    """Validate Bearer API token with Role-Based Access Control (Admin required when configured).

    If `settings.admin_api_key` is set, only the admin key is permitted (standard API key
    receives 403 Forbidden). When `settings.admin_api_key` is not configured, falls back to
    standard API key authentication for seamless local development.
    """
    provided_token = await verify_api_key(
        request=request,
        settings=settings,
        authorization=authorization,
    )

    if settings.admin_api_key and not secrets.compare_digest(
        provided_token, settings.admin_api_key
    ):
        raise AuthorizationError(
            message="Admin privileges required",
            detail="This endpoint requires an administrative API token.",
        )

    return provided_token

