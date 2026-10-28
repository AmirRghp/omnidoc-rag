"""FastAPI dependency injection providers for settings, workflows, and security."""

import secrets
from typing import Annotated

from fastapi import Depends, Header, Request

from src.common.config import Settings, create_settings
from src.common.exceptions import AuthenticationError
from src.graph.workflow import CRAGWorkflow

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
        _cached_workflow = CRAGWorkflow(settings=settings)
    return _cached_workflow


def reset_crag_workflow() -> None:
    """Reset the cached CRAGWorkflow singleton (primarily for testing)."""
    global _cached_workflow
    _cached_workflow = None


async def verify_api_key(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    """Validate Bearer API token against configured environment secrets.

    Args:
        request: Incoming FastAPI request instance.
        authorization: Raw Authorization header value.
        settings: Application settings containing api_key.

    Returns:
        Validated API key string.

    Raises:
        AuthenticationError: If Authorization header is missing, malformed,
            or does not match the configured API_KEY.
    """
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

    provided_token = parts[1]
    expected_token = settings.api_key

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(provided_token, expected_token):
        raise AuthenticationError(
            message="Invalid API key",
            detail="Provided Bearer token is invalid or unauthorized.",
        )

    return provided_token
