"""Domain exceptions conforming to RFC 7807 Problem Details."""

from datetime import datetime, timezone
from typing import Any

UTC = timezone.utc


class BaseRAGException(Exception):
    """Base exception for all OmniDoc RAG domain errors."""

    def __init__(
        self,
        message: str,
        code: str = "ERR_INTERNAL_ERROR",
        status_code: int = 500,
        title: str = "Internal Server Error",
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.title = title
        self.detail = detail or message

    def to_problem_details(self, instance: str = "") -> dict[str, Any]:
        """Convert exception to RFC 7807 Problem Details specification dictionary."""
        return {
            "type": f"https://api.omnidoc.local/errors/{self.code.removeprefix('ERR_')}",
            "title": self.title,
            "status": self.status_code,
            "detail": self.detail,
            "instance": instance,
            "code": self.code,
            "timestamp": datetime.now(UTC).isoformat(),
        }


class DocumentParsingError(BaseRAGException):
    """Raised when document ingestion, parsing, or markdown chunking fails."""

    def __init__(
        self,
        message: str = "Failed to parse document or extract headers",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ERR_DOCUMENT_PARSING",
            status_code=422,
            title="Document Parsing Failed",
            detail=detail,
        )


class VectorStoreTimeoutError(BaseRAGException):
    """Raised when Qdrant search, indexing, or snapshot operation times out."""

    def __init__(
        self,
        message: str = "Vector database request timed out",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ERR_QDRANT_TIMEOUT",
            status_code=504,
            title="Vector Database Gateway Timeout",
            detail=detail,
        )


class InferenceEngineUnavailableError(BaseRAGException):
    """Raised when local vLLM server is unreachable, drops connection, or times out."""

    def __init__(
        self,
        message: str = "Inference engine is currently unavailable",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ERR_VLLM_CONNECTION_FAILED",
            status_code=503,
            title="Inference Engine Unreachable",
            detail=detail,
        )


class InvalidInputSecurityError(BaseRAGException):
    """Raised when user input violates security checks (length, injection, unicode)."""

    def __init__(
        self,
        message: str = "Input rejected by security guardrails",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ERR_SECURITY_INPUT_REJECTED",
            status_code=400,
            title="Security Guardrail Violation",
            detail=detail,
        )


class CacheServiceUnavailableError(BaseRAGException):
    """Raised when Redis caching operations fail or disconnect."""

    def __init__(
        self,
        message: str = "Redis cache service is unreachable",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ERR_REDIS_UNAVAILABLE",
            status_code=503,
            title="Cache Service Unavailable",
            detail=detail,
        )


class ConfigurationError(BaseRAGException):
    """Raised when critical configuration settings are missing or invalid."""

    def __init__(
        self,
        message: str = "Invalid system configuration",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ERR_CONFIGURATION_INVALID",
            status_code=500,
            title="Configuration Error",
            detail=detail,
        )


class AuthenticationError(BaseRAGException):
    """Raised when authentication credentials are missing or invalid."""

    def __init__(
        self,
        message: str = "Invalid or missing authentication credentials",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ERR_AUTHENTICATION_FAILED",
            status_code=401,
            title="Authentication Failed",
            detail=detail,
        )


class AuthorizationError(BaseRAGException):
    """Raised when an authenticated caller lacks permissions for an admin endpoint."""

    def __init__(
        self,
        message: str = "Insufficient permissions for this operation",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ERR_FORBIDDEN",
            status_code=403,
            title="Forbidden",
            detail=detail,
        )


class RateLimitExceededError(BaseRAGException):
    """Raised when client IP exceeds sliding-window request rate limits."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ERR_RATE_LIMIT_EXCEEDED",
            status_code=429,
            title="Too Many Requests",
            detail=detail,
        )
