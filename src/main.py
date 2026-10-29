"""FastAPI application entrypoint, lifecycle management, and middleware setup."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import get_settings
from src.api.errors import register_exception_handlers
from src.api.middleware import RateLimitMiddleware, TraceIDMiddleware
from src.api.routes import router as api_router
from src.common.config import Settings, create_settings
from src.common.logger import configure_logger, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        settings: Optional Settings instance. If omitted, loaded from environment/YAML.

    Returns:
        Configured FastAPI application instance ready for serving.
    """
    app_settings = settings or create_settings()
    configure_logger(log_level="INFO", json_format=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage startup and shutdown events for the application."""
        logger.info(
            "application_startup",
            environment=app_settings.environment,
            host=app_settings.app_host,
            port=app_settings.app_port,
        )
        yield
        logger.info("application_shutdown")

    app = FastAPI(
        title="OmniDoc RAG API",
        version="0.1.0",
        description="Robust Bilingual Technical RAG & Evaluation System API",
        docs_url="/docs" if app_settings.environment != "production" else None,
        redoc_url="/redoc" if app_settings.environment != "production" else None,
        lifespan=lifespan,
    )

    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: app_settings

    # Register RFC 7807 global exception handlers
    register_exception_handlers(app)

    # Register Trace ID cleanup and sliding-window rate limit middlewares
    app.add_middleware(RateLimitMiddleware, settings=app_settings)
    app.add_middleware(TraceIDMiddleware)

    # Configure CORS policy based on environment
    allowed_origins = (
        ["*"]
        if app_settings.environment == "development"
        else [
            f"http://{app_settings.app_host}:{app_settings.app_port}",
            "http://localhost:8000",
            "http://localhost:8080",
        ]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API v1 routes
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        """Service health check endpoint."""
        return {
            "status": "healthy",
            "environment": app_settings.environment,
            "version": "0.1.0",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    server_settings = create_settings()
    uvicorn.run(
        "src.main:app",
        host=server_settings.app_host,
        port=server_settings.app_port,
        reload=server_settings.environment == "development",
    )
