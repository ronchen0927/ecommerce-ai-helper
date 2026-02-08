"""FastAPI application factory and configuration."""
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.auth import check_rate_limit, verify_api_key
from app.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name}...")
    logger.info(f"Auth enabled: {settings.auth_enabled}")
    logger.info(f"Rate limiting enabled: {settings.rate_limit_enabled}")
    logger.info(f"Storage type: {settings.storage_type}")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="AI-powered e-commerce product photo optimization service",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication and rate limiting middleware
    @app.middleware("http")
    async def auth_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Middleware for authentication and rate limiting."""
        # Skip auth for health check and docs
        skip_paths = ["/health", "/docs", "/redoc", "/openapi.json"]
        if request.url.path in skip_paths:
            return await call_next(request)

        try:
            # Verify API key
            api_key = await verify_api_key(request)

            # Check rate limit
            await check_rate_limit(request, api_key)

            # Add API key to request state for downstream use
            request.state.api_key = api_key

            response = await call_next(request)

            # Add rate limit headers
            if settings.rate_limit_enabled:
                response.headers["X-RateLimit-Limit"] = str(
                    settings.rate_limit_requests
                )

            return response

        except Exception as e:
            # Re-raise HTTPExceptions (they have proper status codes)
            from fastapi import HTTPException
            if isinstance(e, HTTPException):
                raise
            # Log and re-raise other exceptions
            logger.error(f"Auth middleware error: {e}")
            raise

    # Include routers
    app.include_router(api_router)

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint (no auth required)."""
        return {"status": "healthy"}

    return app


app = create_app()
