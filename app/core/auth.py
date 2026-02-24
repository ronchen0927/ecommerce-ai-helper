"""Authentication and authorization module."""

import hashlib
import secrets
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, APIKeyQuery
from jose import JWTError, jwt
from pydantic import BaseModel

from app.core.config import get_settings

# API Key security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


class TokenData(BaseModel):
    """JWT token payload data."""

    sub: str  # Subject (user ID or API key identifier)
    exp: int  # Expiration timestamp
    scopes: list[str] = []  # Permission scopes


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self) -> None:
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(
        self, key: str, max_requests: int, window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check if request is allowed under rate limit.

        Args:
            key: Identifier for rate limiting (e.g., API key or IP).
            max_requests: Maximum requests allowed in window.
            window_seconds: Time window in seconds.

        Returns:
            Tuple of (is_allowed, remaining_requests).
        """
        now = time.time()
        window_start = now - window_seconds

        # Clean old requests
        self._requests[key] = [ts for ts in self._requests[key] if ts > window_start]

        current_count = len(self._requests[key])

        if current_count >= max_requests:
            return False, 0

        # Add current request
        self._requests[key].append(now)
        return True, max_requests - current_count - 1

    def get_reset_time(self, key: str, window_seconds: int) -> int:
        """Get seconds until rate limit resets."""
        if not self._requests[key]:
            return 0
        oldest = min(self._requests[key])
        reset_at = oldest + window_seconds
        return max(0, int(reset_at - time.time()))


# Global rate limiter instance
rate_limiter = RateLimiter()


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure comparison."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a new secure API key."""
    return secrets.token_urlsafe(32)


async def get_api_key(
    api_key_header_value: Optional[str] = Security(api_key_header),
    api_key_query_value: Optional[str] = Security(api_key_query),
) -> Optional[str]:
    """Extract API key from header or query parameter."""
    return api_key_header_value or api_key_query_value


async def verify_api_key(request: Request) -> str:
    """
    Verify API key from request.

    Raises:
        HTTPException: If API key is missing or invalid.
    """
    settings = get_settings()

    if not settings.auth_enabled:
        return "anonymous"

    # Get API key from header or query
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    valid_keys = settings.get_api_keys_list()
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key


async def check_rate_limit(request: Request, api_key: str) -> None:
    """
    Check rate limit for the request.

    Raises:
        HTTPException: If rate limit exceeded.
    """
    settings = get_settings()

    if not settings.rate_limit_enabled:
        return

    # Use API key or IP as rate limit key
    rate_key = (
        api_key
        if api_key != "anonymous"
        else request.client.host
        if request.client
        else "unknown"
    )

    is_allowed, remaining = rate_limiter.is_allowed(
        rate_key,
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    )

    if not is_allowed:
        reset_time = rate_limiter.get_reset_time(
            rate_key, settings.rate_limit_window_seconds
        )
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(settings.rate_limit_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_time),
                "Retry-After": str(reset_time),
            },
        )


def create_jwt_token(
    subject: str,
    scopes: list[str] | None = None,
    expires_minutes: int | None = None,
) -> str:
    """
    Create a JWT token.

    Args:
        subject: Token subject (user ID or identifier).
        scopes: Permission scopes.
        expires_minutes: Token expiration in minutes.

    Returns:
        Encoded JWT token string.
    """
    settings = get_settings()

    if not settings.jwt_secret_key:
        raise ValueError("JWT_SECRET_KEY not configured")

    expire_minutes = expires_minutes or settings.jwt_expire_minutes
    expire_time = int(time.time()) + (expire_minutes * 60)

    payload = {
        "sub": subject,
        "exp": expire_time,
        "scopes": scopes or [],
        "iat": int(time.time()),
    }

    return str(
        jwt.encode(
            payload,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
    )


def verify_jwt_token(token: str) -> TokenData:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string.

    Returns:
        TokenData with decoded payload.

    Raises:
        HTTPException: If token is invalid or expired.
    """
    settings = get_settings()

    if not settings.jwt_secret_key:
        raise HTTPException(status_code=500, detail="JWT not configured")

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return TokenData(
            sub=payload["sub"],
            exp=payload["exp"],
            scopes=payload.get("scopes", []),
        )
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def require_scopes(*required_scopes: str) -> Callable[..., Any]:
    """
    Decorator to require specific scopes for an endpoint.

    Usage:
        @app.get("/admin")
        @require_scopes("admin", "write")
        async def admin_endpoint():
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request: Optional[Request] = kwargs.get("request")
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if not request:
                raise HTTPException(status_code=500, detail="Request not found")

            # Get token from Authorization header
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=401,
                    detail="Missing Bearer token",
                )

            token = auth_header[7:]  # Remove "Bearer " prefix
            token_data = verify_jwt_token(token)

            # Check scopes
            for scope in required_scopes:
                if scope not in token_data.scopes:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Missing required scope: {scope}",
                    )

            return await func(*args, **kwargs)

        return wrapper

    return decorator
