"""
Rate Limiting — prevents API abuse.
Default: max 100 requests per minute per IP address.
Uses slowapi which is built on top of limits library.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="api_requests.log")

# Create limiter instance — identifies users by IP address
limiter = Limiter(key_func=get_remote_address)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom response when rate limit is exceeded."""
    logger.warning(f"Rate limit exceeded | IP={get_remote_address(request)}")
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Max 100 requests per minute.",
            "retry_after": "60 seconds"
        }
    )