"""
API Key Authentication Middleware.
Every request must include a valid API key in the header:
    X-API-Key: your-secret-key

Requests without a valid key get a 401 Unauthorized response.
"""

import os
import sys
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__, log_file="api_requests.log")

# Paths that don't need authentication
PUBLIC_PATHS = ["/health", "/docs", "/openapi.json", "/redoc"]


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        expected_key = os.getenv("API_SECRET_KEY")

        if not api_key:
            logger.warning(f"Request rejected — no API key | path={request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing API key. Add X-API-Key header."}
            )

        if api_key != expected_key:
            logger.warning(f"Request rejected — invalid API key | path={request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid API key."}
            )

        logger.info(f"Authenticated request | path={request.url.path}")
        return await call_next(request)