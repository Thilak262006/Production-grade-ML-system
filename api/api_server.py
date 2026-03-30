"""
FastAPI Application Entry Point.
Brings everything together:
- Loads artifacts at startup
- Registers middleware (auth, rate limiting)
- Registers routes (predict, health)
"""

import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.routes.prediction_route import router as prediction_router
from api.routes.health_route import router as health_router
from api.middleware.auth_middleware import AuthMiddleware
from api.middleware.rate_limiter import limiter, rate_limit_exceeded_handler
from api.utils.model_loader import load_all_artifacts
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="api_requests.log")

# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer will churn based on their profile.",
    version="1.0.0",
)

# ── Attach rate limiter to app ────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth Middleware ───────────────────────────────────────────────────────────
app.add_middleware(AuthMiddleware)

# ── Register Routes ───────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(prediction_router)


# ── Startup Event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load all ML artifacts into memory when API starts."""
    logger.info("API starting up — loading artifacts...")
    load_all_artifacts()
    logger.info("API ready!")


# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }