"""
Health Check Route — GET /health
Used by Docker, load balancers, and monitoring tools
to check if the API is running correctly.
"""

from fastapi import APIRouter
from api.utils.model_loader import get_model, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="api_requests.log")

router = APIRouter()


@router.get("/health")
def health_check():
    """
    Returns API status and whether the model is loaded.
    No authentication required for this endpoint.
    """
    model = get_model()
    config = get_config()

    status = "healthy" if model is not None else "unhealthy"
    model_name = type(model).__name__ if model is not None else "not loaded"

    logger.info(f"Health check → status={status}")

    return {
        "status"     : status,
        "model"      : model_name,
        "project"    : config["project"]["name"] if config else "unknown",
        "version"    : config["project"]["version"] if config else "unknown",
    }