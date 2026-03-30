"""
Model Loader — loads artifacts once at startup and caches them.
This means the model is loaded ONCE when API starts,
not on every request (which would be very slow).
"""

import sys
from src.utils.common import load_object, read_yaml
from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException

logger = get_logger(__name__, log_file="api_requests.log")

# Module-level cache — loaded once at startup
_model = None
_transformer = None
_label_encoder = None
_config = None


def load_all_artifacts():
    """Load all artifacts into memory at API startup."""
    global _model, _transformer, _label_encoder, _config
    try:
        _config = read_yaml("configs/config.yaml")
        _model = load_object(_config["artifacts"]["model_path"])
        _transformer = load_object(_config["artifacts"]["transformer_path"])
        _label_encoder = load_object(_config["artifacts"]["label_encoder_path"])
        logger.info("All artifacts loaded into memory at startup")
    except Exception as e:
        raise ChurnModelException(e, sys) from e


def get_model():
    return _model

def get_transformer():
    return _transformer

def get_label_encoder():
    return _label_encoder

def get_config():
    return _config