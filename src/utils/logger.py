import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import yaml

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOGGING_CONFIG_PATH = os.path.join(os.getcwd(), "configs", "logging.yaml")


def _get_log_level(level_str: str) -> int:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_str.upper(), logging.INFO)


def _load_logging_config() -> dict:
    defaults = {
        "level": "INFO",
        "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "max_bytes": 10_485_760,
        "backup_count": 5,
    }
    if os.path.exists(LOGGING_CONFIG_PATH):
        with open(LOGGING_CONFIG_PATH, "r") as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
    return defaults


_configured_loggers: set = set()
_config = _load_logging_config()


def get_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(_get_log_level(_config["level"]))

    formatter = logging.Formatter(
        fmt=_config["format"],
        datefmt=_config["datefmt"],
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = os.path.join(LOG_DIR, log_file)
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=_config["max_bytes"],
        backupCount=_config["backup_count"],
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    _configured_loggers.add(name)
    return logger