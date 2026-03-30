import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import yaml

from src.utils.exception import ChurnModelException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def read_yaml(path: str | Path) -> dict:
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")
        with open(path, "r") as f:
            content = yaml.safe_load(f)
        if content is None:
            logger.warning(f"YAML file is empty: {path}")
            return {}
        logger.debug(f"Loaded YAML config from: {path}")
        return content
    except Exception as e:
        raise ChurnModelException(e, sys) from e


def save_json(path: str | Path, data: dict) -> None:
    try:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved → {path}")
    except Exception as e:
        raise ChurnModelException(e, sys) from e


def load_json(path: str | Path) -> dict:
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, "r") as f:
            content = json.load(f)
        logger.debug(f"Loaded JSON from: {path}")
        return content
    except Exception as e:
        raise ChurnModelException(e, sys) from e


def ensure_dir(path: str | Path) -> None:
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")
    except Exception as e:
        raise ChurnModelException(e, sys) from e


def save_object(path: str | Path, obj: Any) -> None:
    try:
        path = Path(path)
        ensure_dir(path.parent)
        joblib.dump(obj, path)
        logger.info(f"Object saved → {path}  [{get_size_in_kb(path):.1f} KB]")
    except Exception as e:
        raise ChurnModelException(e, sys) from e


def load_object(path: str | Path) -> Any:
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Object file not found: {path}")
        obj = joblib.load(path)
        logger.info(f"Object loaded ← {path}")
        return obj
    except Exception as e:
        raise ChurnModelException(e, sys) from e


def get_size_in_kb(path: str | Path) -> float:
    return os.path.getsize(path) / 1024