import logging
import os

import joblib

from app.core import config

logger = logging.getLogger(__name__)
_models: dict = {}


def load_all_models() -> None:
    for name, filename in [
        ("anomaly", config.ANOMALY_MODEL_FILE),
        ("recommend", config.RECOMMEND_MODEL_FILE),
    ]:
        path = os.path.join(config.MODEL_DIR, filename)
        if os.path.exists(path):
            _models[name] = joblib.load(path)
            logger.info("Loaded model: %s from %s", name, path)
        else:
            logger.warning("Model file not found, skipping: %s", path)
            _models[name] = None


def get_model(name: str):
    model = _models.get(name)
    if model is None:
        raise RuntimeError(f"Model '{name}' is not loaded.")
    return model
