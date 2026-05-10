import logging
import os

import joblib

from app.core.config import settings

logger = logging.getLogger(__name__)
_models: dict = {}


def load_all_models() -> None:
    model_dir = settings.model_dir
    for name, filename in [
        ("anomaly", settings.anomaly_model_file),
        ("recommend", settings.recommend_model_file),
    ]:
        path = os.path.join(model_dir, filename)
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
