"""PCA noise-filter helper used by the runtime pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None


MODEL_PATH = Path(__file__).resolve().parent / "models" / "pca.pkl"
pca = joblib.load(MODEL_PATH) if joblib is not None and MODEL_PATH.exists() else None


def transform(x: np.ndarray) -> np.ndarray:
    """Return PCA-filtered features, or a pass-through vector if no model exists."""
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if pca is not None:
        return pca.transform(X)
    return X
