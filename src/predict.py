"""
Loads trained model + scaler + feature list, and predicts placement.
"""
import os
import joblib
import numpy as np

MODEL_PATH    = "models/placement_model.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_PATH = "models/features.pkl"

_model, _scaler, _features = None, None, None


def _load():
    global _model, _scaler, _features
    if _model is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
            raise FileNotFoundError(
                "❌ Model not found. Please run:\n"
                "   python -m src.train_model"
            )
        _model    = joblib.load(MODEL_PATH)
        _scaler   = joblib.load(SCALER_PATH)
        _features = joblib.load(FEATURES_PATH)


def predict_placement(input_dict):
    """
    input_dict: {feature_name: value, ...}
    Returns: {"placed": bool, "confidence": float}
    """
    _load()

    # Build row in same order as training features
    row = []
    for feat in _features:
        val = input_dict.get(feat, 0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        row.append(val)

    X = np.array([row])
    X_scaled = _scaler.transform(X)

    pred = _model.predict(X_scaled)[0]
    proba = _model.predict_proba(X_scaled)[0]

    return {
        "placed":     bool(pred),
        "confidence": round(float(max(proba)) * 100, 2),
    }