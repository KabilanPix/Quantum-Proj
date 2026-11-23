import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from config import MODEL_PATH, RISK_HIGH, RISK_MED
from ml.features import extract_features, FEATURE_NAMES

# Module-level model cache
_MODEL = None
_FEATURE_NAMES = None

def _load_model():
    global _MODEL, _FEATURE_NAMES
    if _MODEL is None:
        if not MODEL_PATH.exists():
            print(f"[ML Predict] Model file {MODEL_PATH} not found. Training automatically...")
            from ml.train_model import train
            train()
        
        payload = joblib.load(MODEL_PATH)
        if isinstance(payload, dict):
            _MODEL = payload.get("model")
            _FEATURE_NAMES = payload.get("feature_names", FEATURE_NAMES)
        else:
            _MODEL = payload
            _FEATURE_NAMES = FEATURE_NAMES
    return _MODEL, _FEATURE_NAMES

# Baseline reference averages for normal healthy behavior to calculate deviation
BASELINE_MEANS = {
    "temp_mean": 60.0,
    "temp_std": 1.0,
    "temp_slope": 0.0,
    "vib_mean": 0.35,
    "vib_std": 0.04,
    "vib_slope": 0.0,
    "vib_max": 0.45,
    "power_mean": 3.5,
    "power_std": 0.1,
    "power_slope": 0.0
}

FRIENDLY_NAMES = {
    "temp_mean": "elevated temperature",
    "temp_std": "temperature volatility",
    "temp_slope": "rapidly rising temperature",
    "vib_mean": "high baseline vibration",
    "vib_std": "vibration instability",
    "vib_slope": "rising vibration trend",
    "vib_max": "excessive vibration peak",
    "power_mean": "abnormal power draw",
    "power_std": "power fluctuation",
    "power_slope": "surging power usage"
}

def predict_risk(machine_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Predicts machine failure risk:
    - Returns: {"risk": float 0-1, "level": "LOW|MEDIUM|HIGH", "top_factor": str}
    - If fewer than 5 readings, returns risk 0.0 and level "LOW", top_factor "insufficient data".
    """
    if machine_df is None or len(machine_df) < 5:
        return {
            "risk": 0.0,
            "level": "LOW",
            "top_factor": "normal operations (insufficient data)"
        }

    features_df = extract_features(machine_df)
    if features_df is None:
        return {
            "risk": 0.0,
            "level": "LOW",
            "top_factor": "normal operations"
        }

    model, feat_names = _load_model()
    X = features_df[feat_names]

    # Predict risk probability of class 1 (failure within 60 steps)
    probs = model.predict_proba(X)[0]
    classes = list(model.classes_)
    if 1 in classes:
        risk = float(probs[classes.index(1)])
    else:
        risk = 0.0

    # Determine risk level
    if risk >= RISK_HIGH:
        level = "HIGH"
    elif risk >= RISK_MED:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Compute top contributing factor (feature importance * positive deviation from baseline)
    importances = model.feature_importances_
    current_vals = X.iloc[0]
    
    impact_scores = {}
    for i, name in enumerate(feat_names):
        val = current_vals[name]
        base = BASELINE_MEANS.get(name, 0.0)
        # We care most about values exceeding baseline for degradation indicators
        diff = max(0.0, val - base)
        score = importances[i] * diff
        impact_scores[name] = score

    if max(impact_scores.values()) > 0:
        top_feature = max(impact_scores, key=impact_scores.get)
        top_factor = FRIENDLY_NAMES.get(top_feature, top_feature)
    else:
        top_factor = "stable nominal parameters"

    return {
        "risk": round(risk, 4),
        "level": level,
        "top_factor": top_factor
    }
