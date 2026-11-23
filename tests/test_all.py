import pytest
import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import db
import api
from ml.features import extract_features, FEATURE_NAMES
from ml.predict import predict_risk
from ml.anomaly import detector
from ml.forecast import forecast_power
from quantum.scheduler import optimize, DEFAULT_JOBS
from alerts import check_and_trigger_alerts
import pipeline

def test_db_init_and_tables():
    db.init_db()
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        assert "readings" in tables
        assert "events" in tables
        assert "risk_scores" in tables
        assert "schedules" in tables
        assert "alerts" in tables

def test_api_post_reading_with_source():
    db.init_db()
    with api.app.test_client() as client:
        payload = {
            "machine_id": "M1",
            "temperature": 62.0,
            "vibration": 0.35,
            "power": 3.4,
            "source": "esp32-sim",
            "timestamp": time.time()
        }
        res = client.post("/sensor-data", json=payload)
        assert res.status_code == 201
        data = res.get_json()
        assert data["ok"] is True
        assert data["source"] == "esp32-sim"

def test_feature_extraction():
    dummy = pd.DataFrame({
        "temperature": np.linspace(60, 65, 30),
        "vibration": np.linspace(0.3, 0.4, 30),
        "power": np.linspace(3.5, 3.5, 30)
    })
    feats = extract_features(dummy)
    assert feats is not None
    assert list(feats.columns) == FEATURE_NAMES
    assert len(feats) == 1

def test_ml_predict_risk():
    healthy_df = pd.DataFrame({
        "temperature": np.random.normal(60.0, 0.5, 30),
        "vibration": np.random.normal(0.35, 0.02, 30),
        "power": np.random.normal(3.5, 0.05, 30)
    })
    healthy_res = predict_risk(healthy_df)
    assert healthy_res["risk"] < 0.45

    degraded_df = pd.DataFrame({
        "temperature": np.linspace(78.0, 95.0, 30),
        "vibration": np.linspace(0.9, 2.2, 30),
        "power": np.linspace(3.8, 4.5, 30)
    })
    degraded_res = predict_risk(degraded_df)
    assert degraded_res["risk"] > 0.65

def test_anomaly_and_forecasting():
    df = pd.DataFrame({
        "temperature": np.linspace(58, 62, 25),
        "vibration": np.linspace(0.3, 0.4, 25),
        "power": np.linspace(3.2, 3.6, 25)
    })
    detector.fit_machine("M1", df)
    score_normal = detector.score("M1", {"temperature": 60.0, "vibration": 0.35, "power": 3.4})
    score_spike = detector.score("M1", {"temperature": 99.0, "vibration": 3.0, "power": 8.0})
    assert score_spike > score_normal

    preds = forecast_power(df, steps_ahead=10)
    assert len(preds) == 10
    assert all(p > 0 for p in preds)

def test_quantum_scheduler_and_offline():
    res = optimize(jobs=DEFAULT_JOBS, risk_by_machine={"M1": 0.05, "M2": 0.1, "M3": 0.95}, offline_machines=["M3"])
    assignment = res["assignment"]
    for j in DEFAULT_JOBS:
        assert assignment[j["id"]] in ["M1", "M2"]
        assert assignment[j["id"]] != "M3"

def test_alerts_cooldown():
    # First trigger should succeed
    res1 = check_and_trigger_alerts("M3", risk=0.92, top_factor="Test Overheat")
    assert res1 is not None
    # Immediate second trigger should be suppressed by cooldown
    res2 = check_and_trigger_alerts("M3", risk=0.95, top_factor="Test Overheat")
    assert res2 is None

def test_pipeline_step():
    step = pipeline.run_pipeline_step({})
    assert "risks" in step
    assert "recomputed" in step
