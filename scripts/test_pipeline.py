import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import pandas as pd
import numpy as np

def run_acceptance_tests():
    print("=" * 65)
    print("       STARTING ACCEPTANCE TESTS FOR QUANTUM DIGITAL TWIN      ")
    print("=" * 65)
    results = {}

    # Test 1: config, db, api import without errors
    try:
        import config
        import db
        import api
        results["1. Imports (config, db, api)"] = "PASS"
    except Exception as e:
        results["1. Imports (config, db, api)"] = f"FAIL: {e}"

    # Test 2: API lifecycle and endpoints
    api_proc = None
    try:
        # Launch API as temporary subprocess
        api_proc = subprocess.Popen([sys.executable, "api.py"], cwd=str(PROJECT_ROOT))
        # Wait for health check
        health_ok = False
        for _ in range(20):
            try:
                r = requests.get(f"{config.API_URL}/health", timeout=1.0)
                if r.status_code == 200:
                    health_ok = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not health_ok:
            raise RuntimeError("API failed to respond to /health in time.")

        # Valid payload
        valid_payload = {
            "machine_id": "M1",
            "temperature": 60.5,
            "vibration": 0.35,
            "power": 3.6,
            "timestamp": time.time()
        }
        res_post = requests.post(f"{config.API_URL}/sensor-data", json=valid_payload, timeout=2.0)
        if res_post.status_code != 201:
            raise ValueError(f"POST /sensor-data expected 201, got {res_post.status_code}")

        # Bad payload (missing machine_id)
        bad_payload = {"temperature": "not_a_number"}
        res_bad = requests.post(f"{config.API_URL}/sensor-data", json=bad_payload, timeout=2.0)
        if res_bad.status_code != 400:
            raise ValueError(f"POST /sensor-data bad payload expected 400, got {res_bad.status_code}")

        # Latest readings
        res_latest = requests.get(f"{config.API_URL}/latest", timeout=2.0)
        if res_latest.status_code != 200 or len(res_latest.json().get("data", [])) == 0:
            raise ValueError(f"GET /latest failed or returned empty data: {res_latest.text}")

        results["2. API lifecycle & validation"] = "PASS"
    except Exception as e:
        results["2. API lifecycle & validation"] = f"FAIL: {e}"
    finally:
        if api_proc:
            api_proc.terminate()
            api_proc.wait()

    # Test 3: Feature extraction
    try:
        from ml.features import extract_features, FEATURE_NAMES
        dummy_df = pd.DataFrame({
            "temperature": np.linspace(60, 65, 30),
            "vibration": np.linspace(0.3, 0.4, 30),
            "power": np.linspace(3.5, 3.5, 30)
        })
        feats = extract_features(dummy_df)
        if feats is None or list(feats.columns) != FEATURE_NAMES or len(FEATURE_NAMES) != 10:
            raise ValueError(f"Extracted columns do not match FEATURE_NAMES: {feats.columns if feats is not None else None}")
        results["3. Feature extraction (exact 10 features)"] = "PASS"
    except Exception as e:
        results["3. Feature extraction (exact 10 features)"] = f"FAIL: {e}"

    # Test 4: Model training, F1 >= 0.80, and risk predictions
    try:
        from ml.train_model import train
        from ml.predict import predict_risk
        
        _, f1 = train()
        if f1 < 0.80:
            raise ValueError(f"F1 score {f1:.4f} is lower than 0.80")

        # Clearly healthy window
        healthy_df = pd.DataFrame({
            "temperature": np.random.normal(60.0, 0.5, 30),
            "vibration": np.random.normal(0.35, 0.02, 30),
            "power": np.random.normal(3.5, 0.05, 30)
        })
        healthy_risk = predict_risk(healthy_df)["risk"]
        if healthy_risk >= 0.4:
            raise ValueError(f"Healthy window risk too high: {healthy_risk}")

        # Clearly degraded window
        degraded_df = pd.DataFrame({
            "temperature": np.linspace(78.0, 94.0, 30),
            "vibration": np.linspace(0.9, 2.1, 30),
            "power": np.linspace(3.8, 4.4, 30)
        })
        degraded_risk = predict_risk(degraded_df)["risk"]
        if degraded_risk <= 0.7:
            raise ValueError(f"Degraded window risk too low: {degraded_risk}")

        results["4. Model train F1 >= 0.80 & risk differentiation"] = "PASS"
    except Exception as e:
        results["4. Model train F1 >= 0.80 & risk differentiation"] = f"FAIL: {e}"

    # Test 5: Quantum / Classical Scheduler
    try:
        from quantum.scheduler import optimize, DEFAULT_JOBS
        res_sched = optimize(jobs=DEFAULT_JOBS, risk_by_machine={"M1": 0.05, "M2": 0.15, "M3": 0.9})
        assignment = res_sched.get("assignment", {})
        
        # Every job must be assigned to exactly one machine
        job_ids = [j["id"] for j in DEFAULT_JOBS]
        for j_id in job_ids:
            if j_id not in assignment or assignment[j_id] not in ["M1", "M2", "M3"]:
                raise ValueError(f"Job {j_id} not properly assigned in {assignment}")
        
        results["5. Quantum/Classical Scheduler assignment validity"] = "PASS"
    except Exception as e:
        results["5. Quantum/Classical Scheduler assignment validity"] = f"FAIL: {e}"

    # Test 6: Dashboard syntax / compilation
    try:
        import py_compile
        py_compile.compile(str(PROJECT_ROOT / "dashboard.py"), doraise=True)
        results["6. Dashboard syntax compilation"] = "PASS"
    except Exception as e:
        results["6. Dashboard syntax compilation"] = f"FAIL: {e}"

    # Test 7: Extended checks (Real data model, benchmark artifacts, anomaly & pipeline step)
    try:
        if config.MODEL_REAL_PATH.exists() and config.REAL_METRICS_PATH.exists():
            results["7. Real AI4I 2020 ML Model Artifacts"] = "PASS"
        else:
            results["7. Real AI4I 2020 ML Model Artifacts"] = "WARNING (run ml/real_data/train_real.py)"

        from ml.anomaly import detector
        anom = detector.score("M1", {"temperature": 60.0, "vibration": 0.35, "power": 3.4})
        if 0.0 <= anom <= 1.0:
            results["8. Unsupervised Anomaly Detector"] = "PASS"
        else:
            raise ValueError(f"Anomaly score out of bounds: {anom}")

        import pipeline
        p_step = pipeline.run_pipeline_step({})
        if "risks" in p_step:
            results["9. Automated ML-to-Quantum Pipeline Step"] = "PASS"
        else:
            raise ValueError("Pipeline step returned invalid structure")
    except Exception as e:
        results["7-9. Extended Phase Checks"] = f"FAIL: {e}"

    # Print Summary Table
    print("\n" + "=" * 65)
    print("                    TEST SUMMARY RESULTS                       ")
    print("=" * 65)
    all_passed = True
    for test_name, status in results.items():
        print(f"[{status}] - {test_name}")
        if "FAIL" in status:
            all_passed = False
    print("=" * 65)

    if all_passed:
        print("[SUCCESS] ALL ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("[FAILURE] SOME ACCEPTANCE TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_acceptance_tests()
