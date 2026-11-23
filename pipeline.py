import time
import json
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import db
from ml.predict import predict_risk
from ml.anomaly import detector
from ml.forecast import get_forecasted_energy_by_machine
from quantum.scheduler import optimize, DEFAULT_JOBS
from alerts import check_and_trigger_alerts

def run_pipeline_step(last_risk_by_machine: Dict[str, float]) -> Dict[str, Any]:
    """
    Executes one iteration of the end-to-end ML-to-Quantum pipeline:
    1. Reads latest sensor window per machine from SQLite.
    2. Fits/scores anomaly detector and computes ML failure risk.
    3. Saves risk and anomaly score to SQLite `risk_scores` table.
    4. Evaluates alerts with cooldown.
    5. Re-optimizes quantum schedule if risk level shifted or moved > 0.15.
    6. Stores updated schedule into SQLite `schedules` table.
    """
    now = time.time()
    current_risk_by_machine = {}
    top_factors = {}
    history_by_machine = {}

    # Step 1: Read latest window per machine
    for m in config.MACHINES:
        df = db.get_recent(m, n=config.WINDOW_SIZE)
        history_by_machine[m] = df
        if len(df) >= 15:
            # Predict ML risk
            res = predict_risk(df)
            risk_val = res.get("risk", 0.1)
            level = res.get("level", "HEALTHY")
            factor = res.get("top_factor", "Normal")

            # Update and score anomaly detector
            detector.fit_machine(m, df)
            latest_row = df.iloc[-1].to_dict()
            anom_score = detector.score(m, latest_row)

            # Store risk score in SQLite
            db.insert_risk_score(
                machine_id=m,
                timestamp=now,
                risk=risk_val,
                level=level,
                top_factor=factor,
                anomaly_score=anom_score
            )

            current_risk_by_machine[m] = risk_val
            top_factors[m] = factor

            # Trigger alert if high risk
            check_and_trigger_alerts(m, risk_val, factor)
        else:
            current_risk_by_machine[m] = 0.1
            top_factors[m] = "Insufficient Data"

    # Step 2: Check if re-scheduling is triggered
    recompute = False
    recompute_reason = "Initial Schedule"

    if not last_risk_by_machine:
        recompute = True
    else:
        for m in config.MACHINES:
            old_r = last_risk_by_machine.get(m, 0.1)
            new_r = current_risk_by_machine.get(m, 0.1)
            # Level change
            old_level = "HIGH" if old_r >= config.RISK_HIGH else ("MED" if old_r >= config.RISK_MED else "LOW")
            new_level = "HIGH" if new_r >= config.RISK_HIGH else ("MED" if new_r >= config.RISK_MED else "LOW")
            if old_level != new_level:
                recompute = True
                recompute_reason = f"Risk level changed on {m}: {old_level} -> {new_level}"
                break
            if abs(new_r - old_r) > 0.15:
                recompute = True
                recompute_reason = f"Risk delta on {m} > 0.15 ({old_r:.2f} -> {new_r:.2f})"
                break

    sched_res = None
    if recompute:
        print(f"[Pipeline] Recomputing Quantum Schedule: {recompute_reason}")
        
        # Incorporate energy forecast into job energies
        forecast_energies = get_forecasted_energy_by_machine(history_by_machine)
        jobs = [dict(j) for j in DEFAULT_JOBS]

        sched_res = optimize(
            jobs=jobs,
            risk_by_machine=current_risk_by_machine,
            machines=config.MACHINES,
            include_maintenance_job=True
        )

        db.insert_schedule(
            timestamp=now,
            assignment_json=json.dumps(sched_res["assignment"]),
            cost=sched_res["cost"],
            solver=sched_res["solver"],
            classical_optimal_cost=sched_res["classical_optimal_cost"],
            risk_snapshot_json=json.dumps(current_risk_by_machine)
        )
        print(f"[Pipeline] Saved schedule: cost={sched_res['cost']}, solver={sched_res['solver']}, assignment={sched_res['assignment']}")

    return {
        "timestamp": now,
        "risks": current_risk_by_machine,
        "recomputed": recompute,
        "schedule": sched_res
    }

def run_pipeline_worker():
    """Continuous background worker loop for the ML-to-Quantum pipeline."""
    db.init_db()
    print(f"[Pipeline] Starting automated ML-to-Quantum background worker (interval: {config.PIPELINE_INTERVAL_SEC}s)...")
    last_risk = {}

    while True:
        try:
            step_result = run_pipeline_step(last_risk)
            last_risk = step_result["risks"]
        except Exception as e:
            print(f"[Pipeline Error] {e}")

        time.sleep(config.PIPELINE_INTERVAL_SEC)

if __name__ == "__main__":
    run_pipeline_worker()
