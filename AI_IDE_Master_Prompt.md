# MASTER PROMPT (paste everything below the line into your AI IDE)

> Tip: In Claude Code / Cursor / Copilot Chat (agent mode), open an EMPTY folder first, then paste this whole prompt.

---

You are a senior Python engineer. Build a complete, working software prototype for the project **"Quantum Digital Twin for Smart Factories using IoT, ML Predictive Maintenance, and Quantum Optimization"**. This is for a student project review tomorrow, so everything must run end-to-end on a normal laptop with NO hardware and NO cloud accounts. Do not ask me clarifying questions; make sensible decisions, build everything, run it, and fix errors yourself until all acceptance tests at the bottom pass.

## 1. Goal

Simulate a small factory with 3 machines. Fake sensors stream readings (as an ESP32 would) to a Flask API, which stores them in SQLite. A Streamlit dashboard shows a live "digital twin" of the machines. A scikit-learn model predicts each machine's failure risk from recent sensor history, and the dashboard shows it. A stretch module uses Qiskit QAOA to schedule jobs across machines, penalizing high-risk machines.

Pipeline: `Simulator (fake ESP32) -> Flask API -> SQLite -> ML risk model -> Streamlit dashboard -> (stretch) Qiskit QAOA scheduler`

## 2. Tech stack (use exactly this)

- Python 3.10+ with a virtual environment (`venv`)
- Flask (REST API), flask-cors not required
- SQLite via Python's built-in `sqlite3` (file: `data/factory.db`), WAL mode enabled
- pandas, numpy
- scikit-learn (RandomForestClassifier), joblib for saving the model
- Streamlit + plotly for the dashboard
- requests (simulator -> API)
- STRETCH ONLY: qiskit, qiskit-aer, qiskit-algorithms, qiskit-optimization

Create `requirements.txt` with the core packages first. Put Qiskit packages in a separate `requirements-quantum.txt` so the core project installs and runs even if Qiskit install fails.

## 3. Required file structure

```
quantum-digital-twin/
├── README.md
├── requirements.txt
├── requirements-quantum.txt
├── config.py
├── db.py
├── api.py
├── simulator.py
├── ml/
│   ├── __init__.py
│   ├── features.py
│   ├── generate_training_data.py
│   ├── train_model.py
│   ├── predict.py
│   └── model.joblib          (generated)
├── dashboard.py
├── quantum/
│   ├── __init__.py
│   └── scheduler.py          (stretch)
├── scripts/
│   ├── run_all.py            (starts API + simulator + dashboard)
│   └── test_pipeline.py      (acceptance tests)
└── data/
    └── factory.db            (generated)
```

## 4. Detailed specifications

### config.py
Central constants shared by all modules:
- `API_HOST="127.0.0.1"`, `API_PORT=5000`, `DB_PATH="data/factory.db"` (resolve paths relative to project root using `pathlib`, create `data/` if missing)
- `MACHINES = ["M1", "M2", "M3"]`
- `SIM_INTERVAL_SEC = 1.0` (one reading per machine per second)
- Normal operating ranges: temperature 55-65 C, vibration 0.2-0.5 g, power 3.0-4.0 kW
- Status thresholds: temp > 85 C or vibration > 1.5 g = "CRITICAL"; temp > 75 C or vibration > 1.0 g = "WARNING"; otherwise "HEALTHY". A machine with power < 0.3 kW is "IDLE".
- ML: `WINDOW_SIZE = 30` (last 30 readings per machine), `RISK_HIGH = 0.7`, `RISK_MED = 0.4`

### db.py
- `init_db()` creates table `readings(id INTEGER PK AUTOINCREMENT, machine_id TEXT, timestamp REAL, temperature REAL, vibration REAL, power REAL, status TEXT)` and an index on `(machine_id, timestamp)`.
- Every connection uses `sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)` and `PRAGMA journal_mode=WAL` so Flask, simulator and Streamlit can access the DB concurrently without "database is locked" errors.
- Functions: `insert_reading(...)`, `get_recent(machine_id, n)` (returns pandas DataFrame sorted by time ascending), `get_latest_all()` (latest row per machine), `clear_readings()`.

### api.py (Flask)
- `POST /sensor-data` accepts JSON `{machine_id, temperature, vibration, power, timestamp(optional)}`; validates fields; computes `status` using config thresholds; stores it; returns `{"ok": true}` with 201. Return 400 with a message on bad input.
- `GET /latest` returns the latest reading per machine as JSON.
- `GET /history/<machine_id>?n=200` returns recent readings.
- `GET /health` returns `{"status":"ok"}`.
- Runs on `API_HOST:API_PORT`, `debug=False`, `threaded=True`. Calls `init_db()` on startup.

### simulator.py (fake ESP32 sensors)
- Loop forever; every `SIM_INTERVAL_SEC` send one reading for each machine via `requests.post` to the API (with a 2s timeout and try/except so it retries instead of crashing if the API is not up yet).
- M1: healthy, normal values with small Gaussian noise.
- M2: healthy but cycles between running and idle (power drops below 0.3 kW for ~15 seconds every ~90 seconds; temp and vibration drop slightly when idle).
- M3: **degrades over time** so the demo is visible live: starting from normal, over `--degrade-seconds` (default 240) temperature drifts up toward ~95 C and vibration drifts up toward ~2.0 g, with noise and occasional small spikes; power rises slightly. After reaching failure it stays critical for 20 seconds, then "gets repaired" and resets to normal and starts degrading again (loop).
- CLI args via argparse: `--degrade-seconds`, `--interval`, `--reset-db` (clears old readings first).
- Print one concise log line per cycle.

### ml/features.py
One function `extract_features(df)` where `df` has columns temperature, vibration, power for ONE machine (last `WINDOW_SIZE` rows). Returns a single-row feature vector with these exact feature names, always in this order:
`temp_mean, temp_std, temp_slope, vib_mean, vib_std, vib_slope, vib_max, power_mean, power_std, power_slope`
where slope = linear-fit slope per reading using `numpy.polyfit`. Handle windows shorter than `WINDOW_SIZE` (pad by repeating the first value, or return None if fewer than 5 rows). The SAME function must be used for training and for prediction (no train/serve mismatch). Export `FEATURE_NAMES`.

### ml/generate_training_data.py
Do NOT depend on the live database. Generate a synthetic labeled dataset offline:
- Simulate ~200 machine "life cycles": each is a run of readings where healthy behavior is followed (in ~50% of cycles) by a gradual degradation (temperature and vibration trends upward with noise) ending in failure. Vary degradation speed, noise levels and start values randomly (seeded with `np.random.seed(42)`).
- Slide a window of `WINDOW_SIZE` over each cycle, extract features with `extract_features`, and label the window `1` if failure occurs within the next 60 readings, else `0`. Ensure the classes are reasonably balanced (undersample or use `class_weight="balanced"`).
- Save to `data/training_data.csv`.

### ml/train_model.py
- Load `data/training_data.csv` (generate it first if missing).
- Train/test split 80/20 stratified; train `RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)`.
- Print accuracy, precision, recall, F1, confusion matrix and the feature importances (sorted). Save a confusion-matrix and feature-importance PNG into `data/` (useful for review slides).
- Save the model with joblib to `ml/model.joblib` together with `FEATURE_NAMES`.
- Must finish in under 1 minute.

### ml/predict.py
- `predict_risk(machine_df) -> dict` returning `{"risk": float 0-1, "level": "LOW|MEDIUM|HIGH", "top_factor": str}`. Loads the model once (module-level cache). If `model.joblib` is missing, train it automatically. If there are fewer than 5 readings, return risk 0.0 and level "LOW". `top_factor` is the feature with the largest (feature importance x deviation) contribution, used as a human-readable reason, e.g. "rising vibration".

### dashboard.py (Streamlit) - the "Digital Twin"
- Title: "Quantum Digital Twin - Smart Factory". Sidebar: auto-refresh interval slider (1-10 s, default 2), number of history points slider, and a "Clear data" button.
- Auto-refresh using `st.empty()` containers and a loop with `time.sleep`, or `streamlit-autorefresh` if you add it to requirements. It must refresh without a manual reload and must not crash if the DB is empty or the API is down (show a friendly "Waiting for data..." message).
- Top row: one card per machine (3 columns) showing machine name, status badge with an emoji (🟢 HEALTHY, 🟡 WARNING, 🔴 CRITICAL, ⚪ IDLE), current temperature, vibration, power, and the **ML failure risk %** with a colored progress bar (green < 40%, orange 40-70%, red > 70%) and the `top_factor` reason.
- Middle: three plotly line charts (temperature, vibration, power) with all machines as separate lines over the last N points; add dashed horizontal lines for the WARNING/CRITICAL thresholds on temperature and vibration.
- A "Risk over time" line chart per machine (compute risk on a rolling basis over the stored history).
- Bottom: KPI row (total readings stored, machines at HIGH risk, average power, system uptime %).
- Bottom: a "Production Schedule (Quantum Optimizer)" section with a button "Run Optimizer" that calls `quantum.scheduler.optimize(...)` and displays the resulting job-to-machine table (see below). If Qiskit is not installed, it must fall back to the classical solver and show a note saying which solver was used.

### quantum/scheduler.py (STRETCH - must degrade gracefully)
- `optimize(jobs, risk_by_machine, machines) -> dict` where `jobs` is a list of `{id, duration, energy}` (use 4 default jobs, 3 machines).
- Formulate job-to-machine assignment as a QUBO: binary variable `x[j,m]`; each job assigned to exactly one machine (penalty term); minimize `sum(x[j,m] * (energy_j + RISK_WEIGHT * risk_m * duration_j))` plus a load-balancing term so one machine does not get everything. High-risk machines should receive fewer/lighter jobs.
- Solve with Qiskit QAOA (`qiskit_algorithms.QAOA` + `qiskit_optimization.algorithms.MinimumEigenOptimizer`, using the Aer/reference Sampler, `COBYLA`, reps=1 or 2). Keep the problem small (<= 12 qubits) so the simulator is fast.
- Wrap imports and the quantum solve in try/except; on any failure fall back to an exact classical brute-force solver over the same cost function and report `solver: "classical-fallback"`. Always also compute the brute-force optimum so the output can compare quantum vs classical cost.
- Return `{"assignment": {job_id: machine}, "cost": float, "solver": "QAOA"|"classical-fallback", "classical_optimal_cost": float}`.
- Include `if __name__ == "__main__":` demo that prints the result.

### scripts/run_all.py
Starts API, simulator (`--reset-db --degrade-seconds 120` for a fast demo) and the Streamlit dashboard as subprocesses using `sys.executable`, waits for `/health` to return OK before starting the simulator, prints the URLs (API http://127.0.0.1:5000, dashboard http://localhost:8501), and terminates all child processes cleanly on Ctrl+C. Must work on Windows, macOS and Linux (no shell-specific commands).

### scripts/test_pipeline.py (acceptance tests)
Runs automatically and prints PASS/FAIL for each check:
1. `config`, `db`, `api` import without errors.
2. API starts (temporary subprocess), `/health` OK, `POST /sensor-data` with a valid payload returns 201, bad payload returns 400, `/latest` returns data.
3. Feature extraction returns exactly 10 features in the correct order.
4. Model trains and test F1 >= 0.80; `predict_risk` on a clearly healthy window returns risk < 0.4 and on a clearly degraded window returns risk > 0.7.
5. `quantum.scheduler.optimize` returns a valid assignment (every job assigned to exactly one machine), whichever solver ran.
6. Dashboard file passes a syntax/import check (`python -m py_compile dashboard.py`).

### README.md
Include: project overview, architecture diagram (ASCII), setup commands for Windows and macOS/Linux, exact run commands, the demo script for the review (what to click and what to say), explanation of the ML approach (features, model, why simulated data), limitations and next steps (replace simulator with real ESP32 + sensors, real data, run QAOA on real quantum hardware).

## 5. Build order (follow strictly, run each step before moving on)

1. Create folder structure, venv, `requirements.txt`, install packages.
2. `config.py`, `db.py`, `api.py`; test the API with `curl` or `requests`.
3. `simulator.py`; confirm rows appear in SQLite.
4. `dashboard.py` WITHOUT ML (status + charts); confirm it renders and updates live.
5. `ml/` module: generate data, train, predict; print metrics.
6. Integrate risk into the dashboard.
7. `quantum/scheduler.py` with classical fallback first, then QAOA; integrate the "Run Optimizer" button.
8. `scripts/run_all.py`, `scripts/test_pipeline.py`, `README.md`.
9. Run `python scripts/test_pipeline.py` and fix everything until every test passes.

## 6. Quality rules

- Clean, commented, beginner-readable code with type hints and docstrings; no unnecessary abstractions.
- No hardcoded absolute paths; everything via `config.py` and `pathlib`.
- Never crash on empty DB, missing model, or API downtime; show friendly messages.
- Pin nothing exotic; use widely available stable versions. If a Qiskit API has changed, check the installed version and adapt, or use the fallback.
- Do not invent library functions. If unsure, inspect the installed package or run a quick check.
- At the end, print: the exact commands to start the project, the dashboard URL, and a short list of anything that could not be fully completed.

## 7. Definition of done

Running `python scripts/run_all.py` brings up the API, the simulator and the dashboard. Within ~30 seconds the dashboard shows 3 live machines. M3 visibly degrades, its ML risk climbs from LOW to HIGH within ~2 minutes, and the status turns WARNING then CRITICAL. Clicking "Run Optimizer" returns a job schedule that avoids high-risk machines. `python scripts/test_pipeline.py` passes all checks.

---

# END OF PROMPT
