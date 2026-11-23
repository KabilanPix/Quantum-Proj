# CONTINUATION PROMPT (paste everything below the line into the SAME AI IDE project)

> Use this in the same folder where the first build (simulator, Flask API, SQLite, Streamlit, ML, QAOA scheduler) already runs. If your IDE chat has lost context, that is fine: this prompt tells it to inspect the code first.

---

You are continuing work on my existing project **"Quantum Digital Twin for Smart Factories"** (Python, Flask, SQLite, Streamlit, scikit-learn, Qiskit). The base system already exists and works: `config.py`, `db.py`, `api.py`, `simulator.py`, `ml/` (features, training data generation, train, predict), `dashboard.py`, `quantum/scheduler.py`, `scripts/run_all.py`, `scripts/test_pipeline.py`, `README.md`.

I have NO hardware, so this phase upgrades the project using only software: real public datasets, an ESP32 simulation, a more realistic simulator, an automated ML-to-quantum pipeline, a QAOA benchmark, alerts, and engineering polish.

Do not ask me questions; make sensible decisions, build everything, run it, and fix errors yourself.

## 0. Ground rules (read first)

1. **Inspect before changing.** First read every existing file and run `python scripts/test_pipeline.py` to confirm the baseline passes. Print a short summary of what already exists.
2. **Do not break anything.** All new work is additive or backward compatible. After EACH phase below, re-run `python scripts/test_pipeline.py` (extend it with tests for the new phase) and fix failures before continuing.
3. **Commit-ready code:** clean, commented, type hints, docstrings, no absolute paths (use `config.py` and `pathlib`).
4. **Graceful degradation:** anything needing internet, Qiskit, Telegram or Docker must fall back with a clear message instead of crashing.
5. **Honesty in outputs:** label synthetic vs real data everywhere (dashboard, README, reports). Never present simulated results as real-hardware results.
6. Add new dependencies to `requirements.txt` (core) or `requirements-quantum.txt` (Qiskit-related). Use stable, widely available versions.
7. At the end, print the exact commands to run everything plus a list of anything not fully completed.

## 1. PHASE 1: Validate the ML on a real public dataset (highest priority)

Goal: show the predictive-maintenance approach works on real industrial data, not only synthetic.

**Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI).** Columns: `UDI, Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure, TWF, HDF, PWF, OSF, RNF`.

Tasks:
- Create `ml/real_data/` with `download_dataset.py`: try to download the UCI zip (`https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip`) into `data/external/` and extract `ai4i2020.csv`. If the download fails (no internet, URL changed), print clear manual instructions ("download the CSV from the UCI repository or Kaggle and save it as `data/external/ai4i2020.csv`") and exit cleanly; the rest of the pipeline must then skip real-data steps with a warning, not crash.
- `ml/real_data/train_real.py`:
  - Clean column names, one-hot/ordinal encode `Type`, engineer useful features (e.g., temperature difference, power = torque x rotational speed in rad/s).
  - Target: `Machine failure`. Stratified train/test split, handle heavy class imbalance (`class_weight="balanced"` and compare with SMOTE only if `imbalanced-learn` installs; otherwise skip SMOTE).
  - Train and compare: Logistic Regression (baseline), Random Forest, Gradient Boosting (or XGBoost if available). Use 5-fold stratified cross-validation.
  - Report accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, feature importances. Because failures are rare (~3%), emphasize recall, F1 and PR-AUC over accuracy and explain why in the output.
  - Save best model to `ml/model_real.joblib`, metrics to `data/real_metrics.json`, and PNGs (confusion matrix, ROC curve, PR curve, feature importance) to `data/plots/`.
- `ml/compare_models.py`: produce a comparison table (synthetic-trained model vs real-data-trained model, plus the model families) saved to `data/model_comparison.csv` and printed as a clean table.
- Add a **"Model Validation"** page/tab to `dashboard.py` (use Streamlit tabs or `st.sidebar` navigation): show the real-data metrics, plots, comparison table, and a clear banner "Validated on AI4I 2020 (real industrial dataset)" or "Real dataset not found; showing synthetic results only".
- A **"Try it" panel** in that tab: sliders for the AI4I features (air temp, process temp, rpm, torque, tool wear) that call the real-data model and show failure probability.

## 2. PHASE 2: ESP32 simulation (software hardware layer)

Goal: have real firmware code and a documented virtual hardware setup, so swapping to a physical ESP32 later is a copy-paste job.

- Create `firmware/esp32_wokwi/` containing:
  - `sketch.ino`: Arduino-style ESP32 firmware that reads a DHT22 (temperature), an MPU6050 (vibration magnitude computed from accelerometer axes) and a potentiometer (stand-in for the current sensor, scaled to kW), builds a JSON payload `{machine_id, temperature, vibration, power}`, prints it to Serial every second, and POSTs it via `HTTPClient` to a configurable `API_URL` (constant at the top; also supports `WIFI_SSID`/`WIFI_PASS`, with Wokwi's `Wokwi-GUEST` network as default).
  - `diagram.json`: a valid Wokwi diagram wiring ESP32 + DHT22 + MPU6050 + potentiometer (correct pins, I2C on GPIO21/22, DHT22 data on GPIO15, potentiometer on GPIO34).
  - `libraries.txt` listing needed libraries (DHT sensor library, Adafruit MPU6050, ArduinoJson).
  - `README.md` explaining how to paste the files into a new Wokwi ESP32 project, and **honestly noting** that a browser-hosted simulator cannot reach `localhost`: explain two options (a) run `scripts/serial_bridge.py` style workflow, (b) expose the Flask API through a tunnel such as ngrok and set `API_URL` to that public URL. Do not claim it works without those steps.
- `scripts/serial_bridge.py`: reads JSON lines from a serial port (pyserial, `--port`, `--baud`, `--machine-id`) and forwards them to the Flask `/sensor-data` endpoint. This is also what I will use with a REAL ESP32 later. Include a `--mock` mode that reads from a text file of sample lines so it can be tested without any device.
- Extend `api.py` (backward compatible) to accept an optional `source` field (`"simulator" | "esp32-sim" | "esp32-real"`) stored in a new nullable column; migrate the DB safely (`ALTER TABLE` if the column is missing). Show the source on the dashboard machine cards.

## 3. PHASE 3: More realistic simulator

Upgrade `simulator.py` (keep existing CLI flags working) with a `--scenario` argument:

- `normal`: all healthy.
- `bearing_wear`: vibration grows exponentially, temperature rises slowly.
- `overheating`: temperature climbs quickly (cooling failure), vibration stays normal.
- `power_surge`: random power spikes, then temperature rises.
- `mixed` (default for demo): M1 healthy, M2 idle/run cycles with occasional power surge, M3 bearing wear.
- Realism options (`--noise-level low|medium|high`): Gaussian noise, random sensor dropouts (missing readings), occasional outlier spikes, slow sensor drift.
- Maintenance events: when a machine reaches failure it stays down 20 s, then gets repaired and resets; log these events into a new table `events(id, machine_id, timestamp, type, detail)` and expose `GET /events`.
- Extend `ml/generate_training_data.py` to generate data from all failure modes and retrain; report per-failure-mode recall so I can see which mode is hardest.

## 4. PHASE 4: Automated ML-to-Quantum pipeline + what-if panel

Goal: no manual button clicks; the system continuously reacts to new data.

- Create `pipeline.py` with a background worker loop (every N seconds, configurable in `config.py`):
  1. Read the latest window per machine from SQLite.
  2. Compute ML risk via `ml.predict.predict_risk`.
  3. Store results in a new table `risk_scores(id, machine_id, timestamp, risk, level, top_factor)`.
  4. Build the scheduling problem with risk-weighted costs, run `quantum.scheduler.optimize` (QAOA with classical fallback), store the result in a new table `schedules(id, timestamp, assignment_json, cost, solver, classical_optimal_cost, risk_snapshot_json)`.
  5. Only re-run the optimizer if any machine's risk level changed or risk moved by more than 0.15 since the last schedule (avoid wasting QAOA runs); log the reason.
- Add `GET /risk`, `GET /schedule/latest`, `GET /schedule/history` to `api.py`.
- Update `scripts/run_all.py` to also start `pipeline.py`.
- Dashboard "Schedule" section now shows the latest automatic schedule (with timestamp, solver, cost vs classical optimum) and a history table, plus a small timeline showing how the assignment changed as M3's risk rose.
- **What-if panel:** sliders to override each machine's risk (0-100%) and a "Simulate" button that runs the optimizer on the overridden risks WITHOUT touching the database, then shows the assignment next to the current real one with a diff highlight (e.g., "Job J2 moved M3 -> M1"). Add a checkbox "Force Machine X offline for maintenance".
- Extend the scheduler to optionally include a **maintenance slot**: a high-risk machine can be assigned a "maintenance" task that occupies it, modeled in the same QUBO with a penalty if risk is HIGH and maintenance is skipped.

## 5. PHASE 5: QAOA vs classical benchmark (research-style result)

Create `quantum/benchmark.py` and a dashboard tab "Benchmark":
- Problem sizes: (jobs, machines) = (3,2), (4,2), (4,3), (5,3), (6,3), plus bigger ones only if qubit count stays <= 16 (qubits = jobs x machines). Skip larger sizes with a printed note.
- For each size, 5 random seeds (random durations/energies/risks): run (a) exact brute force, (b) greedy heuristic, (c) QAOA with reps=1 and reps=2 (if Qiskit available), and record cost, optimality gap vs brute force, runtime, number of qubits, and whether the returned solution is feasible (every job assigned exactly once).
- Save raw results to `data/benchmark_results.csv`, a summary table to `data/benchmark_summary.csv`, and plots to `data/plots/` (cost gap vs size, runtime vs size, feasibility rate).
- Dashboard tab shows the summary table and plots, with an **honest note** that QAOA runs on a classical simulator and is not expected to beat brute force at this small scale; the point is to show the formulation works and how it scales.
- Add a `--quick` flag (small sizes, 2 seeds) so it finishes in under 2 minutes for the demo.

## 6. PHASE 6: Extra intelligence features

- **Anomaly detection:** `ml/anomaly.py` with an Isolation Forest per machine trained on its recent "normal" readings; add an "Anomaly score" to machine cards and a column in `risk_scores`. Compare it with the Random Forest risk on the dashboard (do they agree?).
- **Energy forecasting:** `ml/forecast.py` predicts next-30-minute power usage per machine from history using a simple model (e.g., lag features + Gradient Boosting or `statsmodels` Holt-Winters; keep it lightweight). Show a forecast line on the power chart and feed the forecast as the `energy` term into the scheduler's cost.
- **Alerts:** `alerts.py` triggers when a machine's risk crosses `RISK_HIGH` (with a cooldown of 5 minutes per machine). Always write to an `alerts` table and the dashboard "Alerts" panel. Optionally send a Telegram message if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set in a `.env` file (use `python-dotenv`; provide `.env.example`); if not set, skip silently.

## 7. PHASE 7: Engineering polish

- **Tests:** convert/extend checks into `pytest` tests under `tests/` (API, features, model predictions, scheduler validity, pipeline loop for one iteration, benchmark `--quick`, alerts cooldown). Keep `scripts/test_pipeline.py` working as a one-command summary.
- **Docker:** `Dockerfile` and `docker-compose.yml` with services `api`, `simulator`, `pipeline`, `dashboard` sharing a volume for `data/`. `docker compose up` must start everything. Note in the README that Docker is optional.
- **CI:** `.github/workflows/ci.yml` that installs requirements, runs `pytest`, and runs a lint check (`ruff` or `flake8`).
- **README rewrite:** architecture diagram, all phases, how to run each part, results section embedding the plots, the data honesty statement (synthetic vs AI4I real vs simulated ESP32), limitations, and a "Next steps with real hardware" section (flash `sketch.ino` to a real ESP32, use `serial_bridge.py`, retrain on logged real data).
- **Demo script:** `DEMO.md` with a 5-minute walkthrough for my mentor: what to run, what to click, what to say at each step, and likely mentor questions with short answers.
- **Streamlit Cloud readiness:** make `dashboard.py` able to run standalone in a "demo mode" (pre-generated sample data from `data/demo_snapshot.db` or CSV) when the API is unreachable, so it can be deployed to Streamlit Community Cloud. Include `scripts/make_demo_snapshot.py` to create that snapshot from the current database.

## 8. Build order

Phase 1 -> 3 -> 4 -> 5 -> 2 -> 6 -> 7. (Real-data validation first, then simulator realism, then automation, then the benchmark, then the firmware layer, then extras, then polish.) After each phase: run tests, fix, and print a one-paragraph summary of what was added and where to see it in the dashboard.

## 9. Definition of done

- `python scripts/run_all.py` starts API, simulator, pipeline and dashboard; the dashboard shows live machines, ML risk, automatic schedules that change when M3's risk rises, alerts, a Model Validation tab with real AI4I metrics, a Benchmark tab, and the What-if panel.
- `pytest` and `python scripts/test_pipeline.py` pass.
- `firmware/esp32_wokwi/` and `scripts/serial_bridge.py` exist and are documented honestly.
- README, DEMO.md, Dockerfile, docker-compose.yml and CI file exist.
- No existing feature from the first build is broken.

---

# END OF PROMPT
