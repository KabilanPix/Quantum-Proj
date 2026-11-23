import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import joblib

import config
import db
from ml.predict import predict_risk
from ml.anomaly import detector
from ml.forecast import forecast_power
from quantum.scheduler import optimize, DEFAULT_JOBS

st.set_page_config(
    page_title="Quantum Digital Twin - Smart Factory",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern glassmorphism aesthetic
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.25);
    }
    .status-badge {
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-healthy { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
    .badge-idle { background: rgba(148, 163, 184, 0.2); color: #94A3B8; border: 1px solid #94A3B8; }
    .top-factor-text {
        font-size: 0.82rem;
        color: #94A3B8;
        font-style: italic;
        margin-top: 4px;
    }
    .source-tag {
        font-size: 0.75rem;
        padding: 2px 6px;
        background: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border-radius: 4px;
        border: 1px solid #3B82F6;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def get_status_badge_html(status: str) -> str:
    if status == "CRITICAL":
        return '<span class="status-badge badge-critical">🔴 CRITICAL</span>'
    elif status == "WARNING":
        return '<span class="status-badge badge-warning">🟡 WARNING</span>'
    elif status == "IDLE":
        return '<span class="status-badge badge-idle">⚪ IDLE</span>'
    else:
        return '<span class="status-badge badge-healthy">🟢 HEALTHY</span>'

# Sidebar Controls
st.sidebar.title("🎛️ Digital Twin Control")
active_tab = st.sidebar.radio(
    "Navigation Views",
    ["🏭 Factory Floor & Telemetry", "⚛️ Schedule & What-If", "🔬 Real Model Validation", "📊 QAOA Benchmark", "🚨 Alerts & Events"]
)

refresh_interval = st.sidebar.slider("Auto-refresh interval (s)", min_value=1, max_value=10, value=2)
history_points = st.sidebar.slider("History points to display", min_value=30, max_value=500, value=120)

if st.sidebar.button("🧹 Clear Factory Data", use_container_width=True):
    try:
        db.clear_readings()
        st.sidebar.success("Database cleared!")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error clearing DB: {e}")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Digital Twin Pipeline**:\nSensors (Simulator/ESP32) ➡️ Flask API ➡️ SQLite (WAL) ➡️ ML Risk & Anomaly ➡️ Quantum QAOA Scheduler")

# ==========================================
# TAB 1: FACTORY FLOOR & LIVE TELEMETRY
# ==========================================
if active_tab == "🏭 Factory Floor & Telemetry":
    st.title("🏭 Smart Factory Digital Twin")
    st.caption("Live Cyber-Physical Telemetry, Sensor Anomaly Scores & Predictive Risk Stream")

    machine_dfs = {}
    latest_readings = {}
    machine_risks = {}

    for m in config.MACHINES:
        try:
            df = db.get_recent(m, n=history_points)
        except Exception:
            df = pd.DataFrame()
        machine_dfs[m] = df

        if not df.empty:
            latest_readings[m] = df.iloc[-1].to_dict()
            risk_info = predict_risk(df)
            machine_risks[m] = risk_info
        else:
            latest_readings[m] = None
            machine_risks[m] = {"risk": 0.0, "level": "LOW", "top_factor": "No data stream"}

    has_data = any(r is not None for r in latest_readings.values())

    if not has_data:
        st.warning("⏳ Waiting for factory sensor stream... Ensure API and Simulator are running, or run: `python scripts/run_all.py`.")
    else:
        # Machine Cards
        st.markdown("### 📡 Machine Digital Twin Nodes")
        cols = st.columns(len(config.MACHINES))
        high_risk_count = 0
        powers = []

        for i, m in enumerate(config.MACHINES):
            reading = latest_readings.get(m)
            risk_info = machine_risks.get(m, {"risk": 0.0, "level": "LOW", "top_factor": "Nominal"})
            risk_val = risk_info["risk"]
            risk_pct = int(risk_val * 100)

            if risk_val >= config.RISK_HIGH:
                high_risk_count += 1

            with cols[i]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                status_str = reading.get("status", "UNKNOWN") if reading else "OFFLINE"
                source_str = reading.get("source", "simulator") if reading else "unknown"
                badge = get_status_badge_html(status_str)
                
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                    f"<h4>Machine {m}</h4><div><span class='source-tag'>{source_str}</span> {badge}</div></div>",
                    unsafe_allow_html=True
                )

                if reading:
                    t_val = reading['temperature']
                    v_val = reading['vibration']
                    p_val = reading['power']
                    powers.append(p_val)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Temp", f"{t_val:.1f} °C")
                    c2.metric("Vib", f"{v_val:.2f} g")
                    c3.metric("Power", f"{p_val:.2f} kW")

                    # Anomaly detector score
                    anom_score = detector.score(m, reading)
                    st.markdown(f"**ML Failure Risk:** `{risk_pct}%` ({risk_info['level']}) | **Anomaly Score:** `{anom_score:.2f}`")
                    st.progress(min(1.0, max(0.0, risk_val)))
                    st.markdown(f'<div class="top-factor-text">⚠️ Driver: {risk_info["top_factor"]}</div>', unsafe_allow_html=True)
                else:
                    st.write("Machine Offline")

                st.markdown('</div>', unsafe_allow_html=True)

        # Telemetry Charts
        st.markdown("### 📈 Real-Time Sensor Telemetry & Energy Forecast")
        chart_col1, chart_col2, chart_col3 = st.columns(3)

        # Temperature Plot
        fig_temp = go.Figure()
        for m in config.MACHINES:
            df = machine_dfs[m]
            if not df.empty:
                fig_temp.add_trace(go.Scatter(y=df["temperature"], mode="lines", name=m))
        fig_temp.add_hline(y=config.TEMP_WARNING, line_dash="dash", line_color="#F59E0B", annotation_text="WARN (75°C)")
        fig_temp.add_hline(y=config.TEMP_CRITICAL, line_dash="dash", line_color="#EF4444", annotation_text="CRIT (85°C)")
        fig_temp.update_layout(title="Temperature (°C)", height=280, margin=dict(l=20, r=20, t=35, b=20), template="plotly_dark")
        chart_col1.plotly_chart(fig_temp, use_container_width=True)

        # Vibration Plot
        fig_vib = go.Figure()
        for m in config.MACHINES:
            df = machine_dfs[m]
            if not df.empty:
                fig_vib.add_trace(go.Scatter(y=df["vibration"], mode="lines", name=m))
        fig_vib.add_hline(y=config.VIB_WARNING, line_dash="dash", line_color="#F59E0B", annotation_text="WARN (1.0g)")
        fig_vib.add_hline(y=config.VIB_CRITICAL, line_dash="dash", line_color="#EF4444", annotation_text="CRIT (1.5g)")
        fig_vib.update_layout(title="Vibration (g)", height=280, margin=dict(l=20, r=20, t=35, b=20), template="plotly_dark")
        chart_col2.plotly_chart(fig_vib, use_container_width=True)

        # Power Plot with Forecast
        fig_power = go.Figure()
        for m in config.MACHINES:
            df = machine_dfs[m]
            if not df.empty:
                fig_power.add_trace(go.Scatter(y=df["power"], mode="lines", name=f"{m} Actual"))
                # Energy forecast for next 15 intervals
                preds = forecast_power(df, steps_ahead=15)
                x_forecast = list(range(len(df), len(df) + len(preds)))
                fig_power.add_trace(go.Scatter(x=x_forecast, y=preds, mode="lines", line=dict(dash="dot"), name=f"{m} Forecast (30m)"))
        fig_power.update_layout(title="Power (kW) & Forecast", height=280, margin=dict(l=20, r=20, t=35, b=20), template="plotly_dark")
        chart_col3.plotly_chart(fig_power, use_container_width=True)

        # Risk Over Time Line Chart
        st.markdown("### ⚠️ Failure Risk Trend (% ML Probability)")
        fig_risk = go.Figure()
        for m in config.MACHINES:
            df = machine_dfs[m]
            if not df.empty and len(df) >= 10:
                indices = list(range(10, len(df) + 1, 3))
                rolling_risks = [predict_risk(df.iloc[max(0, idx - config.WINDOW_SIZE):idx])["risk"] * 100 for idx in indices]
                fig_risk.add_trace(go.Scatter(x=indices, y=rolling_risks, mode="lines+markers", name=f"{m} Risk %"))
        fig_risk.add_hline(y=config.RISK_MED * 100, line_dash="dash", line_color="#F59E0B", annotation_text="MED (40%)")
        fig_risk.add_hline(y=config.RISK_HIGH * 100, line_dash="dash", line_color="#EF4444", annotation_text="HIGH (70%)")
        fig_risk.update_layout(title="Rolling Failure Risk (%)", height=250, margin=dict(l=20, r=20, t=35, b=20), template="plotly_dark")
        st.plotly_chart(fig_risk, use_container_width=True)

# ==========================================
# TAB 2: SCHEDULE & WHAT-IF SIMULATION
# ==========================================
elif active_tab == "⚛️ Schedule & What-If":
    st.title("⚛️ Automated Production Scheduling & What-If Simulation")
    st.caption("Real-Time QUBO Optimization: QAOA with Classical Fallback, Risk-Penalized Workload Balancing")

    # Latest Automatic Schedule from Background Pipeline
    st.markdown("### 🕒 Latest Automatic Schedule (Live Background Pipeline)")
    latest_sched = db.get_latest_schedule()

    if latest_sched:
        assign_dict = json.loads(latest_sched["assignment_json"])
        t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_sched["timestamp"]))
        
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        info_col1.metric("Optimization Timestamp", t_str)
        info_col2.metric("Solver Used", latest_sched["solver"])
        info_col3.metric("Schedule Cost", f"{latest_sched['cost']:.2f}")
        info_col4.metric("Classical Optimal Cost", f"{latest_sched['classical_optimal_cost']:.2f}")

        # Render Assignment Table
        rows = []
        for j in DEFAULT_JOBS:
            m_ass = assign_dict.get(j["id"], "N/A")
            rows.append({
                "Job ID": j["id"],
                "Assigned Machine": m_ass,
                "Duration (min)": j["duration"],
                "Energy (kWh)": j["energy"]
            })
        for k, v in assign_dict.items():
            if k.startswith("MAINT_"):
                rows.append({
                    "Job ID": f"🛠️ {k} (Maintenance Task)",
                    "Assigned Machine": v,
                    "Duration (min)": 20,
                    "Energy (kWh)": 2.0
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No automatic schedule recorded yet. Pipeline will compute one automatically as soon as sensor data arrives.")

    # Schedule History Table
    st.markdown("### 📜 Automated Schedule History")
    hist = db.get_schedule_history(limit=10)
    if hist:
        hist_rows = []
        for h in hist:
            hist_rows.append({
                "Timestamp": time.strftime("%H:%M:%S", time.localtime(h["timestamp"])),
                "Solver": h["solver"],
                "Cost": round(h["cost"], 2),
                "Optimal Cost": round(h["classical_optimal_cost"], 2),
                "Assignments": h["assignment_json"],
                "Risk Snapshot": h["risk_snapshot_json"]
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)

    # WHAT-IF SIMULATION PANEL
    st.markdown("---")
    st.markdown("### 🧪 What-If Production Simulator (Interactive Experimentation)")
    st.write("Override machine risk percentages or simulate offline machines WITHOUT modifying the live database.")

    wcol1, wcol2, wcol3 = st.columns(3)
    with wcol1:
        m1_sim_risk = st.slider("M1 Risk % (Override)", 0, 100, 10) / 100.0
        m1_offline = st.checkbox("Force M1 Offline (Maintenance)", value=False)
    with wcol2:
        m2_sim_risk = st.slider("M2 Risk % (Override)", 0, 100, 20) / 100.0
        m2_offline = st.checkbox("Force M2 Offline (Maintenance)", value=False)
    with wcol3:
        m3_sim_risk = st.slider("M3 Risk % (Override)", 0, 100, 85) / 100.0
        m3_offline = st.checkbox("Force M3 Offline (Maintenance)", value=False)

    if st.button("🚀 Run What-If Simulation", use_container_width=True):
        sim_risks = {"M1": m1_sim_risk, "M2": m2_sim_risk, "M3": m3_sim_risk}
        offline_list = []
        if m1_offline: offline_list.append("M1")
        if m2_offline: offline_list.append("M2")
        if m3_offline: offline_list.append("M3")

        sim_result = optimize(
            jobs=DEFAULT_JOBS,
            risk_by_machine=sim_risks,
            machines=config.MACHINES,
            offline_machines=offline_list,
            include_maintenance_job=True
        )

        st.success(f"Simulation Computed! Solver: **{sim_result['solver']}** | Simulated Cost: **{sim_result['cost']:.2f}**")

        # Compare with current active schedule
        cur_assign = json.loads(latest_sched["assignment_json"]) if latest_sched else {}
        sim_assign = sim_result["assignment"]

        diff_rows = []
        for j in DEFAULT_JOBS:
            j_id = j["id"]
            live_m = cur_assign.get(j_id, "None")
            sim_m = sim_assign.get(j_id, "None")
            status_text = "SAME" if live_m == sim_m else f"⚠️ MOVED ({live_m} ➡️ {sim_m})"
            diff_rows.append({
                "Job": j_id,
                "Live Assignment": live_m,
                "What-If Assignment": sim_m,
                "Allocation Shift": status_text
            })
        st.table(pd.DataFrame(diff_rows))

# ==========================================
# TAB 3: REAL MODEL VALIDATION (UCI AI4I 2020)
# ==========================================
elif active_tab == "🔬 Real Model Validation":
    st.title("🔬 Real Industrial ML Validation: UCI AI4I 2020 Dataset")
    st.caption("Rigorous benchmark proving predictive maintenance models on genuine milling machine sensor logs (10,000 runs, ~3.4% failure rate).")

    if config.REAL_METRICS_PATH.exists() and config.MODEL_COMPARISON_PATH.exists():
        st.success("✅ **Validated on AI4I 2020 (Real Industrial Dataset)**")
        
        with open(config.REAL_METRICS_PATH, "r") as f:
            real_metrics = json.load(f)

        # Overview stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Real Dataset Samples", f"{real_metrics['samples']:,}")
        m2.metric("Failure Instances", f"{real_metrics['failure_count']}")
        m3.metric("Failure Rate (Imbalance)", f"{real_metrics['failure_rate']*100:.2f}%")
        m4.metric("Best Trained Model", real_metrics["best_model"])

        # Display Comparison Table
        st.markdown("### 📊 Model Performance Comparison: Synthetic vs Real")
        df_comp = pd.read_csv(config.MODEL_COMPARISON_PATH)
        st.dataframe(df_comp, use_container_width=True)

        # Display Plots
        st.markdown("### 📈 Evaluation Curves & Confusion Matrices")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            cm_img = config.PLOTS_DIR / "real_confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrix on Real Test Split")
            roc_img = config.PLOTS_DIR / "real_roc_curve.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="ROC Curves (AUC Benchmarking)")
        with pcol2:
            pr_img = config.PLOTS_DIR / "real_pr_curve.png"
            if pr_img.exists():
                st.image(str(pr_img), caption="Precision-Recall Curves (Key for Rare Failures)")
            fi_img = config.PLOTS_DIR / "real_feature_importance.png"
            if fi_img.exists():
                st.image(str(fi_img), caption="Feature Importances on Real Machinery")

        # Interactive "Try It" Live Risk Predictor
        st.markdown("---")
        st.markdown("### 🎛️ 'Try It' Panel: Live AI4I Real Machine Risk Scoring")
        st.write("Adjust actual milling machine parameters to evaluate failure probability using the best real-data model.")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            air_temp = st.slider("Air Temperature [K]", 295.0, 305.0, 298.5, 0.1)
            process_temp = st.slider("Process Temperature [K]", 305.0, 315.0, 308.6, 0.1)
        with col_b:
            rpm = st.slider("Rotational Speed [rpm]", 1100, 2900, 1500, 10)
            torque = st.slider("Torque [Nm]", 3.0, 80.0, 42.0, 0.5)
        with col_c:
            tool_wear = st.slider("Tool Wear [min]", 0, 260, 180, 5)
            machine_type = st.selectbox("Machine Quality Type", ["L (Low)", "M (Medium)", "H (High)"])

        if config.MODEL_REAL_PATH.exists():
            saved_real = joblib.load(config.MODEL_REAL_PATH)
            model_real = saved_real["model"]
            feat_names = saved_real["feature_names"]

            # Compute engineered features
            temp_diff = process_temp - air_temp
            power_w = torque * (rpm * 2.0 * np.pi / 60.0)
            tw_torque = tool_wear * torque
            heat_diss = temp_diff * (rpm / 1000.0)
            type_code = 0 if "L" in machine_type else (1 if "M" in machine_type else 2)

            input_data = pd.DataFrame([{
                "air_temp": air_temp,
                "process_temp": process_temp,
                "rpm": rpm,
                "torque": torque,
                "tool_wear": tool_wear,
                "temp_diff": temp_diff,
                "power_watts": power_w,
                "tool_wear_torque": tw_torque,
                "heat_dissipation": heat_diss,
                "type_code": type_code
            }])[feat_names]

            prob = float(model_real.predict_proba(input_data)[0, 1])
            st.metric("Predicted Machine Failure Probability", f"{prob*100:.1f}%")
            st.progress(prob)
            if prob >= 0.70:
                st.error("🚨 HIGH RISK of Immediate Machine Breakdown! Schedule tool replacement or reduce load.")
            elif prob >= 0.40:
                st.warning("⚠️ MODERATE RISK: Accelerated wear detected.")
            else:
                st.success("🟢 NORMAL OPERATION: Operating well within safety envelopes.")
    else:
        st.warning("⚠️ Real dataset not yet trained. Run `python ml/real_data/train_real.py` to populate AI4I 2020 validation.")

# ==========================================
# TAB 4: QAOA BENCHMARK
# ==========================================
elif active_tab == "📊 QAOA Benchmark":
    st.title("📊 QAOA vs Classical Exact Solver Benchmark")
    st.caption("Empirical scaling analysis across multiple QUBO problem dimensions.")

    st.markdown("""
    > [!NOTE]
    > **Honesty Statement:** QAOA runs here on a statevector/matrix classical simulator. At small qubit counts (6 to 16 qubits), classical brute force and greedy heuristics are faster; the benchmark is designed to empirically demonstrate that the QUBO mapping is correct, explore optimality gaps, and assess feasibility scaling.
    """)

    if config.BENCHMARK_SUMMARY_PATH.exists():
        df_bench = pd.read_csv(config.BENCHMARK_SUMMARY_PATH)
        st.markdown("### 📋 Benchmark Summary Table")
        st.dataframe(df_bench, use_container_width=True)

        st.markdown("### 📉 Benchmark Visualizations")
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            bgap = config.PLOTS_DIR / "benchmark_cost_gap.png"
            if bgap.exists():
                st.image(str(bgap), caption="Optimality Gap vs Problem Size")
            brun = config.PLOTS_DIR / "benchmark_runtime.png"
            if brun.exists():
                st.image(str(brun), caption="Execution Runtime Scaling (Log Scale)")
        with bcol2:
            bfeas = config.PLOTS_DIR / "benchmark_feasibility.png"
            if bfeas.exists():
                st.image(str(bfeas), caption="Solution Feasibility Rate across Qubits")
    else:
        st.info("Run `python quantum/benchmark.py --quick` to generate QAOA benchmark results.")

# ==========================================
# TAB 5: ALERTS & EVENTS
# ==========================================
elif active_tab == "🚨 Alerts & Events":
    st.title("🚨 Factory Incident Alerts & Maintenance Events")
    st.caption("Autonomous Incident Auditing, Threshold Crossings & Maintenance History")

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.markdown("### 🔔 Active Incident Alerts (5m Cooldown)")
        alerts = db.get_recent_alerts(limit=30)
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else:
            st.info("No active incident alerts registered.")

    with ecol2:
        st.markdown("### 🛠️ Maintenance & Machine Lifecycle Events")
        events = db.get_recent_events(limit=30)
        if events:
            st.dataframe(pd.DataFrame(events), use_container_width=True)
        else:
            st.info("No maintenance events logged yet.")

# Sleep and auto-refresh
time.sleep(refresh_interval)
st.rerun()
