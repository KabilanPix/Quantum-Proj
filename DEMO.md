# 🏭 Mentor Walkthrough & Live Demo Script (5 Minutes)

Use this guide for a seamless 5-minute presentation of the **Quantum Digital Twin for Smart Factories**.

---

## ⏱️ Minute 0:00 - 1:00 | System Architecture & Launch
**Action to run:**
```bash
python scripts/run_all.py
```
Open browser at: `http://localhost:8501`

**What to say:**
> *"Good morning. Today I'm presenting our Quantum Digital Twin for Smart Factories. In modern manufacturing, cyber-physical monitoring, predictive maintenance, and operational scheduling are often disconnected. Our system unifies them end-to-end: IoT sensors stream telemetry to a high-throughput REST API with SQLite WAL persistence; an unsupervised anomaly detector and Random Forest evaluate breakdown risk in real-time; and when machine risk crosses critical thresholds, our Quantum-inspired QAOA formulation automatically redistributes production workloads and assigns maintenance slots to prevent catastrophic factory line halts."*

---

## ⏱️ Minute 1:00 - 2:00 | Real-Time Telemetry & Fault Progression
**What to click:**
- View the **"🏭 Factory Floor & Telemetry"** tab on the Streamlit dashboard.

**What to highlight:**
- **Machine Cards:** Point out **M1** (healthy baseline), **M2** (duty cycling between running and idle), and **M3** (degrading bearing wear).
- Note the **Source badge** (`simulator` or `esp32-sim`), proving firmware traceability.
- **Charts:** Point out the live Temperature, Vibration, and Power charts. Point to the dashed dotted lines on the Power chart showing our **Energy Forecaster** (predicting future 30-minute power draw).
- **Driver attribution:** Show how M3's card shows `Driver: elevated vibration / temperature slope`.

---

## ⏱️ Minute 2:00 - 3:00 | Automated ML-to-Quantum Reactive Scheduling
**What to click:**
- Navigate to the **"⚛️ Schedule & What-If"** tab.

**What to highlight:**
- **Automatic Schedule Table:** Explain that the background pipeline automatically evaluated risks without any manual click:
  > *"Notice how Job J2 (the heaviest 25-minute job) has been dynamically steered away from degraded Machine M3 over to M1 or M2. Furthermore, a dedicated `MAINT_M3` maintenance task was inserted into the QUBO optimization problem to service the machine."*
- **What-If Simulation:**
  - Drag the **M1 Risk slider** to 90%.
  - Check the box **"Force Machine M1 Offline"**.
  - Click **"🚀 Run What-If Simulation"**.
  - Show the diff table indicating how assignments shifted immediately in memory without mutating database state.

---

## ⏱️ Minute 3:00 - 4:00 | Real Industrial Data Validation (UCI AI4I 2020)
**What to click:**
- Navigate to the **"🔬 Real Model Validation"** tab.

**What to say & show:**
> *"A frequent critique of digital twin research is reliance solely on synthetic equations. To rigorously validate our predictive maintenance logic, we benchmarked against the real-world UCI AI4I 2020 Predictive Maintenance dataset—10,000 milling cycles with genuine industrial failures (~3.4% imbalance)."*
- Point to the **Comparison Table** comparing our synthetic model vs real Gradient Boosting / Random Forest.
- Point to the **Precision-Recall (PR) and ROC Curves**.
- **Interactive "Try It" Demo:** Move the Tool Wear slider to 220 minutes and Torque to 65 Nm. Watch the predicted failure probability jump to >80% and trigger the breakdown warning.

---

## ⏱️ Minute 4:00 - 5:00 | QAOA Benchmark, Virtual Hardware & Next Steps
**What to click:**
- Navigate to the **"📊 QAOA Benchmark"** tab.
- Explain:
  > *"We benchmarked QAOA against classical brute force and greedy heuristics across 6 to 12 qubits. On small scale simulator runs, classical brute force is naturally fast; this benchmark empirically proves that the QUBO mapping works, achieves zero constraint violations, and models scaling bottlenecks."*
- Show the firmware directory:
  > *"We have also completed the firmware layer in `firmware/esp32_wokwi/` with Wokwi diagrams and a USB Serial Bridge (`scripts/serial_bridge.py`), ready for direct physical ESP32 flashing."*

---

## 🧠 Anticipated Mentor Questions & Short Answers

### Q1: *"Did QAOA run on real quantum hardware or a simulator?"*
**Answer:**
> *"It ran on a classical statevector quantum simulator using Qiskit. Because current NISQ hardware incurs cloud latency and queue delays, the simulator allows us to validate the QUBO quadratic Hamiltonian formulation locally while providing an exact classical fallback for guaranteed production uptime."*

### Q2: *"Why is recall prioritized over accuracy on the real dataset?"*
**Answer:**
> *"In industrial predictive maintenance, failures represent only ~3.4% of total runtime. A naive model that predicts 'healthy' 100% of the time achieves 96.6% accuracy but catches zero failures. Therefore, Recall, F1-Score, and Precision-Recall AUC (PR-AUC) are the only honest indicators of maintenance efficacy."*

### Q3: *"How would this transition to physical factory hardware?"*
**Answer:**
> *"The firmware in `firmware/esp32_wokwi/sketch.ino` is written in standard C++/Arduino for the ESP32 with DHT22 and MPU6050 pins defined. By flashing it to an ESP32 board and plugging it into USB, `scripts/serial_bridge.py` forwards genuine telemetry lines directly into our Flask endpoint with zero codebase modifications."*
