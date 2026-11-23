import time
from flask import Flask, request, jsonify
from config import API_HOST, API_PORT, determine_status
import db

app = Flask(__name__)

# Initialize database on startup
db.init_db()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    if not request.is_json:
        return jsonify({"ok": False, "error": "Request body must be JSON"}), 400
    
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "Invalid payload format"}), 400
    
    machine_id = data.get("machine_id")
    temperature = data.get("temperature")
    vibration = data.get("vibration")
    power = data.get("power")
    timestamp = data.get("timestamp", time.time())
    source = data.get("source", "simulator")
    if source not in ["simulator", "esp32-sim", "esp32-real"]:
        source = "simulator"
    
    # Validation
    if not machine_id or not isinstance(machine_id, str):
        return jsonify({"ok": False, "error": "machine_id is required and must be a string"}), 400
    
    try:
        temperature = float(temperature)
        vibration = float(vibration)
        power = float(power)
        timestamp = float(timestamp)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "temperature, vibration, power, timestamp must be numeric"}), 400

    status = determine_status(temperature, vibration, power)
    row_id = db.insert_reading(machine_id, timestamp, temperature, vibration, power, status, source=source)
    
    return jsonify({"ok": True, "id": row_id, "status": status, "source": source}), 201

@app.route("/latest", methods=["GET"])
def latest():
    latest_rows = db.get_latest_all()
    return jsonify({"ok": True, "data": latest_rows}), 200

@app.route("/history/<machine_id>", methods=["GET"])
def history(machine_id):
    n = request.args.get("n", default=200, type=int)
    if n <= 0:
        n = 200
    df = db.get_recent(machine_id, n=n)
    return jsonify({"ok": True, "machine_id": machine_id, "data": df.to_dict(orient="records")}), 200

@app.route("/events", methods=["GET"])
def get_events():
    limit = request.args.get("limit", default=50, type=int)
    events = db.get_recent_events(limit=limit)
    return jsonify({"ok": True, "data": events}), 200

@app.route("/risk", methods=["GET"])
def get_risk():
    risks = db.get_latest_risk_scores()
    return jsonify({"ok": True, "data": risks}), 200

@app.route("/schedule/latest", methods=["GET"])
def get_latest_schedule():
    sched = db.get_latest_schedule()
    return jsonify({"ok": True, "data": sched}), 200

@app.route("/schedule/history", methods=["GET"])
def get_schedule_history():
    limit = request.args.get("limit", default=20, type=int)
    history = db.get_schedule_history(limit=limit)
    return jsonify({"ok": True, "data": history}), 200

@app.route("/alerts", methods=["GET"])
def get_alerts():
    limit = request.args.get("limit", default=30, type=int)
    alerts = db.get_recent_alerts(limit=limit)
    return jsonify({"ok": True, "data": alerts}), 200

if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=False, threaded=True)
