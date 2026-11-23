import argparse
import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import config

SAMPLE_MOCK_LINES = [
    '{"machine_id": "M1", "temperature": 61.2, "vibration": 0.34, "power": 3.52, "source": "esp32-sim"}',
    '{"machine_id": "M2", "temperature": 59.8, "vibration": 0.38, "power": 3.65, "source": "esp32-sim"}',
    '{"machine_id": "M3", "temperature": 72.4, "vibration": 0.85, "power": 3.90, "source": "esp32-sim"}',
    '{"machine_id": "M1", "temperature": 60.8, "vibration": 0.32, "power": 3.48, "source": "esp32-sim"}',
    '{"machine_id": "M3", "temperature": 86.5, "vibration": 1.62, "power": 4.25, "source": "esp32-sim"}'
]

def forward_payload(payload: dict):
    """Sends payload to Flask /sensor-data endpoint."""
    try:
        r = requests.post(f"{config.API_URL}/sensor-data", json=payload, timeout=2.0)
        if r.status_code == 201:
            print(f"[Bridge -> API OK] {payload['machine_id']} ({payload['temperature']}C, {payload['vibration']}g) -> Status: {r.json().get('status')}")
        else:
            print(f"[Bridge -> API ERR] Status {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[Bridge -> API Offline] {e}")

def run_serial_bridge(port: str, baud: int, default_machine_id: str, mock: bool):
    print("=" * 65)
    print("      ESP32 USB SERIAL TO DIGITAL TWIN API BRIDGE             ")
    print("=" * 65)

    if mock:
        print("[Bridge] Running in --mock mode. Streaming sample firmware telemetry lines...")
        idx = 0
        while True:
            raw_line = SAMPLE_MOCK_LINES[idx % len(SAMPLE_MOCK_LINES)]
            idx += 1
            try:
                data = json.loads(raw_line)
                if default_machine_id:
                    data["machine_id"] = default_machine_id
                data["timestamp"] = time.time()
                data["source"] = "esp32-sim"
                forward_payload(data)
            except Exception as e:
                print(f"[Bridge] Parse error: {e}")
            time.sleep(1.0)
    else:
        try:
            import serial
        except ImportError:
            print("[Bridge Error] pyserial is required for hardware USB reading.")
            print("Run: pip install pyserial")
            return

        print(f"[Bridge] Connecting to serial port {port} at {baud} baud...")
        try:
            ser = serial.Serial(port, baud, timeout=2.0)
            time.sleep(2.0)
            print("[Bridge] Serial connection established. Listening for JSON stream...")
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        if default_machine_id:
                            data["machine_id"] = default_machine_id
                        data["timestamp"] = time.time()
                        data["source"] = "esp32-real"
                        forward_payload(data)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"[Bridge Error] Failed to open port {port}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP32 Serial to Flask API Bridge")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument("--machine-id", type=str, default="", help="Override machine_id")
    parser.add_argument("--mock", action="store_true", help="Run with simulated mock stream")
    args = parser.parse_args()

    run_serial_bridge(args.port, args.baud, args.machine_id, args.mock)
